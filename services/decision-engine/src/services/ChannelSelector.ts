import { 
  DecisionContext, 
  Strategy, 
  MLPrediction, 
  Channel, 
  InterventionType,
  ChannelPerformance 
} from '../types';
import { logger, database, eventsCache } from '@userwhisperer/shared';

export class ChannelSelector {
  private channelCosts: Record<Channel, number>;
  private performanceCache: Map<string, ChannelPerformance[]> = new Map();

  constructor(channelCosts: Record<Channel, number>) {
    this.channelCosts = channelCosts;
  }

  /**
   * Select the optimal channel for communication
   */
  public async selectChannel(
    context: DecisionContext,
    strategy: Strategy,
    mlPrediction: MLPrediction
  ): Promise<Channel> {
    try {
      logger.debug('Selecting optimal channel', {
        userId: context.user_id,
        interventionType: strategy.intervention_type,
        urgency: strategy.urgency
      });

      // Get available channels for user
      const availableChannels = await this.getAvailableChannels(context.user_id);

      if (availableChannels.length === 0) {
        throw new Error('No channels available for user');
      }

      // Score each available channel
      const channelScores = await Promise.all(
        availableChannels.map(async (channel) => {
          const score = await this.scoreChannel(
            channel,
            context,
            strategy,
            mlPrediction
          );
          return { channel, score };
        })
      );

      // Sort by score (highest first)
      channelScores.sort((a, b) => b.score - a.score);

      const selectedChannel = channelScores[0].channel;

      // Log selection reasoning
      await this.logChannelSelection(
        context.user_id,
        selectedChannel,
        channelScores,
        strategy
      );

      return selectedChannel;

    } catch (error) {
      logger.error('Channel selection failed', {
        userId: context.user_id,
        error: error instanceof Error ? error.message : String(error)
      });

      // Fallback to user's preferred channel or email
      return this.getFallbackChannel(context);
    }
  }

  /**
   * Score a channel for suitability based on multiple factors
   */
  private async scoreChannel(
    channel: Channel,
    context: DecisionContext,
    strategy: Strategy,
    mlPrediction: MLPrediction
  ): Promise<number> {
    let score = 0.0;

    // 1. Historical performance (40% weight)
    const historicalPerformance = await this.getChannelPerformance(
      context.user_id,
      channel
    );
    score += historicalPerformance * 0.4;

    // 2. Channel-strategy fit (30% weight)
    const strategyFit = this.getStrategyChannelFit(strategy, channel);
    score += strategyFit * 0.3;

    // 3. User preference (20% weight)
    const userPreference = context.user_profile.channel_preferences[channel] || 0.5;
    score += userPreference * 0.2;

    // 4. Cost efficiency (10% weight)
    const costScore = this.calculateCostScore(channel);
    score += costScore * 0.1;

    // Apply urgency modifiers
    score = this.applyUrgencyModifiers(score, strategy, channel);

    // Apply time-of-day modifiers
    score = this.applyTimingModifiers(score, channel, context.current_time);

    // Apply availability modifiers
    score = await this.applyAvailabilityModifiers(score, channel, context);

    return Math.max(0, Math.min(1, score)); // Clamp between 0 and 1
  }

  /**
   * Calculate how well a channel fits a strategy
   */
  private getStrategyChannelFit(strategy: Strategy, channel: Channel): number {
    const fitMatrix: Record<InterventionType, Record<Channel, number>> = {
      [InterventionType.ONBOARDING]: {
        [Channel.EMAIL]: 0.9,
        [Channel.PUSH]: 0.6,
        [Channel.SMS]: 0.5,
        [Channel.IN_APP]: 1.0,
        [Channel.WEBHOOK]: 0.1
      },
      [InterventionType.RETENTION]: {
        [Channel.EMAIL]: 0.8,
        [Channel.PUSH]: 0.7,
        [Channel.SMS]: 0.6,
        [Channel.IN_APP]: 0.9,
        [Channel.WEBHOOK]: 0.3
      },
      [InterventionType.MONETIZATION]: {
        [Channel.EMAIL]: 0.9,
        [Channel.PUSH]: 0.5,
        [Channel.SMS]: 0.4,
        [Channel.IN_APP]: 0.8,
        [Channel.WEBHOOK]: 0.2
      },
      [InterventionType.SUPPORT]: {
        [Channel.EMAIL]: 0.7,
        [Channel.PUSH]: 0.8,
        [Channel.SMS]: 0.9,
        [Channel.IN_APP]: 0.6,
        [Channel.WEBHOOK]: 0.4
      },
      [InterventionType.REACTIVATION]: {
        [Channel.EMAIL]: 0.8,
        [Channel.PUSH]: 0.6,
        [Channel.SMS]: 0.7,
        [Channel.IN_APP]: 0.5,
        [Channel.WEBHOOK]: 0.3
      },
      [InterventionType.CELEBRATION]: {
        [Channel.EMAIL]: 0.6,
        [Channel.PUSH]: 0.9,
        [Channel.SMS]: 0.5,
        [Channel.IN_APP]: 0.8,
        [Channel.WEBHOOK]: 0.4
      },
      [InterventionType.EDUCATION]: {
        [Channel.EMAIL]: 0.9,
        [Channel.PUSH]: 0.4,
        [Channel.SMS]: 0.3,
        [Channel.IN_APP]: 0.7,
        [Channel.WEBHOOK]: 0.2
      }
    };

    return fitMatrix[strategy.intervention_type]?.[channel] || 0.5;
  }

  /**
   * Get historical performance for a channel-user combination
   */
  private async getChannelPerformance(
    userId: string,
    channel: Channel
  ): Promise<number> {
    try {
      // Check cache first
      const cacheKey = `channel_perf:${userId}:${channel}`;
      const cached = await eventsCache.get(cacheKey);
      
      if (cached) {
        return parseFloat(cached);
      }

      // Query database for performance metrics
      const query = `
        SELECT 
          COUNT(*) as total_sent,
          COUNT(CASE WHEN opened_at IS NOT NULL THEN 1 END) as opened,
          COUNT(CASE WHEN clicked_at IS NOT NULL THEN 1 END) as clicked,
          COUNT(CASE WHEN unsubscribed_at IS NOT NULL THEN 1 END) as unsubscribed,
          COUNT(CASE WHEN bounced_at IS NOT NULL THEN 1 END) as bounced
        FROM message_history
        WHERE user_id = $1 AND channel = $2
        AND sent_at > NOW() - INTERVAL '90 days'
      `;

      const result = await database.query(query, [userId, channel]);
      
      if (!result.rows.length || result.rows[0].total_sent === 0) {
        // Use global averages for this channel
        return await this.getGlobalChannelPerformance(channel);
      }

      const metrics = result.rows[0];
      const openRate = parseInt(metrics.opened) / parseInt(metrics.total_sent);
      const clickRate = parseInt(metrics.clicked) / parseInt(metrics.total_sent);
      const unsubscribeRate = parseInt(metrics.unsubscribed) / parseInt(metrics.total_sent);
      const bounceRate = parseInt(metrics.bounced) / parseInt(metrics.total_sent);

      // Calculate composite performance score
      let performance = 0;
      performance += openRate * 0.4;        // Open rate weight: 40%
      performance += clickRate * 0.4;       // Click rate weight: 40%
      performance -= unsubscribeRate * 0.1; // Unsubscribe penalty: 10%
      performance -= bounceRate * 0.1;      // Bounce penalty: 10%

      performance = Math.max(0, Math.min(1, performance));

      // Cache for 1 hour
      await eventsCache.setex(cacheKey, 3600, performance.toString());

      return performance;

    } catch (error) {
      logger.error('Failed to get channel performance', {
        userId,
        channel,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return channel-specific default
      return this.getChannelDefaultPerformance(channel);
    }
  }

  /**
   * Get global performance averages for a channel
   */
  private async getGlobalChannelPerformance(channel: Channel): Promise<number> {
    try {
      const cacheKey = `global_perf:${channel}`;
      const cached = await eventsCache.get(cacheKey);
      
      if (cached) {
        return parseFloat(cached);
      }

      const query = `
        SELECT 
          COUNT(*) as total_sent,
          COUNT(CASE WHEN opened_at IS NOT NULL THEN 1 END) as opened,
          COUNT(CASE WHEN clicked_at IS NOT NULL THEN 1 END) as clicked
        FROM message_history
        WHERE channel = $1
        AND sent_at > NOW() - INTERVAL '30 days'
      `;

      const result = await database.query(query, [channel]);
      
      if (!result.rows.length || result.rows[0].total_sent === 0) {
        return this.getChannelDefaultPerformance(channel);
      }

      const metrics = result.rows[0];
      const openRate = parseInt(metrics.opened) / parseInt(metrics.total_sent);
      const clickRate = parseInt(metrics.clicked) / parseInt(metrics.total_sent);

      const performance = (openRate * 0.6) + (clickRate * 0.4);

      // Cache for 4 hours
      await eventsCache.setex(cacheKey, 14400, performance.toString());

      return performance;

    } catch (error) {
      logger.error('Failed to get global channel performance', {
        channel,
        error: error instanceof Error ? error.message : String(error)
      });

      return this.getChannelDefaultPerformance(channel);
    }
  }

  /**
   * Calculate cost efficiency score for a channel
   */
  private calculateCostScore(channel: Channel): number {
    const maxCost = Math.max(...Object.values(this.channelCosts));
    const channelCost = this.channelCosts[channel];
    
    // Higher score for lower cost (inverted and normalized)
    return 1.0 - (channelCost / maxCost);
  }

  /**
   * Apply urgency-based score modifiers
   */
  private applyUrgencyModifiers(
    score: number, 
    strategy: Strategy, 
    channel: Channel
  ): number {
    switch (strategy.urgency) {
      case 'critical':
        // Prefer immediate channels for critical messages
        if (channel === Channel.SMS || channel === Channel.PUSH) {
          return score * 1.5;
        } else if (channel === Channel.EMAIL) {
          return score * 0.8;
        }
        break;
        
      case 'high':
        // Slight preference for faster channels
        if (channel === Channel.SMS || channel === Channel.PUSH) {
          return score * 1.2;
        }
        break;
        
      case 'low':
        // Prefer less intrusive channels
        if (channel === Channel.EMAIL) {
          return score * 1.2;
        } else if (channel === Channel.SMS) {
          return score * 0.9;
        }
        break;
    }

    return score;
  }

  /**
   * Apply timing-based score modifiers
   */
  private applyTimingModifiers(
    score: number, 
    channel: Channel, 
    currentTime: Date
  ): number {
    const hour = currentTime.getHours();
    const isWeekend = currentTime.getDay() === 0 || currentTime.getDay() === 6;

    // Channel-specific timing preferences
    switch (channel) {
      case Channel.EMAIL:
        // Email performs better during business hours on weekdays
        if (!isWeekend && (hour >= 9 && hour <= 17)) {
          return score * 1.1;
        } else if (hour >= 22 || hour <= 6) {
          return score * 0.8; // Avoid very late/early hours
        }
        break;

      case Channel.SMS:
        // SMS should avoid very late and very early hours
        if (hour >= 22 || hour <= 8) {
          return score * 0.6;
        } else if (hour >= 9 && hour <= 21) {
          return score * 1.0;
        }
        break;

      case Channel.PUSH:
        // Push notifications more flexible but avoid very late hours
        if (hour >= 23 || hour <= 7) {
          return score * 0.7;
        }
        break;

      case Channel.IN_APP:
        // In-app messages only work when user is active
        // Slightly prefer active hours
        if (hour >= 8 && hour <= 22) {
          return score * 1.1;
        } else {
          return score * 0.3; // Much lower score during inactive hours
        }
        break;
    }

    return score;
  }

  /**
   * Apply availability-based score modifiers
   */
  private async applyAvailabilityModifiers(
    score: number,
    channel: Channel,
    context: DecisionContext
  ): Promise<number> {
    // Check if user has required contact info for the channel
    switch (channel) {
      case Channel.EMAIL:
        if (!context.user_profile.email) {
          return 0; // Can't send email without email address
        }
        break;

      case Channel.SMS:
        if (!context.user_profile.metadata?.phone_number) {
          return 0; // Can't send SMS without phone number
        }
        break;

      case Channel.PUSH:
        // Check if user has push token
        if (!context.user_profile.metadata?.push_token) {
          return 0;
        }
        break;

      case Channel.IN_APP:
        // Check if user is currently active
        const hoursSinceLastActive = this.getHoursSinceLastActive(context);
        if (hoursSinceLastActive > 24) {
          return score * 0.3; // Reduce score if user not recently active
        }
        break;
    }

    return score;
  }

  /**
   * Get available channels for a user
   */
  private async getAvailableChannels(userId: string): Promise<Channel[]> {
    try {
      // In a real implementation, this would check:
      // 1. User's contact information
      // 2. User's channel preferences
      // 3. User's opt-in status for each channel
      // 4. Any channel-specific blocks or failures

      // For now, return all channels as potentially available
      return [Channel.EMAIL, Channel.SMS, Channel.PUSH, Channel.IN_APP];

    } catch (error) {
      logger.error('Failed to get available channels', {
        userId,
        error: error instanceof Error ? error.message : String(error)
      });

      // Conservative fallback
      return [Channel.EMAIL];
    }
  }

  /**
   * Log channel selection reasoning for analysis
   */
  private async logChannelSelection(
    userId: string,
    selectedChannel: Channel,
    channelScores: Array<{ channel: Channel; score: number }>,
    strategy: Strategy
  ): Promise<void> {
    try {
      logger.info('Channel selected', {
        userId,
        selectedChannel,
        interventionType: strategy.intervention_type,
        urgency: strategy.urgency,
        channelScores: channelScores.map(cs => ({
          channel: cs.channel,
          score: Math.round(cs.score * 1000) / 1000 // Round to 3 decimal places
        })),
        selectionReason: this.buildSelectionReason(selectedChannel, channelScores, strategy)
      });

      // Store selection for ML training data
      const selectionData = {
        user_id: userId,
        selected_channel: selectedChannel,
        channel_scores: channelScores,
        strategy_type: strategy.intervention_type,
        urgency: strategy.urgency,
        timestamp: new Date().toISOString()
      };

      // Cache selection for analysis (expires in 7 days)
      await eventsCache.setex(
        `channel_selection:${userId}:${Date.now()}`,
        604800, // 7 days
        JSON.stringify(selectionData)
      );

    } catch (error) {
      logger.error('Failed to log channel selection', {
        userId,
        selectedChannel,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  /**
   * Build human-readable selection reasoning
   */
  private buildSelectionReason(
    selectedChannel: Channel,
    channelScores: Array<{ channel: Channel; score: number }>,
    strategy: Strategy
  ): string {
    const selectedScore = channelScores.find(cs => cs.channel === selectedChannel)?.score || 0;
    const reasons = [];

    // Primary reason based on score
    if (selectedScore > 0.8) {
      reasons.push('High overall performance');
    } else if (selectedScore > 0.6) {
      reasons.push('Good performance match');
    } else {
      reasons.push('Best available option');
    }

    // Strategy-specific reasons
    const strategyFit = this.getStrategyChannelFit(strategy, selectedChannel);
    if (strategyFit > 0.8) {
      reasons.push(`Excellent fit for ${strategy.intervention_type}`);
    }

    // Urgency reasons
    if (strategy.urgency === 'critical' && 
        (selectedChannel === Channel.SMS || selectedChannel === Channel.PUSH)) {
      reasons.push('Immediate delivery required');
    }

    return reasons.join('; ');
  }

  /**
   * Get fallback channel when selection fails
   */
  private getFallbackChannel(context: DecisionContext): Channel {
    // Try user's highest preference
    const preferences = context.user_profile.channel_preferences;
    const sortedPreferences = Object.entries(preferences)
      .sort(([,a], [,b]) => b - a);

    if (sortedPreferences.length > 0) {
      return sortedPreferences[0][0] as Channel;
    }

    // Default to email
    return Channel.EMAIL;
  }

  /**
   * Get default performance scores for channels
   */
  private getChannelDefaultPerformance(channel: Channel): number {
    const defaults: Record<Channel, number> = {
      [Channel.EMAIL]: 0.25,    // 25% engagement
      [Channel.SMS]: 0.45,      // 45% engagement
      [Channel.PUSH]: 0.35,     // 35% engagement
      [Channel.IN_APP]: 0.60,   // 60% engagement
      [Channel.WEBHOOK]: 0.90   // 90% delivery success
    };

    return defaults[channel] || 0.3;
  }

  /**
   * Calculate hours since user was last active
   */
  private getHoursSinceLastActive(context: DecisionContext): number {
    if (!context.user_profile.last_active_at) {
      return 999; // Very high number if never active
    }

    const now = new Date();
    const lastActive = new Date(context.user_profile.last_active_at);
    return (now.getTime() - lastActive.getTime()) / (1000 * 60 * 60);
  }

  /**
   * Health check for channel selector
   */
  public async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    const details: any = {};
    let healthy = true;

    try {
      // Check if we can access performance data
      details.performance_cache_size = this.performanceCache.size;
      details.channel_costs_configured = Object.keys(this.channelCosts).length;

      // Test database connectivity for performance queries
      const testQuery = 'SELECT 1 as test';
      await database.query(testQuery);
      details.database_connection = 'healthy';

      // Test cache connectivity
      await eventsCache.ping();
      details.cache_connection = 'healthy';

    } catch (error) {
      healthy = false;
      details.error = error instanceof Error ? error.message : String(error);
    }

    return { healthy, details };
  }
} 