import { 
  DecisionContext, 
  Strategy, 
  Channel, 
  TimingOptimization 
} from '../types';
import { logger, eventsCache } from '@userwhisperer/shared';
import { addHours, addDays, format, parseISO } from 'date-fns';

export class TimingOptimizer {
  private defaultTimezone: string;
  private timingModel: any = null; // Placeholder for ML model

  constructor(defaultTimezone: string = 'UTC') {
    this.defaultTimezone = defaultTimezone;
  }

  /**
   * Determine optimal send time for a message
   */
  public async optimizeTiming(
    context: DecisionContext,
    strategy: Strategy,
    channel: Channel
  ): Promise<Date> {
    try {
      logger.debug('Optimizing message timing', {
        userId: context.user_id,
        channel,
        urgency: strategy.urgency,
        currentTime: context.current_time
      });

      // Get user's timezone
      const userTimezone = await this.getUserTimezone(context.user_id);

      // Convert current time to user's timezone
      const userCurrentTime = this.convertToUserTime(
        context.current_time,
        userTimezone
      );

      // Handle urgency levels
      if (strategy.urgency === 'critical') {
        // Send immediately for critical messages
        return context.current_time;
      } else if (strategy.urgency === 'high') {
        // Send within next hour, but optimize within that window
        return await this.findNextAvailableSlot(
          context,
          userCurrentTime,
          1, // hours ahead
          channel
        );
      }

      // For medium/low urgency, find optimal timing
      const optimalTime = await this.findOptimalTime(
        context,
        strategy,
        channel,
        userTimezone
      );

      logger.info('Optimal timing calculated', {
        userId: context.user_id,
        channel,
        originalTime: context.current_time,
        optimizedTime: optimalTime,
        delayHours: (optimalTime.getTime() - context.current_time.getTime()) / (1000 * 60 * 60)
      });

      return optimalTime;

    } catch (error) {
      logger.error('Timing optimization failed', {
        userId: context.user_id,
        channel,
        error: error instanceof Error ? error.message : String(error)
      });

      // Fallback to immediate or next available slot
      return this.getFallbackTime(context, strategy);
    }
  }

  /**
   * Find optimal send time using ML and heuristics
   */
  private async findOptimalTime(
    context: DecisionContext,
    strategy: Strategy,
    channel: Channel,
    userTimezone: string
  ): Promise<Date> {
    // Get user's optimal hours from profile or predict them
    let optimalHours = context.user_profile.optimal_send_hours || [];
    
    if (optimalHours.length === 0) {
      optimalHours = await this.predictOptimalHours(context, channel);
    }

    // Get next 48 hours of possible send times
    const possibleTimes = this.generatePossibleTimes(
      context.current_time,
      48, // hours ahead
      userTimezone
    );

    // Score each time slot
    const timeScores = await Promise.all(
      possibleTimes.map(async (timeSlot) => {
        const score = await this.scoreTimeSlot(
          timeSlot,
          context,
          strategy,
          channel,
          optimalHours,
          userTimezone
        );
        return { time: timeSlot, score };
      })
    );

    // Filter out times that don't meet minimum requirements
    const validTimes = timeScores.filter(ts => ts.score > 0.3);

    if (validTimes.length === 0) {
      // If no valid times, return next reasonable slot
      return this.getNextReasonableTime(context.current_time, channel);
    }

    // Sort by score and select best time
    validTimes.sort((a, b) => b.score - a.score);
    return validTimes[0].time;
  }

  /**
   * Score a potential send time based on multiple factors
   */
  private async scoreTimeSlot(
    timeSlot: Date,
    context: DecisionContext,
    strategy: Strategy,
    channel: Channel,
    optimalHours: number[],
    userTimezone: string
  ): Promise<number> {
    let score = 0.0;

    // 1. Hour of day score (30% weight)
    const hourScore = this.getHourScore(timeSlot, optimalHours);
    score += hourScore * 0.3;

    // 2. Day of week score (20% weight)
    const dayScore = this.getDayOfWeekScore(timeSlot, channel);
    score += dayScore * 0.2;

    // 3. Avoid busy times (20% weight)
    const busyTimeScore = this.getBusyTimeScore(timeSlot);
    score += busyTimeScore * 0.2;

    // 4. Message spacing (20% weight)
    const spacingScore = await this.getMessageSpacingScore(
      timeSlot,
      context.user_id,
      context.message_history
    );
    score += spacingScore * 0.2;

    // 5. Channel-specific timing (10% weight)
    const channelTimingScore = this.getChannelTimingScore(channel, timeSlot);
    score += channelTimingScore * 0.1;

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Score based on hour of day and user's optimal hours
   */
  private getHourScore(timeSlot: Date, optimalHours: number[]): number {
    const hour = timeSlot.getHours();

    if (optimalHours.includes(hour)) {
      return 1.0; // Perfect match
    }

    // Score based on proximity to optimal hours
    const distances = optimalHours.map(optimalHour => {
      let distance = Math.abs(hour - optimalHour);
      // Handle wraparound (e.g., 23 and 1 are close)
      distance = Math.min(distance, 24 - distance);
      return distance;
    });

    const minDistance = Math.min(...distances);

    if (minDistance <= 1) return 0.8; // Within 1 hour
    if (minDistance <= 2) return 0.6; // Within 2 hours
    if (minDistance <= 3) return 0.4; // Within 3 hours
    
    return 0.2; // Far from optimal
  }

  /**
   * Score based on day of week and channel preferences
   */
  private getDayOfWeekScore(timeSlot: Date, channel: Channel): number {
    const dayOfWeek = timeSlot.getDay(); // 0 = Sunday, 6 = Saturday
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;

    switch (channel) {
      case Channel.EMAIL:
        // Email performs better on weekdays for business content
        return isWeekend ? 0.5 : 1.0;

      case Channel.SMS:
        // SMS more flexible across days
        return isWeekend ? 0.8 : 1.0;

      case Channel.PUSH:
        // Push notifications flexible
        return 0.9;

      case Channel.IN_APP:
        // In-app depends on user activity patterns
        return isWeekend ? 0.7 : 0.9;

      default:
        return 0.8;
    }
  }

  /**
   * Check if time slot avoids busy periods
   */
  private getBusyTimeScore(timeSlot: Date): number {
    const hour = timeSlot.getHours();
    const dayOfWeek = timeSlot.getDay();
    
    // Universal busy times
    if (hour < 6 || hour > 22) {
      return 0.2; // Very late/early hours
    }

    // Weekday-specific busy times
    if (dayOfWeek >= 1 && dayOfWeek <= 5) { // Monday-Friday
      if ((hour >= 7 && hour <= 9) || // Morning commute
          (hour >= 17 && hour <= 18)) { // Evening commute
        return 0.6; // Moderately busy
      }
      
      if (hour >= 12 && hour <= 13) { // Lunch time
        return 0.7; // Slightly busy
      }
    }

    // Good times
    if (hour >= 10 && hour <= 11) return 1.0; // Mid-morning
    if (hour >= 14 && hour <= 16) return 1.0; // Mid-afternoon
    if (hour >= 19 && hour <= 21) return 0.9; // Evening

    return 0.8; // Default good time
  }

  /**
   * Score based on spacing from previous messages
   */
  private async getMessageSpacingScore(
    timeSlot: Date,
    userId: string,
    messageHistory: any[]
  ): Promise<number> {
    if (messageHistory.length === 0) {
      return 1.0; // Perfect if no previous messages
    }

    // Find most recent message
    const lastMessage = messageHistory.reduce((latest, msg) => {
      return new Date(msg.sent_at) > new Date(latest.sent_at) ? msg : latest;
    });

    const hoursSinceLastMessage = 
      (timeSlot.getTime() - new Date(lastMessage.sent_at).getTime()) / (1000 * 60 * 60);

    // Optimal spacing scores
    if (hoursSinceLastMessage >= 24) return 1.0;  // 24+ hours is perfect
    if (hoursSinceLastMessage >= 12) return 0.8;  // 12+ hours is good
    if (hoursSinceLastMessage >= 6) return 0.6;   // 6+ hours is acceptable
    if (hoursSinceLastMessage >= 3) return 0.4;   // 3+ hours is poor
    
    return 0.1; // Less than 3 hours is very poor
  }

  /**
   * Channel-specific timing preferences
   */
  private getChannelTimingScore(channel: Channel, timeSlot: Date): number {
    const hour = timeSlot.getHours();

    switch (channel) {
      case Channel.EMAIL:
        // Email optimal times: 10am, 2pm, 8pm
        if ([10, 14, 20].includes(hour)) return 1.0;
        if ([9, 11, 13, 15, 19, 21].includes(hour)) return 0.8;
        if (hour >= 6 && hour <= 22) return 0.6;
        return 0.3;

      case Channel.SMS:
        // SMS optimal: 10am-8pm, avoid very early/late
        if (hour >= 10 && hour <= 20) return 1.0;
        if (hour >= 9 && hour <= 21) return 0.7;
        if (hour >= 8 && hour <= 22) return 0.5;
        return 0.2;

      case Channel.PUSH:
        // Push optimal: 12pm, 6pm, 8pm
        if ([12, 18, 20].includes(hour)) return 1.0;
        if ([11, 13, 17, 19, 21].includes(hour)) return 0.8;
        if (hour >= 8 && hour <= 22) return 0.6;
        return 0.3;

      case Channel.IN_APP:
        // In-app: depends on user activity, prefer active hours
        if (hour >= 9 && hour <= 21) return 0.9;
        if (hour >= 8 && hour <= 22) return 0.6;
        return 0.2;

      default:
        return 0.7;
    }
  }

  /**
   * Predict optimal hours for a user based on their activity
   */
  private async predictOptimalHours(
    context: DecisionContext,
    channel: Channel
  ): Promise<number[]> {
    try {
      // Check cache first
      const cacheKey = `optimal_hours:${context.user_id}:${channel}`;
      const cached = await eventsCache.get(cacheKey);
      
      if (cached) {
        return JSON.parse(cached);
      }

      // Use ML model if available
      if (this.timingModel) {
        const prediction = await this.mlPredictOptimalHours(context, channel);
        if (prediction) {
          // Cache for 7 days
          await eventsCache.setex(cacheKey, 604800, JSON.stringify(prediction));
          return prediction;
        }
      }

      // Fallback to heuristic prediction
      const heuristicHours = this.heuristicPredictOptimalHours(context, channel);
      
      // Cache for 24 hours (shorter since it's heuristic)
      await eventsCache.setex(cacheKey, 86400, JSON.stringify(heuristicHours));
      
      return heuristicHours;

    } catch (error) {
      logger.error('Failed to predict optimal hours', {
        userId: context.user_id,
        channel,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return default optimal hours
      return this.getDefaultOptimalHours(channel);
    }
  }

  /**
   * Heuristic-based optimal hours prediction
   */
  private heuristicPredictOptimalHours(
    context: DecisionContext,
    channel: Channel
  ): number[] {
    const userProfile = context.user_profile;
    
    // Base hours by channel
    let baseHours: number[] = [];
    
    switch (channel) {
      case Channel.EMAIL:
        baseHours = [10, 14, 20]; // 10am, 2pm, 8pm
        break;
      case Channel.SMS:
        baseHours = [12, 18]; // 12pm, 6pm
        break;
      case Channel.PUSH:
        baseHours = [12, 18, 20]; // 12pm, 6pm, 8pm
        break;
      case Channel.IN_APP:
        baseHours = [9, 14, 19]; // 9am, 2pm, 7pm
        break;
      default:
        baseHours = [10, 15, 20];
    }

    // Adjust based on user characteristics
    if (userProfile.lifecycle_stage === 'new') {
      // New users might be more active in evenings
      baseHours = baseHours.map(h => h >= 12 ? h : h + 2);
    }

    if (userProfile.subscription_plan === 'enterprise') {
      // Business users prefer business hours
      baseHours = baseHours.filter(h => h >= 9 && h <= 17);
      if (baseHours.length < 2) {
        baseHours = [10, 14]; // Fallback business hours
      }
    }

    // Ensure we have 2-3 optimal hours
    if (baseHours.length < 2) {
      baseHours.push(...this.getDefaultOptimalHours(channel));
    }

    return [...new Set(baseHours)].slice(0, 3); // Remove duplicates, max 3 hours
  }

  /**
   * Get default optimal hours for a channel
   */
  private getDefaultOptimalHours(channel: Channel): number[] {
    const defaults: Record<Channel, number[]> = {
      [Channel.EMAIL]: [10, 14, 20],
      [Channel.SMS]: [12, 18],
      [Channel.PUSH]: [12, 18, 20],
      [Channel.IN_APP]: [9, 14, 19],
      [Channel.WEBHOOK]: [0] // Webhooks can be sent anytime
    };

    return defaults[channel] || [10, 15, 20];
  }

  /**
   * Generate possible send times within a time window
   */
  private generatePossibleTimes(
    startTime: Date,
    hoursAhead: number,
    userTimezone: string
  ): Date[] {
    const times: Date[] = [];
    const endTime = addHours(startTime, hoursAhead);
    
    // Generate hourly slots
    let currentTime = new Date(startTime);
    
    while (currentTime <= endTime) {
      times.push(new Date(currentTime));
      currentTime = addHours(currentTime, 1);
    }

    return times;
  }

  /**
   * Find next available slot within a time window
   */
  private async findNextAvailableSlot(
    context: DecisionContext,
    userCurrentTime: Date,
    hoursAhead: number,
    channel: Channel
  ): Promise<Date> {
    const possibleTimes = this.generatePossibleTimes(
      context.current_time,
      hoursAhead,
      context.user_profile.timezone || this.defaultTimezone
    );

    // Score each slot and find the best available one
    for (const timeSlot of possibleTimes) {
      const score = await this.scoreTimeSlot(
        timeSlot,
        context,
        { urgency: 'high' } as Strategy,
        channel,
        context.user_profile.optimal_send_hours || [9, 14, 20],
        context.user_profile.timezone || this.defaultTimezone
      );

      if (score >= 0.5) {
        return timeSlot;
      }
    }

    // If no good slot found, return the first one
    return possibleTimes[0] || context.current_time;
  }

  /**
   * Get user's timezone
   */
  private async getUserTimezone(userId: string): Promise<string> {
    try {
      // This would typically query the user profile
      // For now, return the default timezone
      return this.defaultTimezone;
    } catch (error) {
      return this.defaultTimezone;
    }
  }

  /**
   * Convert time to user's timezone
   */
  private convertToUserTime(time: Date, userTimezone: string): Date {
    // For now, just return the original time
    // In a real implementation, this would handle timezone conversion
    return time;
  }

  /**
   * Get fallback time when optimization fails
   */
  private getFallbackTime(context: DecisionContext, strategy: Strategy): Date {
    if (strategy.urgency === 'critical') {
      return context.current_time;
    }

    // Add 1 hour for high urgency, 6 hours for others
    const delayHours = strategy.urgency === 'high' ? 1 : 6;
    return addHours(context.current_time, delayHours);
  }

  /**
   * Get next reasonable time for a channel
   */
  private getNextReasonableTime(currentTime: Date, channel: Channel): Date {
    const hour = currentTime.getHours();
    const optimalHours = this.getDefaultOptimalHours(channel);

    // Find next optimal hour
    const nextOptimalHour = optimalHours.find(h => h > hour);
    
    if (nextOptimalHour) {
      // Same day
      const nextTime = new Date(currentTime);
      nextTime.setHours(nextOptimalHour, 0, 0, 0);
      return nextTime;
    } else {
      // Next day, first optimal hour
      const nextTime = addDays(currentTime, 1);
      nextTime.setHours(optimalHours[0], 0, 0, 0);
      return nextTime;
    }
  }

  /**
   * ML prediction placeholder
   */
  private async mlPredictOptimalHours(
    context: DecisionContext,
    channel: Channel
  ): Promise<number[] | null> {
    // Placeholder for ML model prediction
    // This would use user's historical engagement data to predict optimal hours
    return null;
  }

  /**
   * Get timing recommendations for a user
   */
  public async getTimingRecommendations(
    userId: string,
    channel: Channel
  ): Promise<TimingOptimization> {
    try {
      const cacheKey = `timing_rec:${userId}:${channel}`;
      const cached = await eventsCache.get(cacheKey);
      
      if (cached) {
        return JSON.parse(cached);
      }

      // Generate recommendations
      const recommendations: TimingOptimization = {
        optimal_hours: this.getDefaultOptimalHours(channel),
        timezone: this.defaultTimezone,
        avoid_hours: [0, 1, 2, 3, 4, 5, 22, 23], // Late night/early morning
        preferred_days: [1, 2, 3, 4, 5], // Monday-Friday
        minimum_gap_hours: 6
      };

      // Cache for 24 hours
      await eventsCache.setex(cacheKey, 86400, JSON.stringify(recommendations));

      return recommendations;

    } catch (error) {
      logger.error('Failed to get timing recommendations', {
        userId,
        channel,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return default recommendations
      return {
        optimal_hours: this.getDefaultOptimalHours(channel),
        timezone: this.defaultTimezone,
        avoid_hours: [0, 1, 2, 3, 4, 5, 22, 23],
        preferred_days: [1, 2, 3, 4, 5],
        minimum_gap_hours: 6
      };
    }
  }

  /**
   * Health check for timing optimizer
   */
  public async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    const details: any = {};
    let healthy = true;

    try {
      details.default_timezone = this.defaultTimezone;
      details.ml_model_loaded = !!this.timingModel;

      // Test cache connectivity
      await eventsCache.ping();
      details.cache_connection = 'healthy';

      // Test timing calculation
      const testTime = new Date();
      const testOptimalHours = this.getDefaultOptimalHours(Channel.EMAIL);
      const testScore = this.getHourScore(testTime, testOptimalHours);
      details.timing_calculation = testScore >= 0 ? 'healthy' : 'unhealthy';

    } catch (error) {
      healthy = false;
      details.error = error instanceof Error ? error.message : String(error);
    }

    return { healthy, details };
  }
} 