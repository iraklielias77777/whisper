import { FatigueCheck, MessageHistory } from '../types';
import { logger, eventsCache } from '@userwhisperer/shared';
import { differenceInHours, differenceInDays } from 'date-fns';

export interface FatigueLimits {
  daily_max: number;
  weekly_max: number;
  monthly_max: number;
  min_gap_hours: number;
}

interface FatigueData {
  total_messages: number;
  daily_count: number;
  weekly_count: number;
  monthly_count: number;
  last_message_at: string | null;
  message_types: Record<string, number>;
}

export class FatigueManager {
  private limits: FatigueLimits;

  constructor(limits: FatigueLimits) {
    this.limits = limits;
  }

  /**
   * Check if user is fatigued from receiving too many messages
   */
  public async checkFatigue(
    userId: string,
    messageHistory: MessageHistory[]
  ): Promise<FatigueCheck> {
    try {
      const now = new Date();
      
      // Filter recent messages
      const last24Hours = messageHistory.filter(msg => 
        differenceInHours(now, msg.sent_at) <= 24
      );
      
      const last7Days = messageHistory.filter(msg => 
        differenceInDays(now, msg.sent_at) <= 7
      );
      
      const last30Days = messageHistory.filter(msg => 
        differenceInDays(now, msg.sent_at) <= 30
      );

      // Check daily limit
      if (last24Hours.length >= this.limits.daily_max) {
        return {
          is_fatigued: true,
          reason: `Daily limit exceeded (${last24Hours.length}/${this.limits.daily_max})`,
          suggested_wait_time: 24,
          recent_message_count: last24Hours.length
        };
      }

      // Check weekly limit
      if (last7Days.length >= this.limits.weekly_max) {
        return {
          is_fatigued: true,
          reason: `Weekly limit exceeded (${last7Days.length}/${this.limits.weekly_max})`,
          suggested_wait_time: 168, // 7 days
          recent_message_count: last7Days.length
        };
      }

      // Check monthly limit
      if (last30Days.length >= this.limits.monthly_max) {
        return {
          is_fatigued: true,
          reason: `Monthly limit exceeded (${last30Days.length}/${this.limits.monthly_max})`,
          suggested_wait_time: 720, // 30 days
          recent_message_count: last30Days.length
        };
      }

      // Check minimum gap between messages
      if (messageHistory.length > 0) {
        const lastMessage = messageHistory[0]; // Assuming sorted by most recent
        const hoursSinceLastMessage = differenceInHours(now, lastMessage.sent_at);
        
        if (hoursSinceLastMessage < this.limits.min_gap_hours) {
          return {
            is_fatigued: true,
            reason: `Minimum gap not met (${hoursSinceLastMessage}h < ${this.limits.min_gap_hours}h)`,
            suggested_wait_time: this.limits.min_gap_hours - hoursSinceLastMessage,
            recent_message_count: last24Hours.length,
            last_message_hours_ago: hoursSinceLastMessage
          };
        }
      }

      // Check for unsubscription or complaints
      const recentUnsubscribes = messageHistory.filter(msg => 
        msg.unsubscribed_at && differenceInDays(now, msg.unsubscribed_at) <= 30
      );

      if (recentUnsubscribes.length > 0) {
        return {
          is_fatigued: true,
          reason: 'User recently unsubscribed from communications',
          suggested_wait_time: 720, // 30 days
          recent_message_count: last24Hours.length
        };
      }

      // Check for high bounce rate (indicates deliverability issues)
      const recentBounces = messageHistory.filter(msg => 
        msg.bounced_at && differenceInDays(now, msg.bounced_at) <= 7
      );

      if (recentBounces.length >= 3) {
        return {
          is_fatigued: true,
          reason: 'High bounce rate detected',
          suggested_wait_time: 168, // 7 days
          recent_message_count: last24Hours.length
        };
      }

      // No fatigue detected
      return {
        is_fatigued: false,
        recent_message_count: last24Hours.length,
        last_message_hours_ago: messageHistory.length > 0 
          ? differenceInHours(now, messageHistory[0].sent_at)
          : undefined
      };

    } catch (error) {
      logger.error('Fatigue check failed', {
        userId,
        error: error instanceof Error ? error.message : String(error)
      });

      // Conservative fallback - assume not fatigued but with minimal info
      return {
        is_fatigued: false,
        recent_message_count: 0
      };
    }
  }

  /**
   * Get recommended wait time before next message
   */
  public getRecommendedWaitTime(
    messageHistory: MessageHistory[]
  ): number {
    if (messageHistory.length === 0) {
      return 0; // Can send immediately
    }

    const now = new Date();
    const lastMessage = messageHistory[0];
    const hoursSinceLastMessage = differenceInHours(now, lastMessage.sent_at);

    // If minimum gap is already met, return 0
    if (hoursSinceLastMessage >= this.limits.min_gap_hours) {
      return 0;
    }

    // Return remaining wait time
    return this.limits.min_gap_hours - hoursSinceLastMessage;
  }

  /**
   * Check if user can receive a specific type of message
   */
  public async canReceiveMessageType(
    userId: string,
    messageType: string,
    messageHistory: MessageHistory[]
  ): Promise<boolean> {
    // Check general fatigue first
    const fatigueCheck = await this.checkFatigue(userId, messageHistory);
    if (fatigueCheck.is_fatigued) {
      return false;
    }

    // Check type-specific limits
    const now = new Date();
    const recentSameType = messageHistory.filter(msg => 
      msg.intervention_type.toString() === messageType &&
      differenceInDays(now, msg.sent_at) <= 7
    );

    // Type-specific limits
    const typeLimits: Record<string, number> = {
      'monetization': 2, // Max 2 monetization messages per week
      'retention': 3,    // Max 3 retention messages per week
      'support': 5,      // Max 5 support messages per week
      'onboarding': 7,   // Max 7 onboarding messages per week
      'celebration': 1,  // Max 1 celebration message per week
      'education': 2,    // Max 2 education messages per week
      'reactivation': 1  // Max 1 reactivation message per week
    };

    const typeLimit = typeLimits[messageType] || 3; // Default limit
    
    return recentSameType.length < typeLimit;
  }

  /**
   * Update fatigue cache after sending a message
   */
  public async updateFatigueCache(
    userId: string,
    messageType: string,
    channel: string
  ): Promise<void> {
    try {
      const cacheKey = `fatigue:${userId}`;
      const now = new Date();
      
      // Get existing cache data
      const cachedData = await eventsCache.getJSON<FatigueData>(cacheKey);
      const fatigueData: FatigueData = cachedData || {
        total_messages: 0,
        daily_count: 0,
        weekly_count: 0,
        monthly_count: 0,
        last_message_at: null,
        message_types: {}
      };

      // Update counters
      fatigueData.total_messages += 1;
      fatigueData.daily_count += 1;
      fatigueData.weekly_count += 1;
      fatigueData.monthly_count += 1;
      fatigueData.last_message_at = now.toISOString();

      // Update type-specific counter
      if (!fatigueData.message_types[messageType]) {
        fatigueData.message_types[messageType] = 0;
      }
      fatigueData.message_types[messageType] += 1;

      // Store updated data with appropriate TTL
      await eventsCache.setJSON(cacheKey, fatigueData, 2592000); // 30 days

      logger.debug('Fatigue cache updated', {
        userId,
        messageType,
        channel,
        dailyCount: fatigueData.daily_count,
        weeklyCount: fatigueData.weekly_count
      });

    } catch (error) {
      logger.error('Failed to update fatigue cache', {
        userId,
        messageType,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  /**
   * Reset daily counters (should be called by a scheduled job)
   */
  public async resetDailyCounters(): Promise<void> {
    try {
      // This would typically be implemented with a pattern-based cache operation
      // For now, this is a placeholder for the daily reset logic
      logger.info('Daily fatigue counters reset initiated');
      
      // In a real implementation, this would:
      // 1. Scan all fatigue cache keys
      // 2. Reset daily_count to 0 for all users
      // 3. Keep other counters intact
      
    } catch (error) {
      logger.error('Failed to reset daily counters', {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  /**
   * Get fatigue statistics for monitoring
   */
  public async getFatigueStats(userId: string): Promise<{
    daily_count: number;
    weekly_count: number;
    monthly_count: number;
    last_message_hours_ago?: number;
    is_near_limit: boolean;
  }> {
    try {
      const cacheKey = `fatigue:${userId}`;
      const fatigueData = await eventsCache.getJSON<FatigueData>(cacheKey);

      if (!fatigueData) {
        return {
          daily_count: 0,
          weekly_count: 0,
          monthly_count: 0,
          is_near_limit: false
        };
      }

      const lastMessageHoursAgo = fatigueData.last_message_at
        ? differenceInHours(new Date(), new Date(fatigueData.last_message_at))
        : undefined;

      const isNearLimit = (
        fatigueData.daily_count >= this.limits.daily_max * 0.8 ||
        fatigueData.weekly_count >= this.limits.weekly_max * 0.8 ||
        fatigueData.monthly_count >= this.limits.monthly_max * 0.8
      );

      return {
        daily_count: fatigueData.daily_count || 0,
        weekly_count: fatigueData.weekly_count || 0,
        monthly_count: fatigueData.monthly_count || 0,
        last_message_hours_ago: lastMessageHoursAgo,
        is_near_limit: isNearLimit
      };

    } catch (error) {
      logger.error('Failed to get fatigue stats', {
        userId,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        daily_count: 0,
        weekly_count: 0,
        monthly_count: 0,
        is_near_limit: false
      };
    }
  }
} 