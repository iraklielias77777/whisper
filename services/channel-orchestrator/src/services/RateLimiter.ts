import { RateLimitResult, RateLimitConfig } from '../types';
import { logger, eventsCache } from '@userwhisperer/shared';
import { addHours, addDays, addMinutes } from 'date-fns';

export class RateLimiter {
  private config: RateLimitConfig;
  private windows: Record<string, number> = {
    'per_minute': 60,
    'per_hour': 3600,
    'per_day': 86400
  };

  constructor(config: RateLimitConfig) {
    this.config = config;
  }

  public async initialize(): Promise<void> {
    logger.info('Initializing Rate Limiter');
    // Rate limiter is stateless, no initialization needed
    logger.info('Rate Limiter initialized successfully');
  }

  public async checkLimit(userId: string, channel: string): Promise<RateLimitResult> {
    try {
      // Get rate limits for the channel
      const channelLimits = this.config[channel as keyof RateLimitConfig];
      
      if (!channelLimits) {
        logger.warn(`No rate limits configured for channel: ${channel}`);
        return {
          allowed: true,
          limit: 1000, // Default high limit
          remaining: 999,
          reset_time: addHours(new Date(), 1)
        };
      }

      // Check multiple time windows
      const checks = await Promise.all([
        this.checkWindow(userId, channel, 'hour', channelLimits.per_user_per_hour),
        this.checkWindow(userId, channel, 'day', channelLimits.per_user_per_day),
        this.checkGlobalWindow(channel, 'minute', channelLimits.global_per_minute)
      ]);

      // Find the most restrictive limit
      const mostRestrictive = checks.reduce((prev, current) => 
        current.remaining < prev.remaining ? current : prev
      );

      if (!mostRestrictive.allowed) {
        logger.warn(`Rate limit exceeded for user ${userId} on ${channel}`, {
          limit: mostRestrictive.limit,
          remaining: mostRestrictive.remaining,
          reset_time: mostRestrictive.reset_time
        });
      }

      return mostRestrictive;

    } catch (error) {
      logger.error(`Rate limit check failed for user ${userId}:`, error);
      
      // Fail open - allow the request but log the error
      return {
        allowed: true,
        limit: 1,
        remaining: 0,
        reset_time: addMinutes(new Date(), 1)
      };
    }
  }

  public async recordDelivery(userId: string, channel: string): Promise<void> {
    try {
      const now = Date.now();
      const promises: Promise<any>[] = [];

      // Record in all relevant windows
      promises.push(this.recordInWindow(userId, channel, 'hour', now));
      promises.push(this.recordInWindow(userId, channel, 'day', now));
      promises.push(this.recordGlobalWindow(channel, 'minute', now));

      await Promise.all(promises);

      logger.debug(`Recorded delivery for user ${userId} on ${channel}`);

    } catch (error) {
      logger.error(`Failed to record delivery for user ${userId}:`, error);
    }
  }

  public async getUserLimitStats(userId: string, channel: string): Promise<any> {
    try {
      const channelLimits = this.config[channel as keyof RateLimitConfig];
      
      if (!channelLimits) {
        return {
          channel,
          hour: { used: 0, limit: 0, remaining: 0 },
          day: { used: 0, limit: 0, remaining: 0 }
        };
      }

      const [hourlyUsed, dailyUsed] = await Promise.all([
        this.getWindowUsage(userId, channel, 'hour'),
        this.getWindowUsage(userId, channel, 'day')
      ]);

      return {
        channel,
        hour: {
          used: hourlyUsed,
          limit: channelLimits.per_user_per_hour,
          remaining: Math.max(0, channelLimits.per_user_per_hour - hourlyUsed)
        },
        day: {
          used: dailyUsed,
          limit: channelLimits.per_user_per_day,
          remaining: Math.max(0, channelLimits.per_user_per_day - dailyUsed)
        }
      };

    } catch (error) {
      logger.error(`Failed to get limit stats for user ${userId}:`, error);
      return {
        channel,
        hour: { used: 0, limit: 0, remaining: 0 },
        day: { used: 0, limit: 0, remaining: 0 },
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  public async getGlobalLimitStats(channel: string): Promise<any> {
    try {
      const channelLimits = this.config[channel as keyof RateLimitConfig];
      
      if (!channelLimits) {
        return {
          channel,
          minute: { used: 0, limit: 0, remaining: 0 }
        };
      }

      const minuteUsed = await this.getGlobalWindowUsage(channel, 'minute');

      return {
        channel,
        minute: {
          used: minuteUsed,
          limit: channelLimits.global_per_minute,
          remaining: Math.max(0, channelLimits.global_per_minute - minuteUsed)
        }
      };

    } catch (error) {
      logger.error(`Failed to get global limit stats for channel ${channel}:`, error);
      return {
        channel,
        minute: { used: 0, limit: 0, remaining: 0 },
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  public async resetUserLimits(userId: string, channel?: string): Promise<void> {
    try {
      const channels = channel ? [channel] : ['email', 'sms', 'push'];
      const windows = ['hour', 'day'];

      const promises: Promise<any>[] = [];

      for (const ch of channels) {
        for (const window of windows) {
          const key = this.getUserWindowKey(userId, ch, window);
          promises.push(eventsCache.del(key));
        }
      }

      await Promise.all(promises);

      logger.info(`Reset rate limits for user ${userId}`, { channels });

    } catch (error) {
      logger.error(`Failed to reset limits for user ${userId}:`, error);
    }
  }

  public async resetGlobalLimits(channel?: string): Promise<void> {
    try {
      const channels = channel ? [channel] : ['email', 'sms', 'push'];

      const promises: Promise<any>[] = [];

      for (const ch of channels) {
        const key = this.getGlobalWindowKey(ch, 'minute');
        promises.push(eventsCache.del(key));
      }

      await Promise.all(promises);

      logger.info(`Reset global rate limits`, { channels });

    } catch (error) {
      logger.error(`Failed to reset global limits:`, error);
    }
  }

  private async checkWindow(
    userId: string,
    channel: string,
    window: string,
    limit: number
  ): Promise<RateLimitResult> {
    const key = this.getUserWindowKey(userId, channel, window);
    const windowSeconds = this.windows[`per_${window}`];
    const now = Date.now();
    const windowStart = now - (windowSeconds * 1000);

    // Remove old entries and count current usage
    await eventsCache.zremrangebyscore(key, '-inf', windowStart);
    const currentUsage = await eventsCache.zcard(key);

    const remaining = Math.max(0, limit - currentUsage);
    const allowed = currentUsage < limit;

    // Calculate reset time
    let resetTime: Date;
    if (window === 'hour') {
      resetTime = addHours(new Date(), 1);
    } else if (window === 'day') {
      resetTime = addDays(new Date(), 1);
    } else {
      resetTime = addMinutes(new Date(), 1);
    }

    return {
      allowed,
      limit,
      remaining,
      reset_time: resetTime
    };
  }

  private async checkGlobalWindow(
    channel: string,
    window: string,
    limit: number
  ): Promise<RateLimitResult> {
    const key = this.getGlobalWindowKey(channel, window);
    const windowSeconds = this.windows[`per_${window}`];
    const now = Date.now();
    const windowStart = now - (windowSeconds * 1000);

    // Remove old entries and count current usage
    await eventsCache.zremrangebyscore(key, '-inf', windowStart);
    const currentUsage = await eventsCache.zcard(key);

    const remaining = Math.max(0, limit - currentUsage);
    const allowed = currentUsage < limit;

    let resetTime: Date;
    if (window === 'minute') {
      resetTime = addMinutes(new Date(), 1);
    } else if (window === 'hour') {
      resetTime = addHours(new Date(), 1);
    } else {
      resetTime = addDays(new Date(), 1);
    }

    return {
      allowed,
      limit,
      remaining,
      reset_time: resetTime
    };
  }

  private async recordInWindow(
    userId: string,
    channel: string,
    window: string,
    timestamp: number
  ): Promise<void> {
    const key = this.getUserWindowKey(userId, channel, window);
    const windowSeconds = this.windows[`per_${window}`];

    // Add current timestamp
    await eventsCache.zadd(key, timestamp, `${timestamp}-${Math.random()}`);
    
    // Set expiration
    await eventsCache.expire(key, windowSeconds);
  }

  private async recordGlobalWindow(
    channel: string,
    window: string,
    timestamp: number
  ): Promise<void> {
    const key = this.getGlobalWindowKey(channel, window);
    const windowSeconds = this.windows[`per_${window}`];

    // Add current timestamp
    await eventsCache.zadd(key, timestamp, `${timestamp}-${Math.random()}`);
    
    // Set expiration
    await eventsCache.expire(key, windowSeconds);
  }

  private async getWindowUsage(userId: string, channel: string, window: string): Promise<number> {
    const key = this.getUserWindowKey(userId, channel, window);
    const windowSeconds = this.windows[`per_${window}`];
    const now = Date.now();
    const windowStart = now - (windowSeconds * 1000);

    // Clean old entries and count
    await eventsCache.zremrangebyscore(key, '-inf', windowStart);
    return await eventsCache.zcard(key);
  }

  private async getGlobalWindowUsage(channel: string, window: string): Promise<number> {
    const key = this.getGlobalWindowKey(channel, window);
    const windowSeconds = this.windows[`per_${window}`];
    const now = Date.now();
    const windowStart = now - (windowSeconds * 1000);

    // Clean old entries and count
    await eventsCache.zremrangebyscore(key, '-inf', windowStart);
    return await eventsCache.zcard(key);
  }

  private getUserWindowKey(userId: string, channel: string, window: string): string {
    return `rate_limit:user:${userId}:${channel}:${window}`;
  }

  private getGlobalWindowKey(channel: string, window: string): string {
    return `rate_limit:global:${channel}:${window}`;
  }

  public async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    try {
      // Test Redis connectivity
      const testKey = 'rate_limit:health_check';
      await eventsCache.set(testKey, '1');
      await eventsCache.del(testKey);

      // Get stats for all channels
      const channelStats = await Promise.all([
        this.getGlobalLimitStats('email'),
        this.getGlobalLimitStats('sms'),
        this.getGlobalLimitStats('push')
      ]);

      return {
        healthy: true,
        details: {
          redis_connected: true,
          channels_configured: Object.keys(this.config).length,
          global_stats: channelStats
        }
      };
    } catch (error) {
      return {
        healthy: false,
        details: { error: error instanceof Error ? error.message : String(error) }
      };
    }
  }
} 