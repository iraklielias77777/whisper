import Redis, { RedisOptions, Cluster } from 'ioredis';
import Config from './config';
import logger from './logger';

export class RedisConnection {
  private redis: Redis | Cluster;
  private static instance: RedisConnection;

  private constructor() {
    const redisConfig: RedisOptions = {
      host: Config.REDIS_HOST,
      port: Config.REDIS_PORT,
      password: Config.REDIS_PASSWORD,
      db: Config.REDIS_DB,
      maxRetriesPerRequest: 3,
      lazyConnect: true,
      keepAlive: 30000,
      connectTimeout: 10000,
      commandTimeout: 5000,
    };

    // Initialize Redis connection
    this.redis = new Redis(redisConfig);

    // Event handlers
    this.redis.on('connect', () => {
      logger.info('Redis connected successfully');
    });

    this.redis.on('ready', () => {
      logger.info('Redis is ready to receive commands');
    });

    this.redis.on('error', (error) => {
      logger.error('Redis connection error', { error });
    });

    this.redis.on('close', () => {
      logger.warn('Redis connection closed');
    });

    this.redis.on('reconnecting', (ms: number) => {
      logger.info(`Redis reconnecting in ${ms}ms`);
    });
  }

  public static getInstance(): RedisConnection {
    if (!RedisConnection.instance) {
      RedisConnection.instance = new RedisConnection();
    }
    return RedisConnection.instance;
  }

  public getClient(): Redis | Cluster {
    return this.redis;
  }

  // Basic operations
  async get(key: string): Promise<string | null> {
    try {
      return await this.redis.get(key);
    } catch (error) {
      logger.error('Redis GET error', { key, error });
      throw error;
    }
  }

  async set(key: string, value: string, expireInSeconds?: number): Promise<'OK'> {
    try {
      if (expireInSeconds) {
        return await this.redis.setex(key, expireInSeconds, value);
      }
      return await this.redis.set(key, value);
    } catch (error) {
      logger.error('Redis SET error', { key, error });
      throw error;
    }
  }

  async del(key: string): Promise<number> {
    try {
      return await this.redis.del(key);
    } catch (error) {
      logger.error('Redis DEL error', { key, error });
      throw error;
    }
  }

  async exists(key: string): Promise<number> {
    try {
      return await this.redis.exists(key);
    } catch (error) {
      logger.error('Redis EXISTS error', { key, error });
      throw error;
    }
  }

  async expire(key: string, seconds: number): Promise<number> {
    try {
      return await this.redis.expire(key, seconds);
    } catch (error) {
      logger.error('Redis EXPIRE error', { key, seconds, error });
      throw error;
    }
  }

  // Hash operations
  async hget(key: string, field: string): Promise<string | null> {
    try {
      return await this.redis.hget(key, field);
    } catch (error) {
      logger.error('Redis HGET error', { key, field, error });
      throw error;
    }
  }

  async hset(key: string, field: string, value: string): Promise<number> {
    try {
      return await this.redis.hset(key, field, value);
    } catch (error) {
      logger.error('Redis HSET error', { key, field, error });
      throw error;
    }
  }

  async hgetall(key: string): Promise<Record<string, string>> {
    try {
      return await this.redis.hgetall(key);
    } catch (error) {
      logger.error('Redis HGETALL error', { key, error });
      throw error;
    }
  }

  async hdel(key: string, ...fields: string[]): Promise<number> {
    try {
      return await this.redis.hdel(key, ...fields);
    } catch (error) {
      logger.error('Redis HDEL error', { key, fields, error });
      throw error;
    }
  }

  // List operations for queues
  async lpush(key: string, ...values: string[]): Promise<number> {
    try {
      return await this.redis.lpush(key, ...values);
    } catch (error) {
      logger.error('Redis LPUSH error', { key, error });
      throw error;
    }
  }

  async rpop(key: string): Promise<string | null> {
    try {
      return await this.redis.rpop(key);
    } catch (error) {
      logger.error('Redis RPOP error', { key, error });
      throw error;
    }
  }

  async llen(key: string): Promise<number> {
    try {
      return await this.redis.llen(key);
    } catch (error) {
      logger.error('Redis LLEN error', { key, error });
      throw error;
    }
  }

  // Set operations
  async sadd(key: string, ...members: string[]): Promise<number> {
    try {
      return await this.redis.sadd(key, ...members);
    } catch (error) {
      logger.error('Redis SADD error', { key, error });
      throw error;
    }
  }

  async sismember(key: string, member: string): Promise<number> {
    try {
      return await this.redis.sismember(key, member);
    } catch (error) {
      logger.error('Redis SISMEMBER error', { key, member, error });
      throw error;
    }
  }

  // JSON operations
  async getJSON<T = any>(key: string): Promise<T | null> {
    const value = await this.get(key);
    if (!value) return null;
    
    try {
      return JSON.parse(value);
    } catch (error) {
      logger.error('JSON parse error', { key, value, error });
      throw error;
    }
  }

  async setJSON(key: string, value: any, expireInSeconds?: number): Promise<'OK'> {
    const jsonString = JSON.stringify(value);
    return this.set(key, jsonString, expireInSeconds);
  }

  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      const result = await this.redis.ping();
      return result === 'PONG';
    } catch (error) {
      logger.error('Redis health check failed', { error });
      return false;
    }
  }

  // Close connection
  async close(): Promise<void> {
    await this.redis.quit();
    logger.info('Redis connection closed');
  }

  // Get info
  async getInfo(): Promise<any> {
    try {
      const info = await this.redis.info();
      return info;
    } catch (error) {
      logger.error('Redis INFO error', { error });
      throw error;
    }
  }
}

// Event-specific Redis operations
export class EventsCache {
  private redis: RedisConnection;

  constructor() {
    this.redis = RedisConnection.getInstance();
  }

  // Deduplication cache
  async checkEventDuplicate(eventId: string): Promise<boolean> {
    const key = `event:dedup:${eventId}`;
    const exists = await this.redis.exists(key);
    return exists === 1;
  }

  async markEventProcessed(eventId: string, ttlSeconds: number = 3600): Promise<void> {
    const key = `event:dedup:${eventId}`;
    await this.redis.set(key, '1', ttlSeconds);
  }

  // Rate limiting
  async checkRateLimit(identifier: string, maxRequests: number, windowSeconds: number): Promise<{ allowed: boolean; remaining: number; resetTime: number }> {
    const key = `rate_limit:${identifier}`;
    const current = await this.redis.get(key);
    
    if (!current) {
      await this.redis.set(key, '1', windowSeconds);
      return {
        allowed: true,
        remaining: maxRequests - 1,
        resetTime: Date.now() + (windowSeconds * 1000)
      };
    }

    const count = parseInt(current);
    if (count >= maxRequests) {
      const ttl = await this.redis.getClient().ttl(key);
      return {
        allowed: false,
        remaining: 0,
        resetTime: Date.now() + (ttl * 1000)
      };
    }

    await this.redis.getClient().incr(key);
    const ttl = await this.redis.getClient().ttl(key);
    
    return {
      allowed: true,
      remaining: maxRequests - count - 1,
      resetTime: Date.now() + (ttl * 1000)
    };
  }

  // Event queue operations
  async queueEvent(queueName: string, event: any): Promise<number> {
    const key = `queue:${queueName}`;
    const eventString = JSON.stringify(event);
    return this.redis.lpush(key, eventString);
  }

  async dequeueEvent(queueName: string): Promise<any> {
    const key = `queue:${queueName}`;
    const eventString = await this.redis.rpop(key);
    
    if (!eventString) return null;
    
    try {
      return JSON.parse(eventString);
    } catch (error) {
      logger.error('Failed to parse queued event', { queueName, eventString, error });
      return null;
    }
  }

  async getQueueLength(queueName: string): Promise<number> {
    const key = `queue:${queueName}`;
    return this.redis.llen(key);
  }

  // User session cache
  async getUserSession(userId: string): Promise<any> {
    const key = `user:session:${userId}`;
    return this.redis.getJSON(key);
  }

  async setUserSession(userId: string, sessionData: any, ttlSeconds: number = 1800): Promise<void> {
    const key = `user:session:${userId}`;
    await this.redis.setJSON(key, sessionData, ttlSeconds);
  }

  // Real-time metrics
  async incrementCounter(counterName: string, amount: number = 1): Promise<number> {
    const key = `counter:${counterName}`;
    return this.redis.getClient().incrby(key, amount);
  }

  async getCounter(counterName: string): Promise<number> {
    const key = `counter:${counterName}`;
    const value = await this.redis.get(key);
    return value ? parseInt(value) : 0;
  }

  // JSON operations (backwards compatibility)
  async getJSON<T>(key: string): Promise<T | null> {
    return this.redis.getJSON<T>(key);
  }

  async setJSON<T>(key: string, value: T, ttlSeconds?: number): Promise<'OK'> {
    return this.redis.setJSON(key, value, ttlSeconds);
  }

  // Basic operations (backwards compatibility)
  async get(key: string): Promise<string | null> {
    return this.redis.get(key);
  }

  async set(key: string, value: string, ttlSeconds?: number): Promise<'OK'> {
    return this.redis.set(key, value, ttlSeconds);
  }

  async setex(key: string, ttlSeconds: number, value: string): Promise<'OK'> {
    return this.redis.set(key, value, ttlSeconds);
  }

  async del(key: string): Promise<number> {
    return this.redis.del(key);
  }

  // Queue operations (removed duplicates - using the implementation above)

  // Health check
  async ping(): Promise<'PONG'> {
    return this.redis.getClient().ping();
  }

  // Hash operations
  async hget(key: string, field: string): Promise<string | null> {
    return this.redis.hget(key, field);
  }

  async hset(key: string, field: string, value: string): Promise<number> {
    return this.redis.hset(key, field, value);
  }

  async hgetall(key: string): Promise<Record<string, string>> {
    return this.redis.hgetall(key);
  }

  async hdel(key: string, ...fields: string[]): Promise<number> {
    return this.redis.hdel(key, ...fields);
  }

  // Sorted set operations for rate limiting
  async zadd(key: string, score: number, member: string): Promise<number> {
    return this.redis.getClient().zadd(key, score, member);
  }

  async zcard(key: string): Promise<number> {
    return this.redis.getClient().zcard(key);
  }

  async zremrangebyscore(key: string, min: string | number, max: string | number): Promise<number> {
    return this.redis.getClient().zremrangebyscore(key, min.toString(), max.toString());
  }

  // TTL operations
  async expire(key: string, seconds: number): Promise<number> {
    return this.redis.getClient().expire(key, seconds);
  }

  async setWithExpiry(key: string, value: string, ttlSeconds: number): Promise<'OK'> {
    return this.redis.set(key, value, ttlSeconds);
  }

  // Health check
  async healthCheck(): Promise<boolean> {
    return this.redis.healthCheck();
  }
}

// Export singleton instances
export const redis = RedisConnection.getInstance();
export const eventsCache = new EventsCache(); 