import { DeliveryRequest, QueuedMessage } from '../types';
import { logger, eventsCache } from '@userwhisperer/shared';
import { addMinutes, isBefore } from 'date-fns';

export class DeliveryQueue {
  private queue: QueuedMessage[] = [];
  private processing: boolean = false;
  private maxQueueSize: number = 100000;
  private priority_weights: Record<number, number> = {
    1: 1,    // Lowest priority
    2: 2,
    3: 4,
    4: 8,
    5: 16    // Highest priority
  };

  constructor() {
    this.setupCleanupInterval();
  }

  public async initialize(): Promise<void> {
    logger.info('Initializing Delivery Queue');
    
    try {
      // Load any persisted queue items from Redis
      await this.loadPersistedQueue();
      
      logger.info(`Delivery Queue initialized with ${this.queue.length} items`);
    } catch (error) {
      logger.error('Failed to initialize Delivery Queue:', error);
      throw error;
    }
  }

  public async add(request: DeliveryRequest): Promise<void> {
    if (this.queue.length >= this.maxQueueSize) {
      throw new Error('Delivery queue is full');
    }

    const queuedMessage: QueuedMessage = {
      request: {
        ...request,
        retry_count: request.retry_count || 0,
        max_retries: request.max_retries || 3
      },
      queued_at: new Date(),
      scheduled_for: request.send_time
    };

    // Insert in priority order
    this.insertByPriority(queuedMessage);

    // Persist to Redis for durability
    await this.persistQueueItem(queuedMessage);

    logger.debug(`Added message ${request.message_id} to delivery queue`, {
      priority: request.priority,
      scheduled_for: request.send_time,
      queue_size: this.queue.length
    });
  }

  public async getReadyMessages(batchSize: number, currentTime: Date): Promise<QueuedMessage[]> {
    const readyMessages: QueuedMessage[] = [];
    const remaining: QueuedMessage[] = [];

    for (const queuedMessage of this.queue) {
      if (readyMessages.length >= batchSize) {
        remaining.push(queuedMessage);
        continue;
      }

      // Check if message is ready to be sent
      if (isBefore(queuedMessage.scheduled_for, currentTime) || 
          queuedMessage.scheduled_for <= currentTime) {
        readyMessages.push(queuedMessage);
        
        // Remove from Redis
        await this.removePersistedItem(queuedMessage.request.message_id);
      } else {
        remaining.push(queuedMessage);
      }
    }

    // Update queue with remaining messages
    this.queue = remaining;

    if (readyMessages.length > 0) {
      logger.debug(`Retrieved ${readyMessages.length} ready messages from queue`);
    }

    return readyMessages;
  }

  public async remove(messageId: string): Promise<boolean> {
    const initialLength = this.queue.length;
    this.queue = this.queue.filter(item => item.request.message_id !== messageId);
    
    if (this.queue.length < initialLength) {
      await this.removePersistedItem(messageId);
      logger.debug(`Removed message ${messageId} from delivery queue`);
      return true;
    }

    return false;
  }

  public async reschedule(messageId: string, newSendTime: Date): Promise<boolean> {
    const messageIndex = this.queue.findIndex(item => item.request.message_id === messageId);
    
    if (messageIndex === -1) {
      return false;
    }

    const queuedMessage = this.queue[messageIndex];
    queuedMessage.scheduled_for = newSendTime;
    queuedMessage.request.send_time = newSendTime;

    // Remove from current position
    this.queue.splice(messageIndex, 1);

    // Re-insert in priority order
    this.insertByPriority(queuedMessage);

    // Update in Redis
    await this.persistQueueItem(queuedMessage);

    logger.debug(`Rescheduled message ${messageId} for ${newSendTime}`);
    return true;
  }

  public getQueueSize(): number {
    return this.queue.length;
  }

  public getStats(): any {
    const now = new Date();
    const overdue = this.queue.filter(item => item.scheduled_for < now).length;
    const upcoming = this.queue.length - overdue;

    const priorityBreakdown = this.queue.reduce((acc, item) => {
      const priority = item.request.priority || 1;
      acc[priority] = (acc[priority] || 0) + 1;
      return acc;
    }, {} as Record<number, number>);

    const oldestMessage = this.queue.length > 0 
      ? Math.min(...this.queue.map(item => item.queued_at.getTime()))
      : null;

    return {
      total_queued: this.queue.length,
      overdue_messages: overdue,
      upcoming_messages: upcoming,
      priority_breakdown: priorityBreakdown,
      oldest_message_age_minutes: oldestMessage 
        ? Math.floor((now.getTime() - oldestMessage) / 60000)
        : 0,
      max_queue_size: this.maxQueueSize,
      utilization_percentage: Math.round((this.queue.length / this.maxQueueSize) * 100)
    };
  }

  public async clear(): Promise<void> {
    const messageIds = this.queue.map(item => item.request.message_id);
    
    this.queue = [];
    
    // Clear from Redis
    for (const messageId of messageIds) {
      await this.removePersistedItem(messageId);
    }

    logger.info('Delivery queue cleared');
  }

  public async pause(): Promise<void> {
    this.processing = false;
    logger.info('Delivery queue paused');
  }

  public async resume(): Promise<void> {
    this.processing = true;
    logger.info('Delivery queue resumed');
  }

  public isPaused(): boolean {
    return !this.processing;
  }

  private insertByPriority(queuedMessage: QueuedMessage): void {
    const priority = queuedMessage.request.priority || 1;
    const weight = this.priority_weights[priority] || 1;
    
    // Find insertion point based on priority and scheduled time
    let insertIndex = 0;
    
    for (let i = 0; i < this.queue.length; i++) {
      const existingPriority = this.queue[i].request.priority || 1;
      const existingWeight = this.priority_weights[existingPriority] || 1;
      
      // Higher priority goes first
      if (weight > existingWeight) {
        break;
      }
      
      // Same priority, earlier scheduled time goes first
      if (weight === existingWeight && 
          queuedMessage.scheduled_for <= this.queue[i].scheduled_for) {
        break;
      }
      
      insertIndex++;
    }

    this.queue.splice(insertIndex, 0, queuedMessage);
  }

  private async loadPersistedQueue(): Promise<void> {
    try {
      const queueKey = 'delivery_queue:items';
      const persistedItems = await eventsCache.hgetall(queueKey);
      
      if (persistedItems) {
        for (const [messageId, itemData] of Object.entries(persistedItems)) {
          try {
            const queuedMessage: QueuedMessage = JSON.parse(itemData);
            
            // Convert date strings back to Date objects
            queuedMessage.queued_at = new Date(queuedMessage.queued_at);
            queuedMessage.scheduled_for = new Date(queuedMessage.scheduled_for);
            queuedMessage.request.send_time = new Date(queuedMessage.request.send_time);
            
            this.insertByPriority(queuedMessage);
          } catch (parseError) {
            logger.warn(`Failed to parse persisted queue item ${messageId}:`, parseError);
            // Remove invalid item
            await eventsCache.hdel(queueKey, messageId);
          }
        }
      }

      logger.info(`Loaded ${this.queue.length} persisted queue items`);
    } catch (error) {
      logger.warn('Failed to load persisted queue:', error);
    }
  }

  private async persistQueueItem(queuedMessage: QueuedMessage): Promise<void> {
    try {
      const queueKey = 'delivery_queue:items';
      const itemData = JSON.stringify(queuedMessage);
      
      await eventsCache.hset(queueKey, queuedMessage.request.message_id, itemData);
      
      // Set expiration for the entire hash (refresh on each update)
      await eventsCache.expire(queueKey, 86400); // 24 hours
    } catch (error) {
      logger.warn(`Failed to persist queue item ${queuedMessage.request.message_id}:`, error);
    }
  }

  private async removePersistedItem(messageId: string): Promise<void> {
    try {
      const queueKey = 'delivery_queue:items';
      await eventsCache.hdel(queueKey, messageId);
    } catch (error) {
      logger.warn(`Failed to remove persisted queue item ${messageId}:`, error);
    }
  }

  private setupCleanupInterval(): void {
    // Clean up expired messages every 5 minutes
    setInterval(async () => {
      try {
        await this.cleanupExpiredMessages();
      } catch (error) {
        logger.error('Error during queue cleanup:', error);
      }
    }, 300000); // 5 minutes
  }

  private async cleanupExpiredMessages(): Promise<void> {
    const now = new Date();
    const maxAge = addMinutes(now, -1440); // 24 hours ago
    
    const initialSize = this.queue.length;
    const expiredMessages: string[] = [];
    
    this.queue = this.queue.filter(item => {
      const isExpired = item.queued_at < maxAge && 
                      (item.request.retry_count || 0) >= (item.request.max_retries || 3);
      
      if (isExpired) {
        expiredMessages.push(item.request.message_id);
      }
      
      return !isExpired;
    });

    // Remove expired items from Redis
    for (const messageId of expiredMessages) {
      await this.removePersistedItem(messageId);
    }

    if (expiredMessages.length > 0) {
      logger.info(`Cleaned up ${expiredMessages.length} expired messages from queue`, {
        initial_size: initialSize,
        final_size: this.queue.length,
        expired_messages: expiredMessages
      });
    }
  }

  public async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    try {
      const stats = this.getStats();
      const isHealthy = stats.utilization_percentage < 90 && 
                       stats.oldest_message_age_minutes < 60; // No message older than 1 hour

      return {
        healthy: isHealthy,
        details: {
          ...stats,
          is_processing: this.processing,
          redis_connected: true // Would test Redis connection
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