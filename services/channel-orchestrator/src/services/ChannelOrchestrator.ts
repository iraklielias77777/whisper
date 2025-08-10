import { 
  DeliveryRequest, 
  DeliveryResult, 
  ValidationResult, 
  RateLimitResult, 
  ChannelOrchestratorConfig,
  QueuedMessage,
  DeliveryStatus,
  SuppressionReason 
} from '../types';
import { logger, eventsCache, database } from '@userwhisperer/shared';
import { EmailService } from './EmailService';
import { SMSService } from './SMSService';
import { PushService } from './PushService';
import { DeliveryQueue } from './DeliveryQueue';
import { RateLimiter } from './RateLimiter';
import { addHours, differenceInHours } from 'date-fns';
import { v4 as uuidv4 } from 'uuid';

export class ChannelOrchestrator {
  private config: ChannelOrchestratorConfig;
  private emailService: EmailService;
  private smsService: SMSService;
  private pushService: PushService;
  private deliveryQueue: DeliveryQueue;
  private rateLimiter: RateLimiter;
  private isProcessing: boolean = false;

  constructor(config: ChannelOrchestratorConfig) {
    this.config = config;
    this.emailService = new EmailService(config.sendgrid);
    this.smsService = new SMSService(config.twilio);
    this.pushService = new PushService(config.firebase);
    this.deliveryQueue = new DeliveryQueue();
    this.rateLimiter = new RateLimiter(config.rate_limits);
  }

  public async initialize(): Promise<void> {
    logger.info('Initializing Channel Orchestrator');
    
    try {
      await this.emailService.initialize();
      await this.smsService.initialize();
      await this.pushService.initialize();
      await this.deliveryQueue.initialize();
      await this.rateLimiter.initialize();
      
      // Start processing queue
      this.startQueueProcessor();
      
      logger.info('Channel Orchestrator initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize Channel Orchestrator:', error);
      throw error;
    }
  }

  public async scheduleDelivery(request: DeliveryRequest): Promise<string> {
    logger.info(`Scheduling delivery for message ${request.message_id} to user ${request.user_id} via ${request.channel}`);

    try {
      // Validate request
      const validation = await this.validateRequest(request);
      if (!validation.is_valid) {
        throw new Error(`Invalid delivery request: ${validation.errors.join(', ')}`);
      }

      // Check rate limits
      const rateLimitResult = await this.rateLimiter.checkLimit(
        request.user_id,
        request.channel
      );

      if (!rateLimitResult.allowed) {
        logger.warn(`Rate limit exceeded for user ${request.user_id} on ${request.channel}`);
        
        // Reschedule for when rate limit resets
        request.send_time = rateLimitResult.reset_time;
        
        // For high priority, try alternative channel
        if (request.priority >= 4) {
          const altChannel = await this.findAlternativeChannel(
            request.user_id,
            request.channel
          );
          if (altChannel) {
            request.channel = altChannel;
            logger.info(`Switched to alternative channel ${altChannel} for high priority message`);
          }
        }
      }

      // Add to delivery queue
      await this.deliveryQueue.add(request);
      
      // Update delivery status
      await this.updateDeliveryStatus(request.message_id, 'queued');
      
      logger.info(`Message ${request.message_id} queued successfully`);
      
      return request.message_id;

    } catch (error) {
      logger.error(`Failed to schedule delivery for message ${request.message_id}:`, error);
      throw error;
    }
  }

  private startQueueProcessor(): void {
    if (this.isProcessing) {
      return;
    }
    
    this.isProcessing = true;
    this.processDeliveryQueue();
  }

  private async processDeliveryQueue(): Promise<void> {
    logger.info('Starting delivery queue processor');

    while (this.isProcessing) {
      try {
        // Get next batch of messages to send
        const batch = await this.deliveryQueue.getReadyMessages(100, new Date());
        
        if (batch.length === 0) {
          // No messages ready, wait a bit
          await this.delay(1000);
          continue;
        }

        logger.info(`Processing batch of ${batch.length} messages`);

        // Process batch in parallel
        const promises = batch.map(queuedMessage => 
          this.deliverMessage(queuedMessage.request)
        );

        const results = await Promise.allSettled(promises);

        // Handle results
        for (let i = 0; i < batch.length; i++) {
          const request = batch[i].request;
          const result = results[i];
          
          if (result.status === 'fulfilled') {
            await this.handleDeliveryResult(request, result.value);
          } else {
            await this.handleDeliveryError(request, result.reason);
          }
        }

      } catch (error) {
        logger.error('Error in delivery queue processor:', error);
        await this.delay(5000); // Wait 5 seconds before retrying
      }
    }
  }

  public async deliverMessage(request: DeliveryRequest): Promise<DeliveryResult> {
    logger.info(`Delivering message ${request.message_id} via ${request.channel}`);

    try {
      // Pre-delivery checks
      if (!await this.preDeliveryChecks(request)) {
        return {
          success: false,
          channel: request.channel,
          error: "Pre-delivery checks failed",
          should_retry: false
        };
      }

      // Update status to sending
      await this.updateDeliveryStatus(request.message_id, 'sent');

      // Route to appropriate service
      let result: DeliveryResult;
      
      switch (request.channel) {
        case 'email':
          result = await this.emailService.send(request);
          break;
        case 'sms':
          result = await this.smsService.send(request);
          break;
        case 'push':
          result = await this.pushService.send(request);
          break;
        default:
          throw new Error(`Unknown channel: ${request.channel}`);
      }

      // Track delivery
      await this.trackDelivery(request, result);

      logger.info(`Message ${request.message_id} delivery result:`, {
        success: result.success,
        provider_id: result.provider_message_id,
        error: result.error
      });

      return result;

    } catch (error) {
      logger.error(`Delivery failed for message ${request.message_id}:`, error);
      
      return {
        success: false,
        channel: request.channel,
        error: error instanceof Error ? error.message : String(error),
        should_retry: this.shouldRetry(error)
      };
    }
  }

  private async preDeliveryChecks(request: DeliveryRequest): Promise<boolean> {
    // Check if user has unsubscribed
    if (await this.isUnsubscribed(request.user_id, request.channel)) {
      await this.logSuppression(request, {
        type: 'unsubscribed',
        message: `User unsubscribed from ${request.channel}`,
        timestamp: new Date()
      });
      return false;
    }

    // Check if user has valid contact info
    if (!await this.hasValidContact(request.user_id, request.channel)) {
      await this.logSuppression(request, {
        type: 'invalid_contact',
        message: `Invalid contact info for ${request.channel}`,
        timestamp: new Date()
      });
      return false;
    }

    // Check for duplicate sends
    if (await this.isDuplicateSend(request)) {
      await this.logSuppression(request, {
        type: 'duplicate',
        message: 'Duplicate send detected',
        timestamp: new Date()
      });
      return false;
    }

    // Check business hours for SMS
    if (request.channel === 'sms' && !await this.isBusinessHours(request.user_id)) {
      await this.rescheduleForBusinessHours(request);
      await this.logSuppression(request, {
        type: 'business_hours',
        message: 'Outside business hours',
        timestamp: new Date()
      });
      return false;
    }

    return true;
  }

  private async handleDeliveryResult(request: DeliveryRequest, result: DeliveryResult): Promise<void> {
    if (result.success) {
      // Update delivery status
      await this.updateDeliveryStatus(
        request.message_id,
        'delivered',
        result.provider_message_id,
        result.delivered_at
      );

      // Emit success event
      await this.emitDeliveryEvent(request, 'delivered');

    } else if (result.should_retry && (request.retry_count || 0) < (request.max_retries || this.config.retry_settings.max_retries)) {
      // Schedule retry
      const retryCount = (request.retry_count || 0) + 1;
      const retryDelay = this.calculateRetryDelay(retryCount);
      
      const retryRequest: DeliveryRequest = {
        ...request,
        retry_count: retryCount,
        send_time: new Date(Date.now() + retryDelay * 1000)
      };

      await this.deliveryQueue.add(retryRequest);
      await this.logRetry(request, result.error || 'Unknown error');

    } else {
      // Final failure
      await this.updateDeliveryStatus(
        request.message_id,
        'failed',
        undefined,
        undefined,
        result.error
      );

      // Emit failure event
      await this.emitDeliveryEvent(request, 'failed');

      // Try fallback channel for critical messages
      if (request.priority >= 4) {
        await this.tryFallbackChannel(request);
      }
    }
  }

  private async handleDeliveryError(request: DeliveryRequest, error: any): Promise<void> {
    logger.error(`Delivery error for message ${request.message_id}:`, error);
    
    await this.updateDeliveryStatus(
      request.message_id,
      'failed',
      undefined,
      undefined,
      error instanceof Error ? error.message : String(error)
    );

    await this.emitDeliveryEvent(request, 'failed');
  }

  private calculateRetryDelay(retryCount: number): number {
    const baseDelay = this.config.retry_settings.base_delay_seconds;
    const maxDelay = this.config.retry_settings.max_delay_seconds;
    const jitterFactor = this.config.retry_settings.jitter_factor;
    
    let delay = Math.min(baseDelay * Math.pow(2, retryCount - 1), maxDelay);
    
    // Add jitter
    const jitter = delay * jitterFactor * Math.random();
    delay += jitter;
    
    return delay;
  }

  private async findAlternativeChannel(userId: string, currentChannel: string): Promise<string | null> {
    const fallbackMap: Record<string, string> = {
      'email': 'sms',
      'sms': 'push',
      'push': 'email'
    };

    const fallbackChannel = fallbackMap[currentChannel];
    
    if (fallbackChannel && await this.hasValidContact(userId, fallbackChannel)) {
      return fallbackChannel;
    }

    return null;
  }

  private async tryFallbackChannel(request: DeliveryRequest): Promise<void> {
    const fallbackChannel = await this.findAlternativeChannel(request.user_id, request.channel);
    
    if (fallbackChannel) {
      const fallbackRequest: DeliveryRequest = {
        ...request,
        message_id: `${request.message_id}_fallback`,
        channel: fallbackChannel,
        content: await this.adaptContentForChannel(request.content, fallbackChannel),
        send_time: new Date(),
        metadata: { ...request.metadata, is_fallback: true }
      };

      await this.scheduleDelivery(fallbackRequest);
      
      logger.info(`Scheduled fallback delivery on ${fallbackChannel} for message ${request.message_id}`);
    }
  }

  private async adaptContentForChannel(content: any, channel: string): Promise<any> {
    // Basic content adaptation between channels
    switch (channel) {
      case 'sms':
        return {
          body: content.subject || content.title || content.body || 'Message from User Whisperer'
        };
      case 'push':
        return {
          title: content.subject || content.title || 'User Whisperer',
          body: content.body || content.text_body || 'You have a new message'
        };
      case 'email':
        return {
          subject: content.title || content.subject || 'Message from User Whisperer',
          html_body: content.body || `<p>${content.text_body || 'You have a new message'}</p>`,
          text_body: content.body || content.text_body || 'You have a new message'
        };
      default:
        return content;
    }
  }

  private async validateRequest(request: DeliveryRequest): Promise<ValidationResult> {
    const errors: string[] = [];

    if (!request.message_id) {
      errors.push('message_id is required');
    }

    if (!request.user_id) {
      errors.push('user_id is required');
    }

    if (!request.channel || !['email', 'sms', 'push'].includes(request.channel)) {
      errors.push('channel must be email, sms, or push');
    }

    if (!request.content) {
      errors.push('content is required');
    }

    if (!request.send_time) {
      errors.push('send_time is required');
    }

    if (request.priority && (request.priority < 1 || request.priority > 5)) {
      errors.push('priority must be between 1 and 5');
    }

    return {
      is_valid: errors.length === 0,
      errors
    };
  }

  private async isUnsubscribed(userId: string, channel: string): Promise<boolean> {
    try {
      const query = `
        SELECT COUNT(*) as count
        FROM user_unsubscriptions
        WHERE user_id = $1 AND (channel = $2 OR channel = 'all')
      `;
      
      const result = await database.query(query, [userId, channel]);
      return result.rows[0].count > 0;
    } catch (error) {
      logger.error(`Error checking unsubscription status:`, error);
      return false;
    }
  }

  private async hasValidContact(userId: string, channel: string): Promise<boolean> {
    try {
      let query: string;
      let field: string;
      
      switch (channel) {
        case 'email':
          query = `SELECT email, email_verified FROM user_profiles WHERE user_id = $1`;
          break;
        case 'sms':
          query = `SELECT phone_number, phone_verified FROM user_profiles WHERE user_id = $1`;
          break;
        case 'push':
          query = `SELECT COUNT(*) as token_count FROM user_push_tokens WHERE user_id = $1 AND active = true`;
          break;
        default:
          return false;
      }
      
      const result = await database.query(query, [userId]);
      
      if (result.rows.length === 0) {
        return false;
      }
      
      const row = result.rows[0];
      
      switch (channel) {
        case 'email':
          return !!(row.email && row.email_verified);
        case 'sms':
          return !!(row.phone_number && row.phone_verified);
        case 'push':
          return row.token_count > 0;
        default:
          return false;
      }
    } catch (error) {
      logger.error(`Error checking contact info:`, error);
      return false;
    }
  }

  private async isDuplicateSend(request: DeliveryRequest): Promise<boolean> {
    const key = `duplicate:${request.user_id}:${request.channel}:${request.content.subject || request.content.title || request.content.body}`;
    const exists = await eventsCache.get(key);
    
    if (exists) {
      return true;
    }
    
    // Mark as sent for next 24 hours
    await eventsCache.setWithExpiry(key, '1', 86400);
    return false;
  }

  private async isBusinessHours(userId: string): Promise<boolean> {
    // Get user timezone
    const query = `SELECT timezone FROM user_profiles WHERE user_id = $1`;
    const result = await database.query(query, [userId]);
    
    const timezone = result.rows[0]?.timezone || 'UTC';
    const now = new Date();
    
    // Convert to user's timezone (simplified)
    const userHour = now.getHours(); // In a real implementation, we'd use timezone conversion
    
    // Business hours: 8 AM to 9 PM
    return userHour >= 8 && userHour <= 21;
  }

  private async rescheduleForBusinessHours(request: DeliveryRequest): Promise<void> {
    const nextBusinessHour = new Date();
    nextBusinessHour.setHours(8, 0, 0, 0);
    
    if (nextBusinessHour <= new Date()) {
      nextBusinessHour.setDate(nextBusinessHour.getDate() + 1);
    }
    
    request.send_time = nextBusinessHour;
  }

  private async updateDeliveryStatus(
    messageId: string,
    status: string,
    providerMessageId?: string,
    deliveredAt?: Date,
    error?: string
  ): Promise<void> {
    try {
      const query = `
        INSERT INTO delivery_status (
          message_id, status, provider_message_id, delivered_at, error, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (message_id) DO UPDATE SET
          status = $2,
          provider_message_id = COALESCE($3, delivery_status.provider_message_id),
          delivered_at = COALESCE($4, delivery_status.delivered_at),
          error = $5,
          updated_at = $6
      `;
      
      await database.query(query, [
        messageId,
        status,
        providerMessageId,
        deliveredAt,
        error,
        new Date()
      ]);
    } catch (dbError) {
      logger.error(`Failed to update delivery status:`, dbError);
    }
  }

  private async trackDelivery(request: DeliveryRequest, result: DeliveryResult): Promise<void> {
    // Implementation would track delivery metrics
    logger.info(`Tracking delivery for message ${request.message_id}`, {
      channel: request.channel,
      success: result.success,
      provider_id: result.provider_message_id
    });
  }

  private async emitDeliveryEvent(request: DeliveryRequest, eventType: string): Promise<void> {
    // Implementation would emit events to event bus
    logger.info(`Emitting delivery event: ${eventType} for message ${request.message_id}`);
  }

  private async logSuppression(request: DeliveryRequest, reason: SuppressionReason): Promise<void> {
    logger.info(`Message suppressed: ${request.message_id}`, reason);
  }

  private async logRetry(request: DeliveryRequest, error: string): Promise<void> {
    logger.info(`Retrying message ${request.message_id}, attempt ${(request.retry_count || 0) + 1}: ${error}`);
  }

  private shouldRetry(error: any): boolean {
    const retryableErrors = [
      'timeout',
      'connection',
      'rate limit',
      '500',
      '502',
      '503',
      '504'
    ];

    const errorStr = error.message?.toLowerCase() || '';
    return retryableErrors.some(err => errorStr.includes(err));
  }

  private async delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  public async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    try {
      const emailHealth = await this.emailService.healthCheck();
      const smsHealth = await this.smsService.healthCheck();
      const pushHealth = await this.pushService.healthCheck();
      const queueHealth = await this.deliveryQueue.healthCheck();
      
      const healthy = emailHealth.healthy && smsHealth.healthy && pushHealth.healthy && queueHealth.healthy;
      
      return {
        healthy,
        details: {
          email: emailHealth,
          sms: smsHealth,
          push: pushHealth,
          queue: queueHealth
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