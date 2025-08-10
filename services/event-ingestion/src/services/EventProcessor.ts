import { logger, eventsDb, eventsCache } from '@userwhisperer/shared';
import { v4 as uuidv4 } from 'uuid';
import { EventValidator } from './EventValidator';
import { EventEnricher } from './EventEnricher';
import { Deduplicator } from './Deduplicator';
import { BackpressureManager } from './BackpressureManager';

export interface RawEvent {
  event_type: string;
  user_id: string;
  timestamp: string;
  properties?: Record<string, any>;
  context?: {
    ip?: string;
    user_agent?: string;
    session_id?: string;
    device_type?: string;
  };
  app_id?: string;
  event_id?: string;
}

export interface EnrichedEvent extends RawEvent {
  event_id: string;
  app_id: string;
  enrichment: {
    timestamp: string;
    timezone_offset?: number;
    local_timestamp?: string;
  };
  metadata?: {
    processed_at?: string;
    [key: string]: any;
  };
  user_context?: any;
  geo?: {
    country: string;
    region: string;
    city: string;
    timezone: string;
    coordinates: [number, number];
  };
  device?: {
    browser: string;
    browser_version: string;
    os: string;
    os_version: string;
    device_type: string;
    device_vendor?: string;
    device_model?: string;
  };
  session_metrics?: any;
  category: string;
}

export interface ProcessResult {
  status: 'accepted' | 'rejected' | 'duplicate';
  eventId?: string;
  errors?: ValidationError[];
  processingTime: number;
}

export interface ValidationError {
  field: string;
  message: string;
}

export interface EventProcessorConfig {
  batchSize: number;
  maxQueueSize: number;
  deduplicationTtl: number;
  quarantineTtl: number;
}

export class EventProcessor {
  private validator: EventValidator;
  private enricher: EventEnricher;
  private deduplicator: Deduplicator;
  private backpressureManager: BackpressureManager;
  private config: EventProcessorConfig;
  
  constructor(config: EventProcessorConfig) {
    this.config = config;
    this.validator = new EventValidator();
    this.enricher = new EventEnricher();
    this.deduplicator = new Deduplicator();
    this.backpressureManager = new BackpressureManager({
      maxQueueSize: config.maxQueueSize,
      batchSize: config.batchSize,
    });
  }
  
  async processEvent(rawEvent: RawEvent): Promise<ProcessResult> {
    const startTime = Date.now();
    
    try {
      // Step 1: Generate event ID if not provided
      if (!rawEvent.event_id) {
        rawEvent.event_id = `evt_${uuidv4().replace(/-/g, '').substring(0, 16)}`;
      }
      
      // Step 2: Set app_id if not provided (could come from API key in real implementation)
      if (!rawEvent.app_id) {
        rawEvent.app_id = `app_${uuidv4().replace(/-/g, '').substring(0, 16)}`;
      }
      
      // Step 3: Validate event structure
      const validation = await this.validator.validate(rawEvent);
      if (!validation.isValid) {
        await this.quarantineEvent(rawEvent, validation.errors);
        return {
          status: 'rejected',
          errors: validation.errors,
          processingTime: Date.now() - startTime
        };
      }
      
      // Step 4: Check for duplicates
      const isDuplicate = await this.deduplicator.isDuplicate(rawEvent);
      if (isDuplicate) {
        logger.debug('Duplicate event detected', { eventId: rawEvent.event_id });
        return {
          status: 'duplicate',
          processingTime: Date.now() - startTime
        };
      }
      
      // Step 5: Enrich event with context
      const enrichedEvent = await this.enricher.enrich(rawEvent);
      
      // Step 6: Store event
      const eventId = await this.storeEvent(enrichedEvent);
      enrichedEvent.event_id = eventId;
      
      // Step 7: Mark event as processed for deduplication
      await this.deduplicator.markProcessed(rawEvent.event_id!, this.config.deduplicationTtl);
      
      // Step 8: Queue for downstream processing
      await this.queueForDownstream(enrichedEvent);
      
      // Step 9: Update metrics
      await this.updateMetrics(enrichedEvent, Date.now() - startTime);
      
      logger.debug('Event processed successfully', {
        eventId: enrichedEvent.event_id,
        userId: enrichedEvent.user_id,
        eventType: enrichedEvent.event_type,
        processingTime: Date.now() - startTime
      });
      
      return {
        status: 'accepted',
        eventId: enrichedEvent.event_id,
        processingTime: Date.now() - startTime
      };
      
    } catch (error) {
      logger.error('Event processing error', {
        eventId: rawEvent.event_id,
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined
      });
      
      await this.handleError(error, rawEvent);
      throw error;
    }
  }
  
  async processBatch(events: RawEvent[]): Promise<{ accepted: number; rejected: number; errors: any[] }> {
    const results = {
      accepted: 0,
      rejected: 0,
      errors: [] as any[]
    };
    
    const startTime = Date.now();
    
    logger.info('Processing event batch', { 
      batchSize: events.length,
      timestamp: new Date().toISOString()
    });
    
    // Process events in parallel with concurrency limit
    const concurrencyLimit = 10;
    const chunks = this.chunkArray(events, concurrencyLimit);
    
    for (const chunk of chunks) {
      const promises = chunk.map(async (event, index) => {
        try {
          const result = await this.processEvent(event);
          if (result.status === 'accepted') {
            results.accepted++;
          } else if (result.status === 'rejected') {
            results.rejected++;
            results.errors.push({
              index: index,
              eventId: event.event_id,
              errors: result.errors
            });
          }
          // Duplicates are not counted as rejected
        } catch (error) {
          results.rejected++;
          results.errors.push({
            index: index,
            eventId: event.event_id,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      });
      
      await Promise.all(promises);
    }
    
    const processingTime = Date.now() - startTime;
    
    logger.info('Batch processing completed', {
      totalEvents: events.length,
      accepted: results.accepted,
      rejected: results.rejected,
      processingTime: `${processingTime}ms`,
      throughput: `${Math.round(events.length / (processingTime / 1000))} events/sec`
    });
    
    // Update batch metrics
    await this.updateBatchMetrics(events.length, results.accepted, results.rejected, processingTime);
    
    return results;
  }
  
  private async storeEvent(event: EnrichedEvent): Promise<string> {
    try {
      return await eventsDb.insertEvent(event);
    } catch (error) {
      logger.error('Event storage error', {
        eventId: event.event_id,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }
  
  private async queueForDownstream(event: EnrichedEvent): Promise<void> {
    try {
      // Queue for behavioral analysis
      await eventsCache.queueEvent('behavioral-analysis', {
        eventId: event.event_id,
        userId: event.user_id,
        eventType: event.event_type,
        category: event.category,
        timestamp: event.timestamp
      });
      
      // Queue for real-time decision engine if it's a trigger event
      if (this.isTriggerEvent(event)) {
        await eventsCache.queueEvent('decision-engine', {
          eventId: event.event_id,
          userId: event.user_id,
          eventType: event.event_type,
          priority: 'high',
          timestamp: event.timestamp
        });
      }
      
    } catch (error) {
      logger.error('Failed to queue event for downstream processing', {
        eventId: event.event_id,
        error: error instanceof Error ? error.message : String(error)
      });
      // Don't throw here - event is already stored
    }
  }
  
  private isTriggerEvent(event: EnrichedEvent): boolean {
    const triggerEvents = [
      'user_signup',
      'trial_started',
      'subscription_cancelled',
      'payment_failed',
      'feature_limit_reached',
      'error_encountered',
      'support_ticket_created'
    ];
    
    return triggerEvents.includes(event.event_type);
  }
  
  private async quarantineEvent(event: RawEvent, errors: ValidationError[]): Promise<void> {
    try {
      logger.warn('Quarantining invalid event', {
        eventId: event.event_id,
        eventType: event.event_type,
        errors: errors.map(e => `${e.field}: ${e.message}`)
      });
      
      // Store in quarantine for later analysis
      await eventsDb.insertEvent({
        ...event,
        event_id: event.event_id || `quar_${uuidv4().replace(/-/g, '').substring(0, 16)}`,
        app_id: event.app_id || 'unknown',
        event_type: 'quarantined_event',
        properties: {
          original_event: event,
          validation_errors: errors,
          quarantined_at: new Date().toISOString()
        },
        category: 'quarantine',
        enrichment: { timestamp: new Date().toISOString() }
      } as EnrichedEvent);
      
    } catch (error) {
      logger.error('Failed to quarantine event', {
        eventId: event.event_id,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }
  
  private async handleError(error: any, event: RawEvent): Promise<void> {
    try {
      // Log detailed error information
      logger.error('Event processing error details', {
        eventId: event.event_id,
        userId: event.user_id,
        eventType: event.event_type,
        error: error instanceof Error ? {
          name: error.name,
          message: error.message,
          stack: error.stack
        } : String(error)
      });
      
      // Update error metrics
      await eventsCache.incrementCounter('events.processing.errors');
      await eventsCache.incrementCounter(`events.processing.errors.${event.event_type}`);
      
    } catch (metricsError) {
      logger.error('Failed to update error metrics', { error: metricsError });
    }
  }
  
  private async updateMetrics(event: EnrichedEvent, processingTime: number): Promise<void> {
    try {
      // Update counters
      await eventsCache.incrementCounter('events.processed.total');
      await eventsCache.incrementCounter(`events.processed.${event.event_type}`);
      await eventsCache.incrementCounter(`events.processed.category.${event.category}`);
      
      // Update processing time metrics
      const timeKey = `events.processing.time.${event.event_type}`;
      const currentTime = await eventsCache.getCounter(timeKey);
      await eventsCache.incrementCounter(timeKey, processingTime);
      
    } catch (error) {
      logger.error('Failed to update event metrics', {
        eventId: event.event_id,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }
  
  private async updateBatchMetrics(total: number, accepted: number, rejected: number, processingTime: number): Promise<void> {
    try {
      await eventsCache.incrementCounter('events.batch.total');
      await eventsCache.incrementCounter('events.batch.accepted', accepted);
      await eventsCache.incrementCounter('events.batch.rejected', rejected);
      await eventsCache.incrementCounter('events.batch.processing_time', processingTime);
      
    } catch (error) {
      logger.error('Failed to update batch metrics', { error });
    }
  }
  
  private chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }
  
  // Health check method
  async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    try {
      const queueLengths = {
        behavioralAnalysis: await eventsCache.getQueueLength('behavioral-analysis'),
        decisionEngine: await eventsCache.getQueueLength('decision-engine')
      };
      
      const metrics = {
        totalProcessed: await eventsCache.getCounter('events.processed.total'),
        totalErrors: await eventsCache.getCounter('events.processing.errors'),
        batchesProcessed: await eventsCache.getCounter('events.batch.total')
      };
      
      const healthy = queueLengths.behavioralAnalysis < 10000 && queueLengths.decisionEngine < 5000;
      
      return {
        healthy,
        details: {
          queueLengths,
          metrics,
          backpressure: this.backpressureManager.getStatus()
        }
      };
      
    } catch (error) {
      return {
        healthy: false,
        details: {
          error: error instanceof Error ? error.message : String(error)
        }
      };
    }
  }
} 