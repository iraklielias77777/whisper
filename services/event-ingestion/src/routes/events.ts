import { Router, Request, Response } from 'express';
import { Config, logger } from '@userwhisperer/shared';
import { EventProcessor, RawEvent } from '../services/EventProcessor';

const router = Router();

// Initialize EventProcessor
const eventProcessor = new EventProcessor({
  batchSize: Config.QUEUE_BATCH_SIZE,
  maxQueueSize: Config.QUEUE_MAX_SIZE,
  deduplicationTtl: Config.EVENT_DEDUPLICATION_TTL,
  quarantineTtl: Config.EVENT_QUARANTINE_TTL,
});

// POST /v1/events/track - Track a single event
router.post('/track', async (req: Request, res: Response) => {
  const startTime = Date.now();
  
  try {
    const event: RawEvent = req.body;
    
    // Add timestamp if not provided
    if (!event.timestamp) {
      event.timestamp = new Date().toISOString();
    }
    
    // Add context from request
    if (!event.context) {
      event.context = {};
    }
    
    // Enrich with request context
    event.context.ip = event.context.ip || req.ip || req.connection.remoteAddress;
    event.context.user_agent = event.context.user_agent || req.get('User-Agent');
    
    const result = await eventProcessor.processEvent(event);
    
    const responseTime = Date.now() - startTime;
    
    if (result.status === 'accepted') {
      logger.info('Event tracked successfully', {
        eventId: result.eventId,
        userId: event.user_id,
        eventType: event.event_type,
        responseTime: `${responseTime}ms`,
        ip: req.ip
      });
      
      res.status(202).json({
        status: 'accepted',
        event_id: result.eventId,
        processing_time: result.processingTime
      });
    } else if (result.status === 'duplicate') {
      logger.debug('Duplicate event ignored', {
        userId: event.user_id,
        eventType: event.event_type,
        responseTime: `${responseTime}ms`
      });
      
      res.status(202).json({
        status: 'duplicate',
        message: 'Event already processed',
        processing_time: result.processingTime
      });
    } else {
      logger.warn('Event rejected', {
        userId: event.user_id,
        eventType: event.event_type,
        errors: result.errors,
        responseTime: `${responseTime}ms`
      });
      
      res.status(400).json({
        status: 'rejected',
        errors: result.errors,
        processing_time: result.processingTime
      });
    }
    
  } catch (error) {
    const responseTime = Date.now() - startTime;
    
    logger.error('Event tracking failed', {
      error: error instanceof Error ? error.message : String(error),
      body: req.body,
      responseTime: `${responseTime}ms`,
      ip: req.ip
    });
    
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to process event'
    });
  }
});

// POST /v1/events/batch - Track multiple events
router.post('/batch', async (req: Request, res: Response) => {
  const startTime = Date.now();
  
  try {
    const { events } = req.body;
    
    if (!Array.isArray(events)) {
      return res.status(400).json({
        error: 'Invalid payload',
        message: 'events must be an array'
      });
    }
    
    if (events.length === 0) {
      return res.status(400).json({
        error: 'Invalid payload',
        message: 'events array cannot be empty'
      });
    }
    
    if (events.length > 1000) {
      return res.status(400).json({
        error: 'Batch too large',
        message: 'Maximum 1000 events per batch'
      });
    }
    
    // Enrich events with request context
    const enrichedEvents: RawEvent[] = events.map((event: any) => {
      // Add timestamp if not provided
      if (!event.timestamp) {
        event.timestamp = new Date().toISOString();
      }
      
      // Add context from request
      if (!event.context) {
        event.context = {};
      }
      
      event.context.ip = event.context.ip || req.ip || req.connection.remoteAddress;
      event.context.user_agent = event.context.user_agent || req.get('User-Agent');
      
      return event;
    });
    
    const result = await eventProcessor.processBatch(enrichedEvents);
    const responseTime = Date.now() - startTime;
    
    logger.info('Event batch processed', {
      totalEvents: events.length,
      accepted: result.accepted,
      rejected: result.rejected,
      responseTime: `${responseTime}ms`,
      ip: req.ip
    });
    
    return res.status(202).json({
      status: 'processed',
      total: events.length,
      accepted: result.accepted,
      rejected: result.rejected,
      errors: result.errors.length > 0 ? result.errors : undefined,
      processing_time: responseTime
    });
    
  } catch (error) {
    const responseTime = Date.now() - startTime;
    
    logger.error('Batch processing failed', {
      error: error instanceof Error ? error.message : String(error),
      eventsCount: req.body?.events?.length || 0,
      responseTime: `${responseTime}ms`,
      ip: req.ip
    });
    
    return res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to process event batch'
    });
  }
});

// POST /v1/events/identify - Identify a user
router.post('/identify', async (req: Request, res: Response) => {
  const startTime = Date.now();
  
  try {
    const { user_id, traits } = req.body;
    
    if (!user_id) {
      return res.status(400).json({
        error: 'Missing required field',
        message: 'user_id is required'
      });
    }
    
    // Create an identify event
    const identifyEvent: RawEvent = {
      event_type: 'user_identified',
      user_id: user_id,
      timestamp: new Date().toISOString(),
      properties: {
        traits: traits || {},
        identified_at: new Date().toISOString()
      },
      context: {
        ip: req.ip || req.connection.remoteAddress,
        user_agent: req.get('User-Agent')
      }
    };
    
    const result = await eventProcessor.processEvent(identifyEvent);
    const responseTime = Date.now() - startTime;
    
    if (result.status === 'accepted') {
      logger.info('User identified successfully', {
        eventId: result.eventId,
        userId: user_id,
        responseTime: `${responseTime}ms`,
        ip: req.ip
      });
      
      return res.status(202).json({
        status: 'accepted',
        event_id: result.eventId,
        user_id: user_id,
        processing_time: result.processingTime
      });
    } else {
      logger.warn('User identification rejected', {
        userId: user_id,
        errors: result.errors,
        responseTime: `${responseTime}ms`
      });
      
      return res.status(400).json({
        status: 'rejected',
        errors: result.errors,
        processing_time: result.processingTime
      });
    }
    
  } catch (error) {
    const responseTime = Date.now() - startTime;
    
    logger.error('User identification failed', {
      error: error instanceof Error ? error.message : String(error),
      body: req.body,
      responseTime: `${responseTime}ms`,
      ip: req.ip
    });
    
    return res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to process identification'
    });
  }
});

// GET /v1/events/status - Get processing status
router.get('/status', async (req: Request, res: Response) => {
  try {
    const healthCheck = await eventProcessor.healthCheck();
    
    res.json({
      status: healthCheck.healthy ? 'healthy' : 'degraded',
      timestamp: new Date().toISOString(),
      details: healthCheck.details
    });
    
  } catch (error) {
    logger.error('Status check failed', {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: 'Failed to get status'
    });
  }
});

export { router as eventRoutes }; 