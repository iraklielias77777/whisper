import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';

import { Config, logger } from '@userwhisperer/shared';
import { ChannelOrchestrator } from './services/ChannelOrchestrator';
import { 
  ChannelOrchestratorConfig, 
  DeliveryRequest, 
  DeliveryResult 
} from './types';
import { v4 as uuidv4 } from 'uuid';

const app = express();
const port = Config.CHANNEL_ORCHESTRATOR_PORT || 3006;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // Limit each IP to 1000 requests per windowMs
  message: 'Too many requests from this IP, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});
app.use(limiter);

// Request logging
app.use((req, res, next) => {
  logger.info(`${req.method} ${req.path}`, {
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    requestId: req.headers['x-request-id'] || uuidv4()
  });
  next();
});

// Initialize Channel Orchestrator
const orchestratorConfig: ChannelOrchestratorConfig = {
  redis_url: Config.REDIS_URL,
  postgres_url: Config.DATABASE_URL,
  sendgrid: {
    api_key: Config.SENDGRID_API_KEY,
    from_email: Config.SENDGRID_FROM_EMAIL,
    from_name: Config.SENDGRID_FROM_NAME,
    tracking_domain: Config.SENDGRID_TRACKING_DOMAIN,
    webhook_url: Config.SENDGRID_WEBHOOK_URL
  },
  twilio: {
    account_sid: Config.TWILIO_ACCOUNT_SID,
    auth_token: Config.TWILIO_AUTH_TOKEN,
    from_number: Config.TWILIO_FROM_NUMBER,
    messaging_service_sid: Config.TWILIO_MESSAGING_SERVICE_SID,
    status_callback_url: Config.TWILIO_STATUS_CALLBACK_URL
  },
  firebase: {
    service_account_path: Config.FIREBASE_SERVICE_ACCOUNT_PATH,
    dry_run: Config.FIREBASE_DRY_RUN === 'true'
  },
  rate_limits: {
    email: {
      per_user_per_hour: parseInt(Config.EMAIL_RATE_LIMIT_PER_HOUR || '10'),
      per_user_per_day: parseInt(Config.EMAIL_RATE_LIMIT_PER_DAY || '50'),
      global_per_minute: parseInt(Config.EMAIL_GLOBAL_RATE_LIMIT_PER_MINUTE || '100')
    },
    sms: {
      per_user_per_hour: parseInt(Config.SMS_RATE_LIMIT_PER_HOUR || '5'),
      per_user_per_day: parseInt(Config.SMS_RATE_LIMIT_PER_DAY || '20'),
      global_per_minute: parseInt(Config.SMS_GLOBAL_RATE_LIMIT_PER_MINUTE || '50')
    },
    push: {
      per_user_per_hour: parseInt(Config.PUSH_RATE_LIMIT_PER_HOUR || '20'),
      per_user_per_day: parseInt(Config.PUSH_RATE_LIMIT_PER_DAY || '100'),
      global_per_minute: parseInt(Config.PUSH_GLOBAL_RATE_LIMIT_PER_MINUTE || '200')
    }
  },
  retry_settings: {
    max_retries: parseInt(Config.MAX_DELIVERY_RETRIES || '3'),
    base_delay_seconds: parseInt(Config.RETRY_BASE_DELAY_SECONDS || '60'),
    max_delay_seconds: parseInt(Config.RETRY_MAX_DELAY_SECONDS || '3600'),
    jitter_factor: parseFloat(Config.RETRY_JITTER_FACTOR || '0.1')
  }
};

const orchestrator = new ChannelOrchestrator(orchestratorConfig);

// API Routes
app.get('/health', async (req, res) => {
  try {
    const health = await orchestrator.healthCheck();
    
    res.status(health.healthy ? 200 : 503).json({
      status: health.healthy ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      service: 'channel-orchestrator',
      version: process.env.npm_package_version || '1.0.0',
      details: health.details
    });
  } catch (error) {
    logger.error('Health check failed:', error);
    res.status(503).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      service: 'channel-orchestrator',
      error: 'Health check failed'
    });
  }
});

app.post('/v1/delivery/schedule', async (req, res) => {
  try {
    const deliveryRequest = await buildDeliveryRequest(req.body);
    
    const messageId = await orchestrator.scheduleDelivery(deliveryRequest);
    
    return res.status(202).json({
      status: 'scheduled',
      message_id: messageId,
      scheduled_for: deliveryRequest.send_time
    });
    
  } catch (error) {
    logger.error('Failed to schedule delivery:', error);
    return res.status(400).json({
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }
});

app.post('/v1/delivery/send', async (req, res) => {
  try {
    const deliveryRequest = await buildDeliveryRequest(req.body);
    
    // Set immediate send time
    deliveryRequest.send_time = new Date();
    
    const result = await orchestrator.deliverMessage(deliveryRequest);
    
    return res.status(result.success ? 200 : 422).json({
      status: result.success ? 'delivered' : 'failed',
      message_id: deliveryRequest.message_id,
      provider_message_id: result.provider_message_id,
      error: result.error,
      delivered_at: result.delivered_at
    });
    
  } catch (error) {
    logger.error('Failed to send message:', error);
    return res.status(500).json({
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }
});

app.post('/v1/delivery/batch', async (req, res) => {
  try {
    const { requests } = req.body;
    
    if (!Array.isArray(requests) || requests.length === 0) {
      return res.status(400).json({
        status: 'error',
        message: 'requests array is required and must not be empty'
      });
    }
    
    if (requests.length > 100) {
      return res.status(400).json({
        status: 'error',
        message: 'Maximum 100 requests per batch'
      });
    }
    
    const deliveryRequests = await Promise.all(
      requests.map(request => buildDeliveryRequest(request))
    );
    
    const results = await Promise.allSettled(
      deliveryRequests.map(request => orchestrator.scheduleDelivery(request))
    );
    
    const response = {
      status: 'batch_processed',
      total: requests.length,
      scheduled: 0,
      failed: 0,
      results: results.map((result, index) => {
        if (result.status === 'fulfilled') {
          return {
            message_id: result.value,
            status: 'scheduled',
            request_index: index
          };
        } else {
          return {
            status: 'failed',
            error: result.reason.message,
            request_index: index
          };
        }
      })
    };
    
    response.scheduled = response.results.filter(r => r.status === 'scheduled').length;
    response.failed = response.results.filter(r => r.status === 'failed').length;
    
    return res.status(207).json(response);
    
  } catch (error) {
    logger.error('Failed to process batch:', error);
    return res.status(500).json({
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }
});

app.get('/v1/delivery/:messageId/status', async (req, res) => {
  try {
    const { messageId } = req.params;
    
    // This would query the delivery status from database
    // For now, return a mock response
    return res.json({
      message_id: messageId,
      status: 'delivered',
      scheduled_for: new Date(),
      sent_at: new Date(),
      delivered_at: new Date()
    });
    
  } catch (error) {
    logger.error('Failed to get delivery status:', error);
    return res.status(500).json({
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }
});

app.get('/v1/stats/queue', async (req, res) => {
  try {
    // Get queue statistics
    return res.json({
      timestamp: new Date().toISOString(),
      queue_stats: {
        total_queued: 0,
        processing: 0,
        failed: 0
      }
    });
    
  } catch (error) {
    logger.error('Failed to get queue stats:', error);
    return res.status(500).json({
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }
});

app.get('/v1/stats/rate-limits/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const { channel } = req.query;
    
    // Get user rate limit statistics
    return res.json({
      user_id: userId,
      channel: channel || 'all',
      timestamp: new Date().toISOString(),
      rate_limits: {
        email: { used: 0, limit: 10, remaining: 10 },
        sms: { used: 0, limit: 5, remaining: 5 },
        push: { used: 0, limit: 20, remaining: 20 }
      }
    });
    
  } catch (error) {
    logger.error('Failed to get rate limit stats:', error);
    return res.status(500).json({
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }
});

app.get('/metrics', (req, res) => {
  // Return Prometheus metrics
  res.set('Content-Type', 'text/plain');
  res.send('# Channel Orchestrator Metrics\n# TODO: Implement metrics collection\n');
});

// Webhook endpoints for delivery status updates
app.post('/webhooks/sendgrid', async (req, res) => {
  try {
    // Handle SendGrid webhook events
    const events = req.body;
    
    if (Array.isArray(events)) {
      for (const event of events) {
        logger.info('SendGrid webhook event:', event);
        // Process delivery status update
      }
    }
    
    res.status(200).send('OK');
  } catch (error) {
    logger.error('SendGrid webhook error:', error);
    res.status(500).send('Error processing webhook');
  }
});

app.post('/webhooks/twilio', async (req, res) => {
  try {
    // Handle Twilio webhook events
    logger.info('Twilio webhook event:', req.body);
    // Process SMS delivery status update
    
    res.status(200).send('OK');
  } catch (error) {
    logger.error('Twilio webhook error:', error);
    res.status(500).send('Error processing webhook');
  }
});

// Error handling middleware
app.use((err: any, req: any, res: any, next: any) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({
    status: 'error',
    message: 'Internal server error'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    status: 'error',
    message: 'Endpoint not found'
  });
});

async function buildDeliveryRequest(requestBody: any): Promise<DeliveryRequest> {
  const {
    user_id,
    channel,
    content,
    send_time,
    priority = 3,
    max_retries = 3,
    metadata = {}
  } = requestBody;

  // Validate required fields
  if (!user_id) {
    throw new Error('user_id is required');
  }

  if (!channel || !['email', 'sms', 'push'].includes(channel)) {
    throw new Error('channel must be email, sms, or push');
  }

  if (!content) {
    throw new Error('content is required');
  }

  const deliveryRequest: DeliveryRequest = {
    message_id: uuidv4(),
    user_id,
    channel,
    content,
    send_time: send_time ? new Date(send_time) : new Date(),
    priority: Math.max(1, Math.min(5, priority)),
    retry_count: 0,
    max_retries: Math.max(0, Math.min(10, max_retries)),
    metadata
  };

  return deliveryRequest;
}

// Start server
async function startServer() {
  try {
    // Initialize the orchestrator
    await orchestrator.initialize();
    
    app.listen(port, () => {
      logger.info(`Channel Orchestrator service started on port ${port}`);
    });
  } catch (error) {
    logger.error('Failed to start Channel Orchestrator service:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});

// Start the server
startServer();

export default app; 