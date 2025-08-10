import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';

import { Config, logger, aiOrchestrationClient } from '@userwhisperer/shared';
import { DecisionEngine } from './services/DecisionEngine';
import { 
  DecisionEngineConfig, 
  DecisionRequest, 
  DecisionResponse,
  Channel,
  UrgencyLevel
} from './types';

// Initialize Express app
const app = express();
const port = Config.DECISION_ENGINE_PORT;

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.NODE_ENV === 'production' ? false : true,
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 500, // Limit each IP to 500 requests per windowMs
  message: 'Too many requests from this IP'
});
app.use(limiter);

// Body parsing
app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ extended: true, limit: '5mb' }));

// Request logging
app.use((req, res, next) => {
  logger.info('Incoming request', {
    method: req.method,
    url: req.url,
    ip: req.ip,
    userAgent: req.get('User-Agent')
  });
  next();
});

// Initialize Decision Engine
const decisionConfig: DecisionEngineConfig = {
  intervention_threshold: 0.3,
  intervention_weights: {
    churn: 0.4,
    monetization: 0.3,
    onboarding: 0.2,
    support: 0.3,
    celebration: 0.1,
    event_trigger: 0.5
  },
  fatigue_limits: {
    daily_max: 3,
    weekly_max: 10,
    monthly_max: 25,
    min_gap_hours: 6
  },
  channel_costs: {
    [Channel.EMAIL]: 0.001,
    [Channel.SMS]: 0.01,
    [Channel.PUSH]: 0.0001,
    [Channel.IN_APP]: 0.0,
    [Channel.WEBHOOK]: 0.0001
  },
  ml_model_enabled: false,
  default_timezone: 'UTC'
};

const decisionEngine = new DecisionEngine(decisionConfig);

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const healthCheck = await decisionEngine.healthCheck();
    
    res.status(healthCheck.healthy ? 200 : 503).json({
      service: 'decision-engine',
      status: healthCheck.healthy ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      details: healthCheck.details
    });
  } catch (error) {
    logger.error('Health check failed', { error });
    res.status(503).json({
      service: 'decision-engine',
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: 'Health check failed'
    });
  }
});

// Make decision endpoint
app.post('/v1/decisions/make', async (req, res) => {
  const startTime = Date.now();
  
  try {
    const request: DecisionRequest = req.body;

    if (!request.user_id) {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'user_id is required'
      });
    }

    // Build decision context (in real implementation, this would fetch from services)
    const context = await buildDecisionContext(request);
    
    // Get AI orchestration insights to enhance decision-making
    let aiInsights = null;
    try {
      const orchestrationRequest = {
        user_id: request.user_id,
        user_context: request.user_context,
        behavioral_data: request.user_profile,
        trigger_event: request.trigger_event?.event_type || 'manual_trigger',
        business_objectives: { focus: 'engagement_optimization' },
        constraints: { budget: 'medium', urgency: 'normal' }
      };
      
      aiInsights = await aiOrchestrationClient.orchestrateUser(orchestrationRequest);
      logger.info('AI orchestration insights received', {
        userId: request.user_id,
        confidence: aiInsights.ai_insights.confidence,
        interventionType: aiInsights.strategy_decisions.intervention_type
      });
    } catch (error) {
      logger.warn('AI orchestration unavailable, proceeding with standard decision logic', {
        userId: request.user_id,
        error: error instanceof Error ? error.message : String(error)
      });
    }
    
    const decision = await decisionEngine.makeDecision(context, aiInsights);
    const processingTime = Date.now() - startTime;

    const response: DecisionResponse = {
      decision,
      processing_time_ms: processingTime,
      model_version: 'decision-engine-v1.0'
    };

    return res.json({
      status: 'success',
      ...response,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    const processingTime = Date.now() - startTime;
    
    logger.error('Decision making failed', {
      userId: req.body?.user_id,
      error: error instanceof Error ? error.message : String(error),
      processingTime: `${processingTime}ms`
    });

    return res.status(500).json({
      error: 'Decision making failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      processing_time_ms: processingTime
    });
  }
});

// Batch decisions endpoint
app.post('/v1/decisions/batch', async (req, res) => {
  const startTime = Date.now();
  
  try {
    const { requests } = req.body;

    if (!Array.isArray(requests) || requests.length === 0) {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'requests array is required and cannot be empty'
      });
    }

    if (requests.length > 100) {
      return res.status(400).json({
        error: 'Batch too large',
        message: 'Maximum 100 requests per batch'
      });
    }

    const results = await Promise.allSettled(
      requests.map(async (request: DecisionRequest) => {
        try {
          const context = await buildDecisionContext(request);
          const decision = await decisionEngine.makeDecision(context);
          
          return {
            user_id: request.user_id,
            status: 'success',
            decision
          };
        } catch (error) {
          return {
            user_id: request.user_id,
            status: 'error',
            error: error instanceof Error ? error.message : String(error)
          };
        }
      })
    );

    const responses = results.map((result, index) => {
      const baseResponse = {
        request_index: index,
        user_id: requests[index].user_id,
      };
      
      const resultData = result.status === 'fulfilled' 
        ? result.value 
        : { status: 'error', error: result.reason };
      
      return { ...baseResponse, ...resultData };
    });

    const processingTime = Date.now() - startTime;
    const successCount = responses.filter(r => r.status === 'success').length;

    return res.json({
      status: 'completed',
      total_requests: requests.length,
      successful: successCount,
      failed: requests.length - successCount,
      responses,
      processing_time_ms: processingTime,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    const processingTime = Date.now() - startTime;
    
    logger.error('Batch decision making failed', {
      error: error instanceof Error ? error.message : String(error),
      processingTime: `${processingTime}ms`
    });

    return res.status(500).json({
      error: 'Batch processing failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      processing_time_ms: processingTime
    });
  }
});

// Get decision status endpoint
app.get('/v1/decisions/:decisionId', async (req, res) => {
  try {
    const { decisionId } = req.params;

    // This would typically fetch from database
    // For now, return a mock response
    res.json({
      decision_id: decisionId,
      status: 'completed',
      created_at: new Date().toISOString(),
      message: 'Decision details would be fetched from database'
    });

  } catch (error) {
    logger.error('Failed to get decision status', {
      decisionId: req.params.decisionId,
      error: error instanceof Error ? error.message : String(error)
    });

    res.status(500).json({
      error: 'Failed to get decision status',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Configuration endpoint
app.get('/v1/config', (req, res) => {
  try {
    res.json({
      service: 'decision-engine',
      version: '1.0.0',
      config: {
        intervention_threshold: decisionConfig.intervention_threshold,
        fatigue_limits: decisionConfig.fatigue_limits,
        ml_model_enabled: decisionConfig.ml_model_enabled
      },
      channels: Object.values(Channel),
      urgency_levels: Object.values(UrgencyLevel),
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Failed to get config', {
      error: error instanceof Error ? error.message : String(error)
    });

    res.status(500).json({
      error: 'Failed to get configuration'
    });
  }
});

// Metrics endpoint for Prometheus
app.get('/metrics', (req, res) => {
  res.set('Content-Type', 'text/plain');
  res.send('# Decision Engine Metrics\n# TODO: Implement Prometheus metrics\n');
});

// Error handling middleware
app.use((error: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  logger.error('Unhandled error', {
    error: error.message,
    stack: error.stack,
    url: req.url,
    method: req.method
  });

  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? error.message : 'An unexpected error occurred'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    message: `Route ${req.method} ${req.path} not found`
  });
});

// Helper function to build decision context
async function buildDecisionContext(request: DecisionRequest): Promise<any> {
  // In a real implementation, this would:
  // 1. Fetch user profile from database
  // 2. Get behavioral scores from Behavioral Analysis service
  // 3. Fetch message history
  // 4. Build complete context

  // For now, return mock context
  return {
    user_id: request.user_id,
    trigger_event: request.trigger_event || {
      event_id: 'mock-event',
      event_type: 'session_started',
      timestamp: new Date().toISOString(),
      properties: {},
      context: {}
    },
    behavioral_scores: {
      churn_risk: 0.3,
      upgrade_probability: 0.5,
      engagement_score: 0.6,
      feature_adoption_rate: 0.4,
      support_tickets: 0,
      monetization_score: 0.3,
      days_since_last_active: 2,
      lifecycle_stage: 'activated'
    },
    user_profile: {
      user_id: request.user_id,
      external_user_id: request.user_id,
      lifecycle_stage: 'activated',
      days_since_signup: 30,
      channel_preferences: {
        [Channel.EMAIL]: 0.8,
        [Channel.SMS]: 0.3,
        [Channel.PUSH]: 0.7,
        [Channel.IN_APP]: 0.9,
        [Channel.WEBHOOK]: 0.0
      },
      optimal_send_hours: [9, 14, 20],
      created_at: new Date(),
      metadata: {}
    },
    message_history: [],
    current_time: new Date()
  };
}

// Start server
const server = app.listen(port, () => {
  logger.info('Decision Engine started', {
    port,
    environment: process.env.NODE_ENV || 'development',
    pid: process.pid
  });
});

// Graceful shutdown
const gracefulShutdown = (signal: string) => {
  logger.info(`Received ${signal}, shutting down gracefully`);
  
  server.close(() => {
    logger.info('HTTP server closed');
    process.exit(0);
  });

  // Force close after 30 seconds
  setTimeout(() => {
    logger.error('Could not close connections in time, forcefully shutting down');
    process.exit(1);
  }, 30000);
};

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

export default app; 