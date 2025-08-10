import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';

import { Config, logger } from '@userwhisperer/shared';
import { BehavioralAnalysisEngine } from './services/BehavioralAnalysisEngine';
import { AnalysisConfig, UserEvent } from './types';

// Initialize Express app
const app = express();
const port = Config.BEHAVIORAL_ANALYSIS_PORT;

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.NODE_ENV === 'production' ? false : true,
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // Limit each IP to 1000 requests per windowMs
  message: 'Too many requests from this IP'
});
app.use(limiter);

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

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

// Initialize Behavioral Analysis Engine
const analysisConfig: AnalysisConfig = {
  batch_size: 50,
  analysis_window_days: 30,
  min_events_required: 5,
  pattern_detection_enabled: true,
  ml_models_enabled: false, // Disabled for now
  real_time_processing: true,
  feature_calculation_enabled: true
};

const behavioralEngine = new BehavioralAnalysisEngine(analysisConfig);

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const healthCheck = await behavioralEngine.healthCheck();
    
    res.status(healthCheck.healthy ? 200 : 503).json({
      service: 'behavioral-analysis',
      status: healthCheck.healthy ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      details: healthCheck.details
    });
  } catch (error) {
    logger.error('Health check failed', { error });
    res.status(503).json({
      service: 'behavioral-analysis',
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: 'Health check failed'
    });
  }
});

// Analyze single user endpoint
app.post('/v1/analyze/user/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const { events, messageHistory } = req.body;

    if (!events || !Array.isArray(events)) {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'events array is required'
      });
    }

    const result = await behavioralEngine.analyzeBehavior(userId, events, messageHistory);

    return res.json({
      status: 'success',
      result,
      processed_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('User analysis failed', {
      userId: req.params.userId,
      error: error instanceof Error ? error.message : String(error)
    });

    return res.status(500).json({
      error: 'Analysis failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get user behavioral status endpoint
app.get('/v1/status/user/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const status = await behavioralEngine.getUserBehavioralStatus(userId);

    res.json({
      status: 'success',
      user_id: userId,
      ...status,
      retrieved_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Status retrieval failed', {
      userId: req.params.userId,
      error: error instanceof Error ? error.message : String(error)
    });

    res.status(500).json({
      error: 'Status retrieval failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Batch analysis endpoint
app.post('/v1/analyze/batch', async (req, res) => {
  try {
    const { userEvents } = req.body;

    if (!userEvents || typeof userEvents !== 'object') {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'userEvents object is required'
      });
    }

    // Convert to Map
    const userEventMap = new Map(Object.entries(userEvents));
    const results = await behavioralEngine.analyzeBatch(userEventMap as Map<string, UserEvent[]>);

    // Convert Map back to object for JSON response
    const resultsObject = Object.fromEntries(results);

    return res.json({
      status: 'success',
      results: resultsObject,
      processed_count: results.size,
      processed_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Batch analysis failed', {
      error: error instanceof Error ? error.message : String(error)
    });

    return res.status(500).json({
      error: 'Batch analysis failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Metrics endpoint for Prometheus
app.get('/metrics', (req, res) => {
  // This would export Prometheus metrics
  res.set('Content-Type', 'text/plain');
  res.send('# Behavioral Analysis Service Metrics\n# TODO: Implement Prometheus metrics\n');
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

// Start server
const server = app.listen(port, () => {
  logger.info('Behavioral Analysis Service started', {
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