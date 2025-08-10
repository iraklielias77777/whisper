import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { Config, logger, database, redis } from '@userwhisperer/shared';
import { EventProcessor } from './services/EventProcessor';
import { healthRoutes } from './routes/health';
import { eventRoutes } from './routes/events';
import { metricsMiddleware } from './middleware/metrics';
import { errorHandler } from './middleware/errorHandler';
import { requestLogger } from './middleware/requestLogger';

const app = express();

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  credentials: true,
}));

// Rate limiting
if (Config.RATE_LIMIT_ENABLED) {
  const limiter = rateLimit({
    windowMs: Config.RATE_LIMIT_WINDOW_MS,
    max: Config.RATE_LIMIT_MAX_REQUESTS,
    message: {
      error: 'Too many requests from this IP',
      retryAfter: Math.ceil(Config.RATE_LIMIT_WINDOW_MS / 1000),
    },
    standardHeaders: true,
    legacyHeaders: false,
  });
  app.use('/v1/events', limiter);
}

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Middleware
app.use(requestLogger);
if (Config.METRICS_ENABLED) {
  app.use(metricsMiddleware);
}

// Routes
app.use('/health', healthRoutes);
app.use('/v1/events', eventRoutes);

// Metrics endpoint
if (Config.METRICS_ENABLED) {
  const promClient = require('prom-client');
  app.get('/metrics', (req, res) => {
    res.set('Content-Type', promClient.register.contentType);
    res.end(promClient.register.metrics());
  });
}

// Error handling
app.use(errorHandler);

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Route not found',
    path: req.originalUrl,
    method: req.method,
  });
});

// Graceful shutdown
const gracefulShutdown = async (signal: string) => {
  logger.info(`Received ${signal}, starting graceful shutdown...`);
  
  server.close(async () => {
    logger.info('HTTP server closed');
    
    try {
      // Close database connections
      await database.close();
      logger.info('Database connections closed');
      
      // Close Redis connections
      await redis.close();
      logger.info('Redis connections closed');
      
      logger.info('Graceful shutdown complete');
      process.exit(0);
    } catch (error) {
      logger.error('Error during graceful shutdown', { error });
      process.exit(1);
    }
  });
  
  // Force shutdown after 30 seconds
  setTimeout(() => {
    logger.error('Forcing shutdown after timeout');
    process.exit(1);
  }, 30000);
};

// Health check function
const performHealthCheck = async (): Promise<boolean> => {
  try {
    // Check database
    const dbHealthy = await database.healthCheck();
    if (!dbHealthy) {
      logger.error('Database health check failed');
      return false;
    }
    
    // Check Redis
    const redisHealthy = await redis.healthCheck();
    if (!redisHealthy) {
      logger.error('Redis health check failed');
      return false;
    }
    
    return true;
  } catch (error) {
    logger.error('Health check error', { error });
    return false;
  }
};

// Start server
const server = app.listen(Config.EVENT_INGESTION_PORT, Config.EVENT_INGESTION_HOST, async () => {
  logger.info('Event Ingestion Service starting...', {
    port: Config.EVENT_INGESTION_PORT,
    host: Config.EVENT_INGESTION_HOST,
    environment: Config.NODE_ENV,
  });
  
  // Perform initial health check
  const isHealthy = await performHealthCheck();
  if (!isHealthy) {
    logger.error('Initial health check failed');
    process.exit(1);
  }
  
  logger.info('Event Ingestion Service started successfully', {
    url: `http://${Config.EVENT_INGESTION_HOST}:${Config.EVENT_INGESTION_PORT}`,
    pid: process.pid,
  });
});

// Error handling
server.on('error', (error: any) => {
  if (error.syscall !== 'listen') {
    throw error;
  }
  
  switch (error.code) {
    case 'EACCES':
      logger.error(`Port ${Config.EVENT_INGESTION_PORT} requires elevated privileges`);
      process.exit(1);
      break;
    case 'EADDRINUSE':
      logger.error(`Port ${Config.EVENT_INGESTION_PORT} is already in use`);
      process.exit(1);
      break;
    default:
      throw error;
  }
});

// Graceful shutdown handlers
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Unhandled errors
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', { promise, reason });
});

process.on('uncaughtException', (error) => {
  logger.error('Uncaught Exception:', { error });
  process.exit(1);
});

export default app; 