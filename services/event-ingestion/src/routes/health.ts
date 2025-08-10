import { Router, Request, Response } from 'express';
import { database, redis, logger } from '@userwhisperer/shared';

const router = Router();

// GET /health - Basic health check
router.get('/', async (req: Request, res: Response) => {
  try {
    const checks = {
      service: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      pid: process.pid
    };

    res.status(200).json(checks);
  } catch (error) {
    res.status(500).json({
      service: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

// GET /health/detailed - Detailed health check
router.get('/detailed', async (req: Request, res: Response) => {
  const startTime = Date.now();
  const checks: any = {
    service: 'event-ingestion',
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: process.env.npm_package_version || '1.0.0',
    environment: process.env.NODE_ENV || 'development',
    pid: process.pid,
    memory: process.memoryUsage(),
    checks: {}
  };

  let overallHealthy = true;

  // Database health check
  try {
    const dbStart = Date.now();
    const dbHealthy = await database.healthCheck();
    const dbTime = Date.now() - dbStart;

    checks.checks.database = {
      status: dbHealthy ? 'healthy' : 'unhealthy',
      responseTime: `${dbTime}ms`,
      details: dbHealthy ? 'Connection successful' : 'Connection failed'
    };

    if (!dbHealthy) overallHealthy = false;
  } catch (error) {
    checks.checks.database = {
      status: 'unhealthy',
      error: error instanceof Error ? error.message : String(error)
    };
    overallHealthy = false;
  }

  // Redis health check
  try {
    const redisStart = Date.now();
    const redisHealthy = await redis.healthCheck();
    const redisTime = Date.now() - redisStart;

    checks.checks.redis = {
      status: redisHealthy ? 'healthy' : 'unhealthy',
      responseTime: `${redisTime}ms`,
      details: redisHealthy ? 'Connection successful' : 'Connection failed'
    };

    if (!redisHealthy) overallHealthy = false;
  } catch (error) {
    checks.checks.redis = {
      status: 'unhealthy',
      error: error instanceof Error ? error.message : String(error)
    };
    overallHealthy = false;
  }

  // System resource checks
  const memUsage = process.memoryUsage();
  const memoryUsedMB = Math.round(memUsage.heapUsed / 1024 / 1024);
  const memoryTotalMB = Math.round(memUsage.heapTotal / 1024 / 1024);
  const memoryUsagePercent = Math.round((memUsage.heapUsed / memUsage.heapTotal) * 100);

  checks.checks.memory = {
    status: memoryUsagePercent > 90 ? 'warning' : 'healthy',
    used: `${memoryUsedMB}MB`,
    total: `${memoryTotalMB}MB`,
    percentage: `${memoryUsagePercent}%`
  };

  if (memoryUsagePercent > 95) overallHealthy = false;

  // Response time check
  const responseTime = Date.now() - startTime;
  checks.checks.responseTime = {
    status: responseTime > 5000 ? 'warning' : 'healthy',
    value: `${responseTime}ms`
  };

  if (responseTime > 10000) overallHealthy = false;

  // Set overall status
  checks.status = overallHealthy ? 'healthy' : 'unhealthy';

  const statusCode = overallHealthy ? 200 : 503;
  res.status(statusCode).json(checks);
});

// GET /health/ready - Readiness probe
router.get('/ready', async (req: Request, res: Response) => {
  try {
    // Check if service is ready to accept traffic
    const dbHealthy = await database.healthCheck();
    const redisHealthy = await redis.healthCheck();

    if (dbHealthy && redisHealthy) {
      res.status(200).json({
        status: 'ready',
        timestamp: new Date().toISOString(),
        checks: {
          database: 'ready',
          redis: 'ready'
        }
      });
    } else {
      res.status(503).json({
        status: 'not_ready',
        timestamp: new Date().toISOString(),
        checks: {
          database: dbHealthy ? 'ready' : 'not_ready',
          redis: redisHealthy ? 'ready' : 'not_ready'
        }
      });
    }
  } catch (error) {
    logger.error('Readiness check failed', {
      error: error instanceof Error ? error.message : String(error)
    });

    res.status(503).json({
      status: 'not_ready',
      timestamp: new Date().toISOString(),
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

// GET /health/live - Liveness probe
router.get('/live', (req: Request, res: Response) => {
  // Simple liveness check - if we can respond, we're alive
  res.status(200).json({
    status: 'alive',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    pid: process.pid
  });
});

// GET /health/metrics - Basic metrics endpoint
router.get('/metrics', async (req: Request, res: Response) => {
  try {
    const metrics = {
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      cpu: process.cpuUsage(),
      eventLoop: {
        // These would need additional monitoring setup
        lag: 'N/A',
        utilization: 'N/A'
      },
      database: {
        ...database.poolInfo
      }
    };

    res.json(metrics);
  } catch (error) {
    logger.error('Metrics collection failed', {
      error: error instanceof Error ? error.message : String(error)
    });

    res.status(500).json({
      error: 'Failed to collect metrics',
      timestamp: new Date().toISOString()
    });
  }
});

export { router as healthRoutes }; 