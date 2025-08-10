import { Router, Request, Response } from 'express';
import { HealthChecker, logger } from '@userwhisperer/shared';

const router = Router();
const healthChecker = new HealthChecker('channel-orchestrator');

// GET /health - Basic health check
router.get('/', async (req: Request, res: Response) => {
  try {
    const health = await healthChecker.performHealthCheck();
    const statusCode = health.status === 'healthy' ? 200 : 503;
    res.status(statusCode).json(health);
  } catch (error) {
    logger.error('Health check failed', { error });
    res.status(500).json({
      service: 'channel-orchestrator',
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

// GET /health/ready - Readiness probe
router.get('/ready', async (req: Request, res: Response) => {
  try {
    const readiness = await healthChecker.checkReadiness();
    const statusCode = readiness.status === 'ready' ? 200 : 503;
    
    res.status(statusCode).json({
      status: readiness.status,
      timestamp: new Date().toISOString(),
      checks: readiness.checks
    });
  } catch (error) {
    logger.error('Readiness check failed', { error });
    res.status(503).json({
      status: 'not_ready',
      timestamp: new Date().toISOString(),
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

// GET /health/live - Liveness probe
router.get('/live', (req: Request, res: Response) => {
  const liveness = healthChecker.checkLiveness();
  res.status(200).json({
    ...liveness,
    timestamp: new Date().toISOString()
  });
});

export { router as healthRoutes };
