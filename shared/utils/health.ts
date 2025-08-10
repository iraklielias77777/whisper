import { database, redis } from '../index';
import { createServiceLogger } from './logger';

export interface HealthCheck {
  status: 'healthy' | 'unhealthy' | 'warning';
  responseTime?: string;
  details?: string;
  error?: string;
}

export interface ServiceHealth {
  service: string;
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  uptime: number;
  version: string;
  environment: string;
  pid: number;
  memory: NodeJS.MemoryUsage;
  checks: {
    database?: HealthCheck;
    redis?: HealthCheck;
    memory?: HealthCheck;
    responseTime?: HealthCheck;
  };
}

export class HealthChecker {
  private serviceName: string;

  constructor(serviceName: string) {
    this.serviceName = serviceName;
  }

  async performHealthCheck(): Promise<ServiceHealth> {
    const startTime = Date.now();
    const checks: ServiceHealth['checks'] = {};
    let overallHealthy = true;

    // Database health check
    try {
      const dbStart = Date.now();
      const dbHealthy = await database.healthCheck();
      const dbTime = Date.now() - dbStart;

      checks.database = {
        status: dbHealthy ? 'healthy' : 'unhealthy',
        responseTime: `${dbTime}ms`,
        details: dbHealthy ? 'Connection successful' : 'Connection failed'
      };

      if (!dbHealthy) overallHealthy = false;
    } catch (error) {
      checks.database = {
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

      checks.redis = {
        status: redisHealthy ? 'healthy' : 'unhealthy',
        responseTime: `${redisTime}ms`,
        details: redisHealthy ? 'Connection successful' : 'Connection failed'
      };

      if (!redisHealthy) overallHealthy = false;
    } catch (error) {
      checks.redis = {
        status: 'unhealthy',
        error: error instanceof Error ? error.message : String(error)
      };
      overallHealthy = false;
    }

    // Memory health check
    const memUsage = process.memoryUsage();
    const memoryUsedMB = Math.round(memUsage.heapUsed / 1024 / 1024);
    const memoryTotalMB = Math.round(memUsage.heapTotal / 1024 / 1024);
    const memoryUsagePercent = Math.round((memUsage.heapUsed / memUsage.heapTotal) * 100);

    checks.memory = {
      status: memoryUsagePercent > 90 ? 'warning' : 'healthy',
      details: `${memoryUsedMB}MB / ${memoryTotalMB}MB (${memoryUsagePercent}%)`
    };

    if (memoryUsagePercent > 95) overallHealthy = false;

    // Response time check
    const responseTime = Date.now() - startTime;
    checks.responseTime = {
      status: responseTime > 5000 ? 'warning' : 'healthy',
      responseTime: `${responseTime}ms`
    };

    if (responseTime > 10000) overallHealthy = false;

    return {
      service: this.serviceName,
      status: overallHealthy ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: process.env.npm_package_version || '1.0.0',
      environment: process.env.NODE_ENV || 'development',
      pid: process.pid,
      memory: memUsage,
      checks
    };
  }

  async checkReadiness(): Promise<{ status: 'ready' | 'not_ready'; checks: Record<string, string> }> {
    try {
      const dbHealthy = await database.healthCheck();
      const redisHealthy = await redis.healthCheck();

      const checks = {
        database: dbHealthy ? 'ready' : 'not_ready',
        redis: redisHealthy ? 'ready' : 'not_ready'
      };

      const status = dbHealthy && redisHealthy ? 'ready' : 'not_ready';

      return { status, checks };
    } catch (error) {
      const logger = createServiceLogger('health-check');
      logger.error('Readiness check failed', { error });
      return {
        status: 'not_ready',
        checks: {
          database: 'not_ready',
          redis: 'not_ready'
        }
      };
    }
  }

  checkLiveness(): { status: 'alive'; uptime: number; pid: number } {
    return {
      status: 'alive',
      uptime: process.uptime(),
      pid: process.pid
    };
  }
}
