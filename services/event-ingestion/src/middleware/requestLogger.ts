import { Request, Response, NextFunction } from 'express';
import { logger } from '@userwhisperer/shared';

export interface RequestWithStartTime extends Request {
  startTime?: number;
}

// Request logger middleware
export const requestLogger = (
  req: RequestWithStartTime,
  res: Response,
  next: NextFunction
) => {
  // Record start time
  req.startTime = Date.now();

  // Generate request ID if not present
  if (!req.headers['x-request-id']) {
    req.headers['x-request-id'] = generateRequestId();
  }

  // Log incoming request
  logger.info('Incoming Request', {
    requestId: req.headers['x-request-id'],
    method: req.method,
    url: req.url,
    userAgent: req.get('User-Agent'),
    ip: req.ip || req.connection?.remoteAddress,
    contentType: req.get('Content-Type'),
    contentLength: req.get('Content-Length'),
    origin: req.get('Origin'),
    referer: req.get('Referer'),
    timestamp: new Date().toISOString()
  });

  // Log response when finished
  res.on('finish', () => {
    const duration = Date.now() - (req.startTime || Date.now());
    const responseSize = res.get('Content-Length');

    const logData = {
      requestId: req.headers['x-request-id'],
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
      duration: `${duration}ms`,
      userAgent: req.get('User-Agent'),
      ip: req.ip || req.connection?.remoteAddress,
      responseSize: responseSize ? `${responseSize} bytes` : undefined,
      timestamp: new Date().toISOString()
    };

    // Log based on status code
    if (res.statusCode >= 500) {
      logger.error('Server Error Response', logData);
    } else if (res.statusCode >= 400) {
      logger.warn('Client Error Response', logData);
    } else if (res.statusCode >= 300) {
      logger.info('Redirect Response', logData);
    } else {
      logger.info('Success Response', logData);
    }
  });

  // Log if client disconnects
  req.on('close', () => {
    if (!res.headersSent) {
      const duration = Date.now() - (req.startTime || Date.now());
      
      logger.warn('Client Disconnected', {
        requestId: req.headers['x-request-id'],
        method: req.method,
        url: req.url,
        duration: `${duration}ms`,
        ip: req.ip || req.connection?.remoteAddress,
        timestamp: new Date().toISOString()
      });
    }
  });

  next();
};

// Request logger for sensitive endpoints (excludes body)
export const sensitiveRequestLogger = (
  req: RequestWithStartTime,
  res: Response,
  next: NextFunction
) => {
  // Record start time
  req.startTime = Date.now();

  // Generate request ID if not present
  if (!req.headers['x-request-id']) {
    req.headers['x-request-id'] = generateRequestId();
  }

  // Log incoming request (without sensitive data)
  logger.info('Incoming Sensitive Request', {
    requestId: req.headers['x-request-id'],
    method: req.method,
    url: req.url,
    userAgent: req.get('User-Agent'),
    ip: req.ip || req.connection?.remoteAddress,
    contentType: req.get('Content-Type'),
    timestamp: new Date().toISOString(),
    note: 'Body and query parameters excluded for security'
  });

  // Log response when finished
  res.on('finish', () => {
    const duration = Date.now() - (req.startTime || Date.now());

    const logData = {
      requestId: req.headers['x-request-id'],
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
      duration: `${duration}ms`,
      ip: req.ip || req.connection?.remoteAddress,
      timestamp: new Date().toISOString()
    };

    if (res.statusCode >= 400) {
      logger.warn('Sensitive Request Error', logData);
    } else {
      logger.info('Sensitive Request Success', logData);
    }
  });

  next();
};

// Performance logger for slow requests
export const performanceLogger = (threshold: number = 1000) => {
  return (req: RequestWithStartTime, res: Response, next: NextFunction) => {
    req.startTime = Date.now();

    res.on('finish', () => {
      const duration = Date.now() - (req.startTime || Date.now());

      if (duration > threshold) {
        logger.warn('Slow Request Detected', {
          requestId: req.headers['x-request-id'],
          method: req.method,
          url: req.url,
          duration: `${duration}ms`,
          threshold: `${threshold}ms`,
          statusCode: res.statusCode,
          ip: req.ip || req.connection?.remoteAddress,
          userAgent: req.get('User-Agent'),
          timestamp: new Date().toISOString()
        });
      }
    });

    next();
  };
};

// Error request logger
export const errorRequestLogger = (
  err: any,
  req: RequestWithStartTime,
  res: Response,
  next: NextFunction
) => {
  const duration = Date.now() - (req.startTime || Date.now());

  logger.error('Request Error', {
    requestId: req.headers['x-request-id'],
    method: req.method,
    url: req.url,
    duration: `${duration}ms`,
    error: {
      message: err.message,
      stack: err.stack,
      statusCode: err.statusCode
    },
    ip: req.ip || req.connection?.remoteAddress,
    userAgent: req.get('User-Agent'),
    body: req.body,
    query: req.query,
    params: req.params,
    timestamp: new Date().toISOString()
  });

  next(err);
};

// Generate unique request ID
const generateRequestId = (): string => {
  return `req_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
};

// Skip logging for certain paths
export const skipLogger = (paths: string[]) => {
  return (req: Request, res: Response, next: NextFunction) => {
    const shouldSkip = paths.some(path => {
      if (path.includes('*')) {
        const pattern = path.replace(/\*/g, '.*');
        return new RegExp(`^${pattern}$`).test(req.path);
      }
      return req.path === path;
    });

    if (shouldSkip) {
      return next();
    }

    return requestLogger(req as RequestWithStartTime, res, next);
  };
};

// Export commonly used configurations
export const loggerConfig = {
  // Skip health check endpoints
  skipHealthChecks: skipLogger(['/health', '/health/*', '/metrics']),
  
  // Performance logging for API endpoints
  apiPerformance: performanceLogger(500), // 500ms threshold for API calls
  
  // Performance logging for event processing
  eventPerformance: performanceLogger(100), // 100ms threshold for event processing
};

export default requestLogger; 