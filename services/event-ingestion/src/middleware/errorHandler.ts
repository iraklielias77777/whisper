import { Request, Response, NextFunction } from 'express';
import { logger } from '@userwhisperer/shared';

export interface AppError extends Error {
  statusCode?: number;
  status?: string;
  operational?: boolean;
}

export const createError = (
  message: string,
  statusCode: number = 500,
  operational: boolean = true
): AppError => {
  const error: AppError = new Error(message);
  error.statusCode = statusCode;
  error.status = statusCode >= 400 && statusCode < 500 ? 'fail' : 'error';
  error.operational = operational;
  return error;
};

// Global error handler middleware
export const errorHandler = (
  err: AppError,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  // Set default error properties
  err.statusCode = err.statusCode || 500;
  err.status = err.status || 'error';

  // Log error
  const errorLog = {
    message: err.message,
    stack: err.stack,
    statusCode: err.statusCode,
    status: err.status,
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    body: req.body,
    query: req.query,
    params: req.params,
    headers: req.headers,
    timestamp: new Date().toISOString(),
    requestId: req.headers['x-request-id'] || 'unknown'
  };

  if (err.statusCode >= 500) {
    logger.error('Server Error', errorLog);
  } else {
    logger.warn('Client Error', errorLog);
  }

  // Determine response based on environment and error type
  const response = createErrorResponse(err, req);

  res.status(err.statusCode).json(response);
};

const createErrorResponse = (err: AppError, req: Request) => {
  const isDevelopment = process.env.NODE_ENV === 'development';
  const isProduction = process.env.NODE_ENV === 'production';

  const baseResponse = {
    status: err.status,
    timestamp: new Date().toISOString(),
    path: req.url,
    method: req.method,
    requestId: req.headers['x-request-id'] || undefined
  };

  // Production environment - minimal error info
  if (isProduction) {
    if (err.statusCode && err.statusCode >= 500) {
      return {
        ...baseResponse,
        error: 'Internal Server Error',
        message: 'An unexpected error occurred'
      };
    } else {
      return {
        ...baseResponse,
        error: err.message || 'Bad Request'
      };
    }
  }

  // Development/staging environment - detailed error info
  return {
    ...baseResponse,
    error: err.message,
    ...(isDevelopment && {
      stack: err.stack,
      details: {
        statusCode: err.statusCode,
        operational: err.operational
      }
    })
  };
};

// Async error wrapper for route handlers
export const asyncHandler = (
  fn: (req: Request, res: Response, next: NextFunction) => Promise<any>
) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

// 404 handler for unmatched routes
export const notFoundHandler = (req: Request, res: Response, next: NextFunction) => {
  const error = createError(
    `Route ${req.method} ${req.originalUrl} not found`,
    404
  );
  next(error);
};

// Validation error handler
export const validationErrorHandler = (errors: any[], req: Request) => {
  const message = errors.map(err => err.message || err).join(', ');
  return createError(`Validation Error: ${message}`, 400);
};

// Database error handler
export const databaseErrorHandler = (err: any) => {
  logger.error('Database Error', {
    message: err.message,
    code: err.code,
    detail: err.detail,
    stack: err.stack
  });

  // Handle specific database errors
  if (err.code === '23505') { // Unique violation
    return createError('Duplicate resource', 409);
  }

  if (err.code === '23503') { // Foreign key violation
    return createError('Referenced resource not found', 400);
  }

  if (err.code === '23502') { // Not null violation
    return createError('Missing required field', 400);
  }

  // Generic database error
  return createError('Database operation failed', 500, false);
};

// Rate limit error handler
export const rateLimitErrorHandler = (req: Request, res: Response) => {
  logger.warn('Rate limit exceeded', {
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    url: req.url,
    method: req.method
  });

  res.status(429).json({
    status: 'error',
    error: 'Too Many Requests',
    message: 'Rate limit exceeded. Please try again later.',
    timestamp: new Date().toISOString(),
    retryAfter: 60
  });
};

// Timeout error handler
export const timeoutErrorHandler = (req: Request, res: Response) => {
  logger.error('Request timeout', {
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent')
  });

  res.status(408).json({
    status: 'error',
    error: 'Request Timeout',
    message: 'Request took too long to process',
    timestamp: new Date().toISOString()
  });
};

// Payload too large error handler
export const payloadTooLargeErrorHandler = (req: Request, res: Response) => {
  logger.warn('Payload too large', {
    url: req.url,
    method: req.method,
    ip: req.ip,
    contentLength: req.get('Content-Length')
  });

  res.status(413).json({
    status: 'error',
    error: 'Payload Too Large',
    message: 'Request payload is too large',
    timestamp: new Date().toISOString()
  });
}; 