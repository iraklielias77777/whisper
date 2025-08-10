import winston from 'winston';

// Define log levels
const levels = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3,
};

// Define colors for each level
const colors = {
  error: 'red',
  warn: 'yellow',
  info: 'green',
  debug: 'blue',
};

winston.addColors(colors);

// Create format for development (pretty)
const developmentFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.errors({ stack: true }),
  winston.format.colorize({ all: true }),
  winston.format.printf(({ timestamp, level, message, service, ...meta }) => {
    const metaStr = Object.keys(meta).length ? JSON.stringify(meta, null, 2) : '';
    const serviceStr = service ? `[${service}]` : '';
    return `${timestamp} ${level} ${serviceStr}: ${message} ${metaStr}`;
  })
);

// Create format for production (JSON)
const productionFormat = winston.format.combine(
  winston.format.timestamp(),
  winston.format.errors({ stack: true }),
  winston.format.json()
);

// Determine format based on environment
const logFormat = process.env.NODE_ENV === 'production' ? productionFormat : developmentFormat;

// Create the logger instance
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  levels,
  format: logFormat,
  defaultMeta: {
    service: process.env.SERVICE_NAME || 'user-whisperer',
    version: process.env.npm_package_version || '1.0.0',
    environment: process.env.NODE_ENV || 'development',
  },
  transports: [
    // Console transport for all environments
    new winston.transports.Console({
      level: process.env.LOG_LEVEL || 'info',
      handleExceptions: true,
      handleRejections: true,
    }),
  ],
  exitOnError: false,
});

// Add file transports for production
if (process.env.NODE_ENV === 'production') {
  logger.add(
    new winston.transports.File({
      filename: 'logs/error.log',
      level: 'error',
      maxsize: 5242880, // 5MB
      maxFiles: 10,
    })
  );

  logger.add(
    new winston.transports.File({
      filename: 'logs/combined.log',
      maxsize: 5242880, // 5MB
      maxFiles: 10,
    })
  );
}

// Create a stream for Morgan HTTP logging
export const logStream = {
  write: (message: string) => {
    logger.info(message.trim());
  },
};

// Custom logging methods for specific use cases
export const requestLogger = (req: any, res: any, next: any) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    const logData = {
      method: req.method,
      url: req.url,
      status: res.statusCode,
      duration: `${duration}ms`,
      userAgent: req.get('User-Agent'),
      ip: req.ip,
      userId: req.user?.id,
    };
    
    if (res.statusCode >= 400) {
      logger.warn('HTTP Request', logData);
    } else {
      logger.info('HTTP Request', logData);
    }
  });
  
  next();
};

export const performanceLogger = {
  time: (label: string) => logger.profile(label),
  timeEnd: (label: string) => logger.profile(label),
  
  measure: async <T>(label: string, fn: () => Promise<T> | T): Promise<T> => {
    const start = Date.now();
    try {
      const result = await fn();
      const duration = Date.now() - start;
      logger.debug(`Performance: ${label}`, { duration: `${duration}ms` });
      return result;
    } catch (error) {
      const duration = Date.now() - start;
      logger.error(`Performance: ${label} (failed)`, { 
        duration: `${duration}ms`,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  },
};

export const errorLogger = {
  logError: (error: Error, context?: any) => {
    logger.error('Application Error', {
      message: error.message,
      stack: error.stack,
      name: error.name,
      context,
    });
  },
  
  logUnhandledRejection: (reason: any, promise: Promise<any>) => {
    logger.error('Unhandled Promise Rejection', {
      reason: reason instanceof Error ? reason.message : String(reason),
      stack: reason instanceof Error ? reason.stack : undefined,
      promise: promise.toString(),
    });
  },
  
  logUncaughtException: (error: Error) => {
    logger.error('Uncaught Exception', {
      message: error.message,
      stack: error.stack,
      name: error.name,
    });
  },
};

// Set up global error handlers
process.on('unhandledRejection', errorLogger.logUnhandledRejection);
process.on('uncaughtException', errorLogger.logUncaughtException);

// Create child logger for specific services
export const createServiceLogger = (serviceName: string) => {
  return logger.child({ service: serviceName });
};

// Structured logging helpers
export const audit = {
  userAction: (userId: string, action: string, details?: any) => {
    logger.info('User Action', {
      type: 'audit',
      userId,
      action,
      details,
      timestamp: new Date().toISOString(),
    });
  },
  
  systemEvent: (event: string, details?: any) => {
    logger.info('System Event', {
      type: 'audit',
      event,
      details,
      timestamp: new Date().toISOString(),
    });
  },
  
  securityEvent: (event: string, details?: any) => {
    logger.warn('Security Event', {
      type: 'security',
      event,
      details,
      timestamp: new Date().toISOString(),
    });
  },
};

// Metrics logging
export const metrics = {
  counter: (name: string, value: number = 1, tags?: any) => {
    logger.info('Metric Counter', {
      type: 'metric',
      metric: 'counter',
      name,
      value,
      tags,
    });
  },
  
  gauge: (name: string, value: number, tags?: any) => {
    logger.info('Metric Gauge', {
      type: 'metric',
      metric: 'gauge',
      name,
      value,
      tags,
    });
  },
  
  timing: (name: string, duration: number, tags?: any) => {
    logger.info('Metric Timing', {
      type: 'metric',
      metric: 'timing',
      name,
      duration,
      tags,
    });
  },
};

// Export the main logger as default
export default logger; 