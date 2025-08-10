import { Request, Response, NextFunction } from 'express';
import promClient from 'prom-client';

// Create metrics registry
const register = new promClient.Registry();

// Add default metrics
promClient.collectDefaultMetrics({
  register,
  prefix: 'event_ingestion_'
});

// Custom metrics
const httpRequestDuration = new promClient.Histogram({
  name: 'event_ingestion_http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.1, 0.5, 1, 2, 5, 10]
});

const httpRequestsTotal = new promClient.Counter({
  name: 'event_ingestion_http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code']
});

const eventsProcessedTotal = new promClient.Counter({
  name: 'event_ingestion_events_processed_total',
  help: 'Total number of events processed',
  labelNames: ['status', 'event_type']
});

const eventsProcessingDuration = new promClient.Histogram({
  name: 'event_ingestion_events_processing_duration_seconds',
  help: 'Duration of event processing in seconds',
  labelNames: ['event_type', 'status'],
  buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2]
});

const activeConnections = new promClient.Gauge({
  name: 'event_ingestion_active_connections',
  help: 'Number of active connections'
});

const queueSize = new promClient.Gauge({
  name: 'event_ingestion_queue_size',
  help: 'Current size of the processing queue'
});

// Register custom metrics
register.registerMetric(httpRequestDuration);
register.registerMetric(httpRequestsTotal);
register.registerMetric(eventsProcessedTotal);
register.registerMetric(eventsProcessingDuration);
register.registerMetric(activeConnections);
register.registerMetric(queueSize);

// Metrics middleware
export const metricsMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const startTime = Date.now();
  
  // Track active connections
  activeConnections.inc();
  
  res.on('finish', () => {
    const duration = (Date.now() - startTime) / 1000;
    const route = req.route?.path || req.path;
    const method = req.method;
    const statusCode = res.statusCode.toString();
    
    // Record metrics
    httpRequestDuration
      .labels(method, route, statusCode)
      .observe(duration);
    
    httpRequestsTotal
      .labels(method, route, statusCode)
      .inc();
    
    // Decrease active connections
    activeConnections.dec();
  });
  
  next();
};

// Export metrics for /metrics endpoint
export const getMetrics = () => register.metrics();

// Export individual metrics for use in other parts of the application
export const metrics = {
  httpRequestDuration,
  httpRequestsTotal,
  eventsProcessedTotal,
  eventsProcessingDuration,
  activeConnections,
  queueSize,
  register
};

// Helper functions to update metrics from other parts of the application
export const updateEventMetrics = (
  eventType: string,
  status: 'accepted' | 'rejected' | 'duplicate',
  processingTimeMs: number
) => {
  eventsProcessedTotal.labels(status, eventType).inc();
  eventsProcessingDuration.labels(eventType, status).observe(processingTimeMs / 1000);
};

export const updateQueueSize = (size: number) => {
  queueSize.set(size);
}; 