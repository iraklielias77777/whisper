import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';

import { Config, logger } from '@userwhisperer/shared';
import { ContentGenerator } from './services/ContentGenerator';
import { 
  ContentRequest, 
  ContentGenConfig,
  GenerationRequest,
  GenerationResponse 
} from './types';
import { v4 as uuidv4 } from 'uuid';

// Initialize Express app
const app = express();
const port = Config.CONTENT_GENERATION_PORT;

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.NODE_ENV === 'production' ? false : true,
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // Higher limit for content generation
  message: 'Too many content generation requests'
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

// Initialize Content Generator
const contentConfig: ContentGenConfig = {
  llm_providers: {
    openai: process.env.OPENAI_API_KEY ? {
      api_key: process.env.OPENAI_API_KEY,
      model: 'gpt-4-turbo-preview',
      max_tokens: 2000,
      temperature: 0.7
    } : undefined,
    anthropic: process.env.ANTHROPIC_API_KEY ? {
      api_key: process.env.ANTHROPIC_API_KEY,
      model: 'claude-3-sonnet-20240229',
      max_tokens: 2000,
      temperature: 0.7
    } : undefined
  },
  template_engine: 'nunjucks',
  cache_ttl: 3600, // 1 hour
  max_variations: 3,
  quality_threshold: 0.7,
  personalization_threshold: 0.5,
  fallback_strategy: 'template',
  content_moderation: true,
  a_b_testing_enabled: true
};

const contentGenerator = new ContentGenerator(contentConfig);

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const healthCheck = await contentGenerator.healthCheck();
    
    res.status(healthCheck.healthy ? 200 : 503).json({
      service: 'content-generation',
      status: healthCheck.healthy ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      details: healthCheck.details
    });
  } catch (error) {
    logger.error('Health check failed', { error });
    res.status(503).json({
      service: 'content-generation',
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: 'Health check failed'
    });
  }
});

// Generate content endpoint
app.post('/v1/content/generate', async (req, res) => {
  const startTime = Date.now();
  const requestId = uuidv4();
  
  try {
    const contentRequest: ContentRequest = req.body;

    if (!contentRequest.user_id) {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'user_id is required'
      });
    }

    if (!contentRequest.intervention_type) {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'intervention_type is required'
      });
    }

    if (!contentRequest.channel) {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'channel is required'
      });
    }

    logger.info('Content generation request received', {
      requestId,
      userId: contentRequest.user_id,
      interventionType: contentRequest.intervention_type,
      channel: contentRequest.channel,
      personalizationLevel: contentRequest.personalization_level
    });

    const generatedContent = await contentGenerator.generateContent(contentRequest);
    const processingTime = Date.now() - startTime;

    const response: GenerationResponse = {
      request_id: requestId,
      status: 'success',
      content: generatedContent,
      processing_time_ms: processingTime,
      generation_method: generatedContent.metadata.generation_method
    };

    return res.json({
      ...response,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    const processingTime = Date.now() - startTime;
    
    logger.error('Content generation failed', {
      requestId,
      userId: req.body?.user_id,
      error: error instanceof Error ? error.message : String(error),
      processingTime: `${processingTime}ms`
    });

    const response: GenerationResponse = {
      request_id: requestId,
      status: 'failed',
      processing_time_ms: processingTime,
      generation_method: 'fallback',
      error: error instanceof Error ? error.message : 'Unknown error'
    };

    return res.status(500).json(response);
  }
});

// Batch content generation endpoint
app.post('/v1/content/generate/batch', async (req, res) => {
  const startTime = Date.now();
  const batchId = uuidv4();
  
  try {
    const { requests } = req.body;

    if (!Array.isArray(requests) || requests.length === 0) {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'requests array is required and cannot be empty'
      });
    }

    if (requests.length > 50) {
      return res.status(400).json({
        error: 'Batch too large',
        message: 'Maximum 50 requests per batch'
      });
    }

    logger.info('Batch content generation started', {
      batchId,
      requestCount: requests.length
    });

    const results = await Promise.allSettled(
      requests.map(async (request: ContentRequest, index: number) => {
        try {
          const content = await contentGenerator.generateContent(request);
          return {
            request_index: index,
            user_id: request.user_id,
            status: 'success',
            content,
            generation_method: content.metadata.generation_method
          };
        } catch (error) {
          return {
            request_index: index,
            user_id: request.user_id,
            status: 'error',
            error: error instanceof Error ? error.message : String(error)
          };
        }
      })
    );

    const responses = results.map(result => 
      result.status === 'fulfilled' ? result.value : {
        ...result.reason,
        status: 'error'
      }
    );

    const processingTime = Date.now() - startTime;
    const successCount = responses.filter(r => r.status === 'success').length;

    return res.json({
      batch_id: batchId,
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
    
    logger.error('Batch content generation failed', {
      batchId,
      error: error instanceof Error ? error.message : String(error),
      processingTime: `${processingTime}ms`
    });

    return res.status(500).json({
      batch_id: batchId,
      status: 'failed',
      error: error instanceof Error ? error.message : 'Unknown error',
      processing_time_ms: processingTime
    });
  }
});

// Preview content endpoint (for testing templates and personalization)
app.post('/v1/content/preview', async (req, res) => {
  try {
    const { template_id, user_context, personalization_data } = req.body;

    if (!template_id) {
      return res.status(400).json({
        error: 'Invalid request',
        message: 'template_id is required'
      });
    }

    // This would generate a preview using the template and sample data
    // For now, return a mock preview
    return res.json({
      preview_id: uuidv4(),
      template_id,
      preview_content: {
        subject: "Preview: Your personalized message",
        body: `Hi ${user_context?.name || 'User'}! This is a preview of your personalized content.`,
        cta_text: "Preview CTA",
        cta_link: "{{cta_link}}"
      },
      personalization_elements: Object.keys(personalization_data || {}),
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Content preview failed', {
      error: error instanceof Error ? error.message : String(error)
    });

    return res.status(500).json({
      error: 'Preview generation failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Template management endpoints
app.get('/v1/templates', async (req, res) => {
  try {
    const { intervention_type, channel } = req.query;

    // This would fetch templates from database
    // For now, return mock templates
    const mockTemplates = [
      {
        id: 'retention_email_v1',
        name: 'Retention Email - Value Reminder',
        intervention_type: 'retention',
        channel: 'email',
        success_rate: 0.32,
        usage_count: 1250,
        created_at: '2024-01-15T10:00:00Z'
      },
      {
        id: 'monetization_push_v1',
        name: 'Monetization Push - Upgrade Prompt',
        intervention_type: 'monetization',
        channel: 'push',
        success_rate: 0.28,
        usage_count: 890,
        created_at: '2024-01-20T14:30:00Z'
      }
    ];

    let filteredTemplates = mockTemplates;

    if (intervention_type) {
      filteredTemplates = filteredTemplates.filter(t => t.intervention_type === intervention_type);
    }

    if (channel) {
      filteredTemplates = filteredTemplates.filter(t => t.channel === channel);
    }

    return res.json({
      templates: filteredTemplates,
      total: filteredTemplates.length,
      filters: { intervention_type, channel },
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Template listing failed', {
      error: error instanceof Error ? error.message : String(error)
    });

    return res.status(500).json({
      error: 'Failed to list templates',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Content analytics endpoint
app.get('/v1/content/:contentId/analytics', async (req, res) => {
  try {
    const { contentId } = req.params;

    // This would fetch analytics from database
    // For now, return mock analytics
    const mockAnalytics = {
      content_id: contentId,
      delivery_stats: {
        sent_count: 1000,
        delivered_count: 980,
        bounced_count: 20,
        delivery_rate: 0.98
      },
      engagement_stats: {
        opened_count: 350,
        clicked_count: 85,
        shared_count: 12,
        open_rate: 0.357,
        click_rate: 0.087,
        engagement_rate: 0.447
      },
      conversion_stats: {
        conversion_count: 25,
        conversion_rate: 0.025,
        revenue_generated: 1250.00
      },
      quality_metrics: {
        readability_score: 0.82,
        sentiment_score: 0.78,
        personalization_score: 0.85,
        relevance_score: 0.79,
        spam_score: 0.05
      }
    };

    return res.json({
      ...mockAnalytics,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Analytics retrieval failed', {
      contentId: req.params.contentId,
      error: error instanceof Error ? error.message : String(error)
    });

    return res.status(500).json({
      error: 'Failed to retrieve analytics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Configuration endpoint
app.get('/v1/config', (req, res) => {
  try {
    return res.json({
      service: 'content-generation',
      version: '1.0.0',
      config: {
        max_variations: contentConfig.max_variations,
        quality_threshold: contentConfig.quality_threshold,
        cache_ttl: contentConfig.cache_ttl,
        template_engine: contentConfig.template_engine,
        llm_providers: {
          openai_enabled: !!contentConfig.llm_providers.openai,
          anthropic_enabled: !!contentConfig.llm_providers.anthropic
        }
      },
      supported_channels: ['email', 'sms', 'push', 'in_app'],
      supported_interventions: ['retention', 'monetization', 'onboarding', 'support', 'celebration'],
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Failed to get config', {
      error: error instanceof Error ? error.message : String(error)
    });

    return res.status(500).json({
      error: 'Failed to get configuration'
    });
  }
});

// Metrics endpoint for Prometheus
app.get('/metrics', (req, res) => {
  res.set('Content-Type', 'text/plain');
  res.send('# Content Generation Metrics\n# TODO: Implement Prometheus metrics\n');
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
  logger.info('Content Generation Service started', {
    port,
    environment: process.env.NODE_ENV || 'development',
    pid: process.pid,
    llm_providers: {
      openai: !!contentConfig.llm_providers.openai,
      anthropic: !!contentConfig.llm_providers.anthropic
    }
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