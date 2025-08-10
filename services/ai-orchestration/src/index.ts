/**
 * AI Orchestration Service - TypeScript Interface
 * Bridges TypeScript services with Python AI orchestration system
 */

import express, { Request, Response, NextFunction } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import { v4 as uuidv4 } from 'uuid';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';

// Import shared utilities
import { Config, logger, database, redis } from '@userwhisperer/shared';

// Types
interface OrchestrationRequest extends Request {
  orchestrationId?: string;
}

interface AIOrchestrationResult {
  orchestration_id: string;
  user_id: string;
  strategy_decisions: {
    intervention_type: string;
    channel: string;
    timing: string;
    content_strategy: any;
  };
  ai_insights: {
    confidence: number;
    reasoning: string[];
    learned_patterns: string[];
  };
  next_actions: string[];
  metadata: {
    processing_time_ms: number;
    models_used: string[];
    adaptation_applied: boolean;
  };
}

class AIOrchestrationApp {
  private app: express.Application;
  private port: number;
  private pythonProcess: ChildProcess | null = null;
  private pythonOrchestrationPath: string;
  
  constructor() {
    this.app = express();
    this.port = Config.AI_ORCHESTRATION_PORT || 8085;
    this.pythonOrchestrationPath = path.join(__dirname, '../../../shared/ai_orchestration');
    
    this.setupMiddleware();
    this.setupRoutes();
    this.setupErrorHandling();
  }
  
  private setupMiddleware(): void {
    // Security middleware
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "https:"],
        },
      },
    }));
    
    this.app.use(cors({
      origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
      credentials: true
    }));
    
    this.app.use(compression());
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));
    
    // Rate limiting
    const limiter = rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 1000, // limit each IP to 1000 requests per windowMs
      message: 'Too many requests from this IP, please try again later.',
      standardHeaders: true,
      legacyHeaders: false,
    });
    this.app.use(limiter);
    
    // Request tracking
    this.app.use((req: OrchestrationRequest, res: Response, next: NextFunction) => {
      req.orchestrationId = uuidv4();
      res.setHeader('X-Orchestration-ID', req.orchestrationId);
      
      logger.info('AI Orchestration request received', {
        orchestrationId: req.orchestrationId,
        method: req.method,
        url: req.url,
        userAgent: req.get('User-Agent'),
        ip: req.ip
      });
      
      next();
    });
  }
  
  private setupRoutes(): void {
    // Health check
    this.app.get('/health', async (req: Request, res: Response) => {
      try {
        const healthStatus = {
          service: 'ai-orchestration',
          status: 'healthy',
          timestamp: new Date().toISOString(),
          version: process.env.npm_package_version || '1.0.0',
          uptime: process.uptime(),
          checks: {
            database: await this.checkDatabaseHealth(),
            redis: await this.checkRedisHealth(),
            python_ai: await this.checkPythonAIHealth()
          }
        };
        
        const isHealthy = Object.values(healthStatus.checks).every(check => check === true);
        res.status(isHealthy ? 200 : 503).json(healthStatus);
      } catch (error) {
        logger.error('Health check failed', { error: error instanceof Error ? error.message : String(error) });
        res.status(503).json({
          service: 'ai-orchestration',
          status: 'unhealthy',
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    });
    
    // Orchestrate AI decision for a user
    this.app.post('/v1/orchestrate/user/:userId', async (req: OrchestrationRequest, res: Response) => {
      try {
        const { userId } = req.params;
        const {
          user_context,
          behavioral_data,
          engagement_history,
          business_objectives,
          constraints
        } = req.body;
        
        logger.info('Starting AI orchestration for user', {
          orchestrationId: req.orchestrationId,
          userId,
          hasUserContext: !!user_context,
          hasBehavioralData: !!behavioral_data
        });
        
        const orchestrationResult = await this.executeAIOrchestration({
          user_id: userId,
          user_context: user_context || {},
          behavioral_data: behavioral_data || {},
          engagement_history: engagement_history || [],
          business_objectives: business_objectives || {},
          constraints: constraints || {}
        });
        
        return res.json({
          orchestration_id: req.orchestrationId,
          status: 'success',
          result: orchestrationResult,
          timestamp: new Date().toISOString()
        });
        
      } catch (error) {
        logger.error('AI orchestration failed', {
          orchestrationId: req.orchestrationId,
          userId: req.params.userId,
          error: error instanceof Error ? error.message : String(error)
        });
        
        return res.status(500).json({
          orchestration_id: req.orchestrationId,
          status: 'error',
          error: 'AI orchestration failed',
          message: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    });
    
    // Batch orchestration for multiple users
    this.app.post('/v1/orchestrate/batch', async (req: OrchestrationRequest, res: Response) => {
      try {
        const { users, global_context } = req.body;
        
        if (!Array.isArray(users) || users.length === 0) {
          return res.status(400).json({
            orchestration_id: req.orchestrationId,
            status: 'error',
            error: 'Invalid request: users array is required'
          });
        }
        
        logger.info('Starting batch AI orchestration', {
          orchestrationId: req.orchestrationId,
          userCount: users.length
        });
        
        const results = await Promise.allSettled(
          users.map(async (user) => {
            const result = await this.executeAIOrchestration({
              user_id: user.user_id,
              user_context: user.user_context || {},
              behavioral_data: user.behavioral_data || {},
              engagement_history: user.engagement_history || [],
              business_objectives: global_context?.business_objectives || {},
              constraints: global_context?.constraints || {}
            });
            return {
              user_id: user.user_id,
              result
            };
          })
        );
        
        const successful = results.filter(r => r.status === 'fulfilled').length;
        const failed = results.filter(r => r.status === 'rejected').length;
        
        return res.json({
          orchestration_id: req.orchestrationId,
          status: 'completed',
          summary: {
            total: users.length,
            successful,
            failed
          },
          results: results.map(r => 
            r.status === 'fulfilled' ? r.value : { error: (r.reason as Error).message }
          ),
          timestamp: new Date().toISOString()
        });
        
      } catch (error) {
        logger.error('Batch AI orchestration failed', {
          orchestrationId: req.orchestrationId,
          error: error instanceof Error ? error.message : String(error)
        });
        
        return res.status(500).json({
          orchestration_id: req.orchestrationId,
          status: 'error',
          error: 'Batch orchestration failed',
          message: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    });
    
    // Get AI system insights and analytics
    this.app.get('/v1/insights/system', async (req: Request, res: Response) => {
      try {
        const insights = await this.getSystemInsights();
        return res.json({
          status: 'success',
          insights,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        logger.error('Failed to get system insights', { error: error instanceof Error ? error.message : String(error) });
        return res.status(500).json({
          status: 'error',
          error: 'Failed to retrieve system insights'
        });
      }
    });
    
    // Update AI system configuration
    this.app.post('/v1/config/update', async (req: Request, res: Response) => {
      try {
        const { configuration } = req.body;
        await this.updateAIConfiguration(configuration);
        return res.json({
          status: 'success',
          message: 'AI configuration updated successfully',
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        logger.error('Failed to update AI configuration', { error: error instanceof Error ? error.message : String(error) });
        return res.status(500).json({
          status: 'error',
          error: 'Failed to update AI configuration'
        });
      }
    });
  }
  
  private async executeAIOrchestration(params: any): Promise<AIOrchestrationResult> {
    // This would interface with the Python AI orchestration system
    // For now, returning a mock response with the expected structure
    const startTime = Date.now();
    
    // Simulate AI processing
    await new Promise(resolve => setTimeout(resolve, 100));
    
    return {
      orchestration_id: uuidv4(),
      user_id: params.user_id,
      strategy_decisions: {
        intervention_type: this.determineInterventionType(params),
        channel: this.selectOptimalChannel(params),
        timing: this.optimizeTiming(params),
        content_strategy: this.generateContentStrategy(params)
      },
      ai_insights: {
        confidence: 0.85,
        reasoning: [
          'User engagement pattern indicates high receptivity to email',
          'Behavioral data suggests optimal timing in evening hours',
          'Previous interactions show preference for personalized content'
        ],
        learned_patterns: [
          'User responds 40% better to feature-focused messaging',
          'Email engagement peaks at 7-9 PM in user timezone'
        ]
      },
      next_actions: [
        'Schedule content generation',
        'Prepare personalization data',
        'Set up delivery tracking'
      ],
      metadata: {
        processing_time_ms: Date.now() - startTime,
        models_used: ['churn_prediction', 'content_optimization', 'timing_optimization'],
        adaptation_applied: true
      }
    };
  }
  
  private determineInterventionType(params: any): string {
    // Mock intervention type logic
    const churnRisk = params.behavioral_data?.churn_risk || 0;
    if (churnRisk > 0.7) return 'retention';
    if (params.user_context?.subscription_plan === 'free') return 'monetization';
    return 'engagement';
  }
  
  private selectOptimalChannel(params: any): string {
    // Mock channel selection logic
    const preferences = params.user_context?.communication_preferences;
    if (preferences?.preferred_channel) return preferences.preferred_channel;
    return 'email'; // Default
  }
  
  private optimizeTiming(params: any): string {
    // Mock timing optimization
    const timezone = params.user_context?.timezone || 'UTC';
    return `optimal_evening_${timezone}`;
  }
  
  private generateContentStrategy(params: any): any {
    // Mock content strategy
    return {
      personalization_level: 'high',
      tone: 'friendly',
      focus_areas: ['product_value', 'user_success'],
      cta_strategy: 'soft_nudge'
    };
  }
  
  private async getSystemInsights(): Promise<any> {
    // Mock system insights
    return {
      performance_metrics: {
        total_orchestrations: 15420,
        success_rate: 0.94,
        avg_processing_time_ms: 245,
        model_accuracy: {
          churn_prediction: 0.89,
          content_optimization: 0.92,
          timing_optimization: 0.87
        }
      },
      learning_insights: {
        patterns_discovered: 127,
        adaptations_applied: 89,
        improvement_rate: 0.15
      },
      system_health: {
        ml_models_status: 'healthy',
        data_pipeline_status: 'healthy',
        last_training_update: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
      }
    };
  }
  
  private async updateAIConfiguration(config: any): Promise<void> {
    // Mock configuration update
    logger.info('AI configuration updated', { config });
  }
  
  private async checkDatabaseHealth(): Promise<boolean> {
    try {
      return await database.healthCheck();
    } catch (error) {
      return false;
    }
  }
  
  private async checkRedisHealth(): Promise<boolean> {
    try {
      return await redis.healthCheck();
    } catch (error) {
      return false;
    }
  }
  
  private async checkPythonAIHealth(): Promise<boolean> {
    try {
      // Check if Python AI orchestration system is accessible
      // This would ping the Python system
      return true; // Mock for now
    } catch (error) {
      return false;
    }
  }
  
  private setupErrorHandling(): void {
    // 404 handler
    this.app.use('*', (req: Request, res: Response) => {
      res.status(404).json({
        status: 'error',
        error: 'Endpoint not found',
        path: req.originalUrl,
        method: req.method
      });
    });
    
    // Global error handler
    this.app.use((error: Error, req: OrchestrationRequest, res: Response, next: NextFunction) => {
      logger.error('Unhandled error in AI orchestration service', {
        error: error.message,
        stack: error.stack,
        orchestrationId: req.orchestrationId,
        url: req.url,
        method: req.method
      });
      
      res.status(500).json({
        orchestration_id: req.orchestrationId,
        status: 'error',
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
      });
    });
  }
  
  public async start(): Promise<void> {
    try {
      // Initialize connections
      await this.initializeConnections();
      
      // Start server
      const server = this.app.listen(this.port, () => {
        logger.info(`AI Orchestration Service started successfully`, {
          port: this.port,
          environment: process.env.NODE_ENV || 'development',
          nodeVersion: process.version
        });
      });
      
      // Graceful shutdown
      process.on('SIGTERM', async () => {
        logger.info('SIGTERM received, starting graceful shutdown');
        server.close(async () => {
          try {
            await this.cleanup();
            logger.info('AI Orchestration Service shut down gracefully');
            process.exit(0);
          } catch (error) {
            logger.error('Error during shutdown', { error: error instanceof Error ? error.message : String(error) });
            process.exit(1);
          }
        });
      });
      
    } catch (error) {
      logger.error('Failed to start AI Orchestration Service', { error: error instanceof Error ? error.message : String(error) });
      process.exit(1);
    }
  }
  
  private async initializeConnections(): Promise<void> {
    // Test database connection
    const dbHealthy = await this.checkDatabaseHealth();
    if (!dbHealthy) {
      throw new Error('Database connection failed');
    }
    
    // Test Redis connection
    const redisHealthy = await this.checkRedisHealth();
    if (!redisHealthy) {
      throw new Error('Redis connection failed');
    }
    
    logger.info('All connections initialized successfully');
  }
  
  private async cleanup(): Promise<void> {
    // Cleanup Python process if running
    if (this.pythonProcess) {
      this.pythonProcess.kill();
    }
  }
}

// Start the service
if (require.main === module) {
  const app = new AIOrchestrationApp();
  app.start().catch((error) => {
    logger.error('Failed to start AI Orchestration Service', { error });
    process.exit(1);
  });
}

export default AIOrchestrationApp;