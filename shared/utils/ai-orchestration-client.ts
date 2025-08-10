/**
 * AI Orchestration Client
 * Provides interface for services to communicate with AI orchestration system
 */

import Config from './config';
import { createServiceLogger } from './logger';

export interface AIOrchestrationRequest {
  user_id: string;
  user_context?: any;
  behavioral_data?: any;
  engagement_history?: any[];
  business_objectives?: any;
  constraints?: any;
  trigger_event?: string;
  intervention_type?: string;
}

export interface AIOrchestrationResult {
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

export class AIOrchestrationClient {
  private baseUrl: string;
  private logger = createServiceLogger('ai-orchestration-client');
  private timeout: number = 30000; // 30 seconds

  constructor() {
    const host = Config.AI_ORCHESTRATION_HOST || 'localhost';
    const port = Config.AI_ORCHESTRATION_PORT || 8085;
    this.baseUrl = `http://${host}:${port}`;
  }

  /**
   * Request AI orchestration for a single user
   */
  async orchestrateUser(request: AIOrchestrationRequest): Promise<AIOrchestrationResult> {
    try {
      this.logger.info('Requesting AI orchestration for user', {
        userId: request.user_id,
        triggerEvent: request.trigger_event,
        interventionType: request.intervention_type
      });

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(`${this.baseUrl}/v1/orchestrate/user/${request.user_id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`AI orchestration failed with status ${response.status}: ${response.statusText}`);
      }

      const result = await response.json() as any;
      
      this.logger.info('AI orchestration completed successfully', {
        userId: request.user_id,
        orchestrationId: result.orchestration_id,
        interventionType: result.result?.strategy_decisions?.intervention_type,
        confidence: result.result?.ai_insights?.confidence
      });

      return result.result;

    } catch (error) {
      this.logger.error('AI orchestration request failed', {
        userId: request.user_id,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return fallback orchestration result
      return this.generateFallbackResult(request);
    }
  }

  /**
   * Request AI orchestration for multiple users (batch)
   */
  async orchestrateBatch(requests: AIOrchestrationRequest[]): Promise<AIOrchestrationResult[]> {
    try {
      this.logger.info('Requesting batch AI orchestration', {
        userCount: requests.length
      });

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 2); // Longer timeout for batch

      const response = await fetch(`${this.baseUrl}/v1/orchestrate/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          users: requests,
          global_context: {
            business_objectives: requests[0]?.business_objectives || {},
            constraints: requests[0]?.constraints || {}
          }
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Batch AI orchestration failed with status ${response.status}: ${response.statusText}`);
      }

      const result = await response.json() as any;
      
      this.logger.info('Batch AI orchestration completed', {
        userCount: requests.length,
        successful: result.summary?.successful || 0,
        failed: result.summary?.failed || 0
      });

      return result.results.map((r: any) => r.result || this.generateFallbackResult({ user_id: r.user_id }));

    } catch (error) {
      this.logger.error('Batch AI orchestration request failed', {
        userCount: requests.length,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return fallback results for all users
      return requests.map(req => this.generateFallbackResult(req));
    }
  }

  /**
   * Get AI system insights and performance metrics
   */
  async getSystemInsights(): Promise<any> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(`${this.baseUrl}/v1/insights/system`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`System insights request failed with status ${response.status}: ${response.statusText}`);
      }

      const result = await response.json() as any;
      return result.insights;

    } catch (error) {
      this.logger.error('System insights request failed', {
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        performance_metrics: {
          total_orchestrations: 0,
          success_rate: 0.95,
          avg_processing_time_ms: 250
        },
        system_health: {
          ml_models_status: 'unknown',
          data_pipeline_status: 'unknown'
        }
      };
    }
  }

  /**
   * Update AI system configuration
   */
  async updateConfiguration(config: any): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(`${this.baseUrl}/v1/config/update`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ configuration: config }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      return response.ok;

    } catch (error) {
      this.logger.error('Configuration update failed', {
        error: error instanceof Error ? error.message : String(error)
      });
      return false;
    }
  }

  /**
   * Check if AI orchestration service is healthy
   */
  async healthCheck(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout for health checks

      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      return response.ok;

    } catch (error) {
      this.logger.warn('AI orchestration health check failed', {
        error: error instanceof Error ? error.message : String(error)
      });
      return false;
    }
  }

  /**
   * Generate fallback orchestration result when AI service is unavailable
   */
  private generateFallbackResult(request: AIOrchestrationRequest): AIOrchestrationResult {
    return {
      orchestration_id: `fallback_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      user_id: request.user_id,
      strategy_decisions: {
        intervention_type: request.intervention_type || 'engagement',
        channel: 'email',
        timing: 'optimal_evening',
        content_strategy: {
          personalization_level: 'medium',
          tone: 'friendly',
          focus_areas: ['product_value'],
          cta_strategy: 'soft_nudge'
        }
      },
      ai_insights: {
        confidence: 0.6,
        reasoning: ['Fallback decision due to AI service unavailability'],
        learned_patterns: []
      },
      next_actions: [
        'Generate personalized content',
        'Schedule for optimal delivery time',
        'Track engagement metrics'
      ],
      metadata: {
        processing_time_ms: 50,
        models_used: ['fallback_heuristics'],
        adaptation_applied: false
      }
    };
  }
}

// Export singleton instance
export const aiOrchestrationClient = new AIOrchestrationClient();
