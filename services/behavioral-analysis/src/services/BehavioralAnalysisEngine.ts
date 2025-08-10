import { 
  UserEvent, 
  BehavioralMetrics, 
  AnalysisResult, 
  AnalysisConfig,
  UserLifecycleStage,
  UserProfile,
  EngagementScores
} from '../types';
import { logger, database, eventsCache } from '@userwhisperer/shared';
import { BehavioralMetricsCalculator } from './BehavioralMetricsCalculator';
import { UserStateMachine, StateTransitionResult } from './UserStateMachine';
import { PatternDetector, PatternDetectionConfig } from './PatternDetector';
import { ScoringModels, ScoringConfig } from './ScoringModels';

export class BehavioralAnalysisEngine {
  private metricsCalculator: BehavioralMetricsCalculator;
  private stateMachine: UserStateMachine;
  private patternDetector: PatternDetector;
  private scoringModels: ScoringModels;
  private config: AnalysisConfig;

  constructor(
    analysisConfig: AnalysisConfig,
    patternConfig?: Partial<PatternDetectionConfig>,
    scoringConfig?: ScoringConfig
  ) {
    this.config = analysisConfig;
    this.metricsCalculator = new BehavioralMetricsCalculator();
    this.stateMachine = new UserStateMachine();
    this.patternDetector = new PatternDetector(patternConfig);
    
    // Default scoring config if not provided
    const defaultScoringConfig: ScoringConfig = {
      churnModel: {
        enabled: false,
        version: 'rule-based-1.0',
        features: ['engagement_score', 'days_since_last_active', 'support_tickets']
      },
      ltvModel: {
        enabled: false,
        version: 'rule-based-1.0',
        baseValue: 500,
        timeHorizonDays: 365
      },
      upgradeModel: {
        enabled: false,
        version: 'rule-based-1.0',
        features: ['monetization_score', 'feature_adoption_rate', 'engagement_score']
      }
    };

    this.scoringModels = new ScoringModels(scoringConfig || defaultScoringConfig);
  }

  /**
   * Analyze user behavior and generate insights
   */
  public async analyzeBehavior(
    userId: string,
    events: UserEvent[],
    messageHistory?: any[]
  ): Promise<AnalysisResult> {
    const startTime = Date.now();

    try {
      logger.info('Starting behavioral analysis', {
        userId,
        eventsCount: events.length,
        windowDays: this.config.analysis_window_days
      });

      // Filter events within analysis window
      const filteredEvents = this.filterEventsByWindow(events);

      // Check minimum events requirement
      if (filteredEvents.length < this.config.min_events_required) {
        logger.warn('Insufficient events for analysis', {
          userId,
          eventsCount: filteredEvents.length,
          required: this.config.min_events_required
        });
        
        return this.createMinimalAnalysisResult(userId, filteredEvents);
      }

      // Calculate behavioral metrics
      const metrics = await this.metricsCalculator.calculateAllMetrics(
        userId,
        filteredEvents,
        messageHistory
      );

      // Get current user profile for state transition
      const userProfile = await this.getUserProfile(userId);
      const currentLifecycleStage = userProfile?.lifecycle_stage || UserLifecycleStage.NEW;

      // Determine state transitions
      const stateTransition = this.stateMachine.determineStateTransition(
        currentLifecycleStage,
        metrics,
        filteredEvents.slice(-50) // Last 50 events for transition analysis
      );

      // Detect behavioral patterns
      let patterns: any[] = [];
      if (this.config.pattern_detection_enabled) {
        patterns = this.patternDetector.detectPatterns(filteredEvents, metrics);
      }

      // Calculate engagement scores
      const engagementScores = this.calculateEngagementScores(metrics);

      // Generate predictions using ML models
      let churnPrediction, ltvPrediction, upgradePrediction;

      if (this.config.ml_models_enabled) {
        [churnPrediction, ltvPrediction, upgradePrediction] = await Promise.all([
          this.scoringModels.calculateChurnRisk(metrics),
          this.scoringModels.calculateLTVPrediction(metrics),
          this.scoringModels.calculateUpgradePrediction(metrics)
        ]);
      } else {
        // Use simplified predictions
        churnPrediction = await this.scoringModels.calculateChurnRisk(metrics);
        ltvPrediction = await this.scoringModels.calculateLTVPrediction(metrics);
        upgradePrediction = await this.scoringModels.calculateUpgradePrediction(metrics);
      }

      // Build complete analysis result
      const analysisResult: AnalysisResult = {
        user_id: userId,
        metrics,
        patterns,
        engagement_scores: engagementScores,
        churn_prediction: churnPrediction,
        ltv_prediction: ltvPrediction,
        upgrade_prediction: upgradePrediction,
        lifecycle_transition: stateTransition.shouldTransition ? {
          from: currentLifecycleStage,
          to: stateTransition.newState!,
          confidence: stateTransition.confidence,
          trigger_events: stateTransition.triggerEvents
        } : undefined,
        processed_at: new Date()
      };

      // Update user profile if state transition occurred
      if (stateTransition.shouldTransition) {
        await this.updateUserProfile(userId, stateTransition, metrics, churnPrediction.churn_probability);
      }

      // Store analysis results
      await this.storeAnalysisResult(analysisResult);

      // Cache key metrics for faster access
      await this.cacheUserMetrics(userId, metrics, churnPrediction.churn_probability);

      const processingTime = Date.now() - startTime;

      logger.info('Behavioral analysis completed', {
        userId,
        processingTime: `${processingTime}ms`,
        lifecycleStage: stateTransition.newState || currentLifecycleStage,
        engagementScore: metrics.engagement_score,
        churnRisk: churnPrediction.risk_level,
        patternsDetected: patterns.length
      });

      return analysisResult;

    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      logger.error('Behavioral analysis failed', {
        userId,
        processingTime: `${processingTime}ms`,
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined
      });

      throw error;
    }
  }

  /**
   * Batch process multiple users
   */
  public async analyzeBatch(userEventMap: Map<string, UserEvent[]>): Promise<Map<string, AnalysisResult>> {
    const results = new Map<string, AnalysisResult>();
    const batchSize = this.config.batch_size;
    const userIds = Array.from(userEventMap.keys());

    logger.info('Starting batch behavioral analysis', {
      totalUsers: userIds.length,
      batchSize
    });

    for (let i = 0; i < userIds.length; i += batchSize) {
      const batch = userIds.slice(i, i + batchSize);
      const batchPromises = batch.map(async (userId) => {
        try {
          const events = userEventMap.get(userId) || [];
          const result = await this.analyzeBehavior(userId, events);
          return { userId, result };
        } catch (error) {
          logger.error('Batch analysis failed for user', {
            userId,
            error: error instanceof Error ? error.message : String(error)
          });
          return null;
        }
      });

      const batchResults = await Promise.all(batchPromises);
      
      batchResults.forEach(item => {
        if (item) {
          results.set(item.userId, item.result);
        }
      });

      logger.debug('Processed batch', {
        batchNumber: Math.floor(i / batchSize) + 1,
        processed: Math.min(i + batchSize, userIds.length),
        total: userIds.length
      });
    }

    logger.info('Batch behavioral analysis completed', {
      totalUsers: userIds.length,
      successfulAnalyses: results.size,
      failedAnalyses: userIds.length - results.size
    });

    return results;
  }

  /**
   * Get user's current behavioral status
   */
  public async getUserBehavioralStatus(userId: string): Promise<{
    metrics: BehavioralMetrics | null;
    lifecycleStage: UserLifecycleStage;
    churnRisk: number;
    lastAnalyzed: Date | null;
  }> {
    try {
      // Try to get cached metrics first
      const cachedMetrics = await this.getCachedUserMetrics(userId);
      
      if (cachedMetrics) {
        return cachedMetrics;
      }

      // Fall back to database
      const userProfile = await this.getUserProfile(userId);
      
      return {
        metrics: null,
        lifecycleStage: userProfile?.lifecycle_stage || UserLifecycleStage.NEW,
        churnRisk: userProfile?.churn_risk_score || 0.5,
        lastAnalyzed: userProfile?.updated_at || null
      };

    } catch (error) {
      logger.error('Failed to get user behavioral status', {
        userId,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        metrics: null,
        lifecycleStage: UserLifecycleStage.NEW,
        churnRisk: 0.5,
        lastAnalyzed: null
      };
    }
  }

  // Private helper methods

  private filterEventsByWindow(events: UserEvent[]): UserEvent[] {
    if (this.config.analysis_window_days <= 0) {
      return events;
    }

    const windowStart = new Date();
    windowStart.setDate(windowStart.getDate() - this.config.analysis_window_days);

    return events.filter(event => new Date(event.timestamp) >= windowStart);
  }

  private async getUserProfile(userId: string): Promise<UserProfile | null> {
    try {
      const query = `
        SELECT user_id, external_user_id, email, name, lifecycle_stage,
               engagement_score, churn_risk_score, ltv_prediction,
               created_at, updated_at, subscription_status, subscription_plan,
               last_active_at, metadata
        FROM user_profiles.profiles
        WHERE external_user_id = $1
      `;
      
      const result = await database.query(query, [userId]);
      
      if (result.rows.length === 0) {
        return null;
      }

      const row = result.rows[0];
      return {
        user_id: row.user_id,
        external_user_id: row.external_user_id,
        email: row.email,
        name: row.name,
        lifecycle_stage: row.lifecycle_stage as UserLifecycleStage,
        engagement_score: row.engagement_score,
        churn_risk_score: row.churn_risk_score,
        ltv_prediction: row.ltv_prediction,
        created_at: row.created_at,
        updated_at: row.updated_at,
        subscription_status: row.subscription_status,
        subscription_plan: row.subscription_plan,
        last_active_at: row.last_active_at,
        metadata: row.metadata || {}
      };

    } catch (error) {
      logger.error('Failed to get user profile', {
        userId,
        error: error instanceof Error ? error.message : String(error)
      });
      return null;
    }
  }

  private async updateUserProfile(
    userId: string,
    stateTransition: StateTransitionResult,
    metrics: BehavioralMetrics,
    churnRisk: number
  ): Promise<void> {
    try {
      const query = `
        INSERT INTO user_profiles.profiles (
          external_user_id, lifecycle_stage, engagement_score,
          churn_risk_score, ltv_prediction, last_active_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (external_user_id) 
        DO UPDATE SET
          lifecycle_stage = EXCLUDED.lifecycle_stage,
          engagement_score = EXCLUDED.engagement_score,
          churn_risk_score = EXCLUDED.churn_risk_score,
          ltv_prediction = EXCLUDED.ltv_prediction,
          last_active_at = EXCLUDED.last_active_at,
          updated_at = EXCLUDED.updated_at
      `;

      const lastActiveAt = new Date();
      lastActiveAt.setDate(lastActiveAt.getDate() - metrics.days_since_last_active);

      await database.query(query, [
        userId,
        stateTransition.newState,
        metrics.engagement_score,
        churnRisk,
        metrics.lifetime_value_score,
        lastActiveAt,
        new Date()
      ]);

    } catch (error) {
      logger.error('Failed to update user profile', {
        userId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private calculateEngagementScores(metrics: BehavioralMetrics): EngagementScores {
    return {
      user_id: metrics.user_id,
      overall_score: metrics.engagement_score,
      feature_usage_score: metrics.feature_adoption_rate,
      frequency_score: Math.min(metrics.session_frequency * 2, 1), // Normalize frequency
      depth_score: Math.min(metrics.avg_session_duration / 30, 1), // Normalize by 30 min sessions
      recency_score: Math.max(0, 1 - (metrics.days_since_last_active / 30)), // Decay over 30 days
      trend_score: Math.max(0, Math.min(metrics.engagement_trend + 1, 1)), // Normalize trend
      calculated_at: new Date()
    };
  }

  private async storeAnalysisResult(result: AnalysisResult): Promise<void> {
    try {
      const query = `
        INSERT INTO behavioral_analysis.analysis_results (
          user_id, metrics, patterns, churn_prediction,
          ltv_prediction, upgrade_prediction, lifecycle_transition,
          processed_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
      `;

      await database.query(query, [
        result.user_id,
        JSON.stringify(result.metrics),
        JSON.stringify(result.patterns),
        JSON.stringify(result.churn_prediction),
        JSON.stringify(result.ltv_prediction),
        JSON.stringify(result.upgrade_prediction),
        JSON.stringify(result.lifecycle_transition),
        result.processed_at
      ]);

    } catch (error) {
      logger.error('Failed to store analysis result', {
        userId: result.user_id,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async cacheUserMetrics(
    userId: string,
    metrics: BehavioralMetrics,
    churnRisk: number
  ): Promise<void> {
    try {
      const cacheData = {
        metrics,
        lifecycleStage: 'unknown', // Would be set from profile
        churnRisk,
        lastAnalyzed: new Date()
      };

      const key = `behavioral:user:${userId}`;
      await eventsCache.setJSON(key, cacheData, 3600); // Cache for 1 hour

    } catch (error) {
      logger.error('Failed to cache user metrics', {
        userId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async getCachedUserMetrics(userId: string): Promise<any> {
    try {
      const key = `behavioral:user:${userId}`;
      return await eventsCache.getJSON(key);
    } catch (error) {
      logger.debug('Cache miss for user metrics', { userId });
      return null;
    }
  }

  private createMinimalAnalysisResult(userId: string, events: UserEvent[]): AnalysisResult {
    const now = new Date();
    
    return {
      user_id: userId,
      metrics: {
        user_id: userId,
        calculated_at: now,
        engagement_score: 0,
        daily_active_rate: 0,
        weekly_active_days: 0,
        engagement_trend: 0,
        total_events: events.length,
        unique_event_types: new Set(events.map(e => e.event_type)).size,
        days_since_last_active: 999,
        total_sessions: 0,
        avg_session_duration: 0,
        avg_events_per_session: 0,
        session_frequency: 0,
        session_regularity: 0,
        bounce_rate: 0,
        pages_per_session: 0,
        feature_adoption_rate: 0,
        feature_depth: 0,
        feature_breadth: 0,
        power_features_used: 0,
        unique_features_used: 0,
        feature_usage_depth: 0,
        new_feature_adoption: 0,
        feature_stickiness: 0,
        most_used_features: [],
        monetization_score: 0,
        pricing_page_views: 0,
        upgrade_attempts: 0,
        monetization_events_count: 0,
        days_since_last_monetization_event: 999,
        error_rate: 0,
        support_tickets: 0,
        usage_decline: 0,
        payment_failures: 0,
        cancellation_signals: 0,
        days_since_signup: 0,
        lifetime_value_score: 0,
        upgrade_probability: 0,
        churn_risk_score: 0
      },
      patterns: [],
      engagement_scores: {
        user_id: userId,
        overall_score: 0,
        feature_usage_score: 0,
        frequency_score: 0,
        depth_score: 0,
        recency_score: 0,
        trend_score: 0,
        calculated_at: now
      },
      churn_prediction: {
        user_id: userId,
        churn_probability: 0.5,
        risk_level: 'medium',
        contributing_factors: ['Insufficient data'],
        recommended_actions: ['Collect more behavioral data'],
        prediction_date: now,
        model_version: 'minimal-1.0'
      },
      ltv_prediction: {
        user_id: userId,
        predicted_ltv: 100,
        confidence_interval: [50, 150],
        time_horizon_days: 365,
        contributing_factors: { insufficient_data: 1.0 },
        prediction_date: now,
        model_version: 'minimal-1.0'
      },
      upgrade_prediction: {
        user_id: userId,
        upgrade_probability: 0.2,
        optimal_timing_days: 30,
        recommended_plan: 'Premium',
        motivation_factors: ['Insufficient data'],
        prediction_date: now,
        model_version: 'minimal-1.0'
      },
      processed_at: now
    };
  }

  /**
   * Health check for the analysis engine
   */
  public async healthCheck(): Promise<{
    healthy: boolean;
    details: Record<string, any>;
  }> {
    const details: Record<string, any> = {};
    let healthy = true;

    try {
      // Check database connectivity
      const dbHealthy = await database.healthCheck();
      details.database = dbHealthy ? 'healthy' : 'unhealthy';
      if (!dbHealthy) healthy = false;

      // Check Redis connectivity
      const redisHealthy = await eventsCache.healthCheck();
      details.redis = redisHealthy ? 'healthy' : 'unhealthy';
      if (!redisHealthy) healthy = false;

      // Check component initialization
      details.components = {
        metricsCalculator: !!this.metricsCalculator,
        stateMachine: !!this.stateMachine,
        patternDetector: !!this.patternDetector,
        scoringModels: !!this.scoringModels
      };

      details.config = this.config;

    } catch (error) {
      healthy = false;
      details.error = error instanceof Error ? error.message : String(error);
    }

    return { healthy, details };
  }
} 