import { BehavioralMetrics, ChurnPrediction, LTVPrediction, UpgradePrediction, ModelFeatures } from '../types';
import { logger } from '@userwhisperer/shared';

export interface ScoringConfig {
  churnModel: {
    enabled: boolean;
    modelPath?: string;
    version: string;
    features: string[];
  };
  ltvModel: {
    enabled: boolean;
    modelPath?: string;
    version: string;
    baseValue: number;
    timeHorizonDays: number;
  };
  upgradeModel: {
    enabled: boolean;
    modelPath?: string;
    version: string;
    features: string[];
  };
}

export interface ModelPredictionResult {
  prediction: number;
  confidence: number;
  features: ModelFeatures;
  modelUsed: 'ml' | 'rule_based';
}

export class ScoringModels {
  private config: ScoringConfig;
  private churnModel: any = null;
  private ltvModel: any = null;
  private upgradeModel: any = null;

  constructor(config: ScoringConfig) {
    this.config = config;
    this.initializeModels();
  }

  /**
   * Calculate churn risk score (0-1, higher = more risk)
   */
  public async calculateChurnRisk(metrics: BehavioralMetrics): Promise<ChurnPrediction> {
    try {
      const features = this.extractChurnFeatures(metrics);
      let prediction: ModelPredictionResult;

      if (this.config.churnModel.enabled && this.churnModel) {
        prediction = await this.predictWithMLModel(this.churnModel, features, 'churn');
      } else {
        prediction = this.calculateRuleBasedChurnRisk(metrics, features);
      }

      const riskLevel = this.determineRiskLevel(prediction.prediction);
      const contributingFactors = this.identifyChurnFactors(metrics, features);
      const recommendedActions = this.generateChurnActions(riskLevel, contributingFactors);

      return {
        user_id: metrics.user_id,
        churn_probability: prediction.prediction,
        risk_level: riskLevel,
        contributing_factors: contributingFactors,
        recommended_actions: recommendedActions,
        prediction_date: new Date(),
        model_version: this.config.churnModel.version
      };

    } catch (error) {
      logger.error('Churn prediction failed', {
        userId: metrics.user_id,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return safe default
      return this.createSafeChurnPrediction(metrics.user_id);
    }
  }

  /**
   * Calculate customer lifetime value prediction
   */
  public async calculateLTVPrediction(metrics: BehavioralMetrics): Promise<LTVPrediction> {
    try {
      const features = this.extractLTVFeatures(metrics);
      let prediction: ModelPredictionResult;

      if (this.config.ltvModel.enabled && this.ltvModel) {
        prediction = await this.predictWithMLModel(this.ltvModel, features, 'ltv');
      } else {
        prediction = this.calculateRuleBasedLTV(metrics, features);
      }

      const contributingFactors = this.identifyLTVFactors(metrics, features);
      const confidenceInterval = this.calculateLTVConfidenceInterval(
        prediction.prediction,
        prediction.confidence
      );

      return {
        user_id: metrics.user_id,
        predicted_ltv: prediction.prediction,
        confidence_interval: confidenceInterval,
        time_horizon_days: this.config.ltvModel.timeHorizonDays,
        contributing_factors: contributingFactors,
        prediction_date: new Date(),
        model_version: this.config.ltvModel.version
      };

    } catch (error) {
      logger.error('LTV prediction failed', {
        userId: metrics.user_id,
        error: error instanceof Error ? error.message : String(error)
      });

      return this.createSafeLTVPrediction(metrics.user_id);
    }
  }

  /**
   * Calculate upgrade probability
   */
  public async calculateUpgradePrediction(metrics: BehavioralMetrics): Promise<UpgradePrediction> {
    try {
      const features = this.extractUpgradeFeatures(metrics);
      let prediction: ModelPredictionResult;

      if (this.config.upgradeModel.enabled && this.upgradeModel) {
        prediction = await this.predictWithMLModel(this.upgradeModel, features, 'upgrade');
      } else {
        prediction = this.calculateRuleBasedUpgradeProbability(metrics, features);
      }

      const motivationFactors = this.identifyUpgradeMotivations(metrics, features);
      const optimalTiming = this.calculateOptimalUpgradeTiming(metrics);
      const recommendedPlan = this.recommendUpgradePlan(metrics, features);

      return {
        user_id: metrics.user_id,
        upgrade_probability: prediction.prediction,
        optimal_timing_days: optimalTiming,
        recommended_plan: recommendedPlan,
        motivation_factors: motivationFactors,
        prediction_date: new Date(),
        model_version: this.config.upgradeModel.version
      };

    } catch (error) {
      logger.error('Upgrade prediction failed', {
        userId: metrics.user_id,
        error: error instanceof Error ? error.message : String(error)
      });

      return this.createSafeUpgradePrediction(metrics.user_id);
    }
  }

  // ML Model Methods

  private async initializeModels(): Promise<void> {
    try {
      // In a real implementation, these would load actual ML models
      // For now, we'll use rule-based models as fallbacks
      
      if (this.config.churnModel.enabled) {
        // this.churnModel = await this.loadModel(this.config.churnModel.modelPath);
        logger.info('Churn model initialization skipped (using rule-based)');
      }

      if (this.config.ltvModel.enabled) {
        // this.ltvModel = await this.loadModel(this.config.ltvModel.modelPath);
        logger.info('LTV model initialization skipped (using rule-based)');
      }

      if (this.config.upgradeModel.enabled) {
        // this.upgradeModel = await this.loadModel(this.config.upgradeModel.modelPath);
        logger.info('Upgrade model initialization skipped (using rule-based)');
      }

    } catch (error) {
      logger.error('Model initialization failed', {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async predictWithMLModel(
    model: any,
    features: ModelFeatures,
    modelType: string
  ): Promise<ModelPredictionResult> {
    // This would call the actual ML model
    // For now, fall back to rule-based
    logger.debug(`ML model not available for ${modelType}, using rule-based fallback`);
    
    // Return a placeholder that indicates ML model usage
    return {
      prediction: 0.5, // This would be the actual ML prediction
      confidence: 0.7,
      features,
      modelUsed: 'ml'
    };
  }

  // Rule-based Prediction Methods

  private calculateRuleBasedChurnRisk(
    metrics: BehavioralMetrics,
    features: ModelFeatures
  ): ModelPredictionResult {
    let riskScore = 0.0;

    // Engagement factors (40% weight)
    const engagementScore = metrics.engagement_score;
    riskScore += (1 - engagementScore) * 0.4;

    // Activity factors (30% weight)
    const daysInactive = metrics.days_since_last_active;
    if (daysInactive >= 14) {
      riskScore += 0.3;
    } else if (daysInactive >= 7) {
      riskScore += 0.2;
    } else if (daysInactive >= 3) {
      riskScore += 0.1;
    }

    // Trend factors (20% weight)
    const engagementTrend = metrics.engagement_trend;
    if (engagementTrend < -0.5) {
      riskScore += 0.2;
    } else if (engagementTrend < -0.2) {
      riskScore += 0.1;
    }

    // Risk indicators (10% weight)
    if (metrics.payment_failures > 0) riskScore += 0.05;
    if (metrics.support_tickets >= 3) riskScore += 0.05;
    if (metrics.cancellation_signals > 0) riskScore += 0.1;
    if (metrics.error_rate > 0.1) riskScore += 0.03;

    const finalScore = Math.min(riskScore, 1.0);
    const confidence = this.calculateRuleBasedConfidence(metrics);

    return {
      prediction: finalScore,
      confidence,
      features,
      modelUsed: 'rule_based'
    };
  }

  private calculateRuleBasedLTV(
    metrics: BehavioralMetrics,
    features: ModelFeatures
  ): ModelPredictionResult {
    const baseLTV = this.config.ltvModel.baseValue;

    // Engagement multiplier
    const engagementMultiplier = 1 + metrics.engagement_score;

    // Feature adoption multiplier
    const featureMultiplier = 1 + metrics.feature_adoption_rate;

    // Tenure multiplier (caps at 3x for users over 90 days)
    const tenureMultiplier = Math.min(metrics.days_since_signup / 30, 3);

    // Monetization multiplier
    const monetizationMultiplier = 1 + (metrics.monetization_score * 2);

    // Risk penalty
    const churnRisk = this.calculateSimpleChurnRisk(metrics);
    const riskPenalty = 1 - (churnRisk * 0.5); // Max 50% penalty

    const ltvPrediction = baseLTV * 
      engagementMultiplier * 
      featureMultiplier * 
      tenureMultiplier * 
      monetizationMultiplier * 
      riskPenalty;

    const confidence = this.calculateRuleBasedConfidence(metrics);

    return {
      prediction: Math.round(ltvPrediction * 100) / 100,
      confidence,
      features,
      modelUsed: 'rule_based'
    };
  }

  private calculateRuleBasedUpgradeProbability(
    metrics: BehavioralMetrics,
    features: ModelFeatures
  ): ModelPredictionResult {
    let upgradeScore = 0;

    // Monetization signals (40% weight)
    upgradeScore += metrics.monetization_score * 0.4;

    // Usage at limits (30% weight)
    if (metrics.feature_adoption_rate > 0.8) {
      upgradeScore += 0.3;
    } else if (metrics.feature_adoption_rate > 0.6) {
      upgradeScore += 0.2;
    } else if (metrics.feature_adoption_rate > 0.4) {
      upgradeScore += 0.1;
    }

    // Engagement level (20% weight)
    if (metrics.engagement_score > 0.7) {
      upgradeScore += 0.2;
    } else if (metrics.engagement_score > 0.5) {
      upgradeScore += 0.1;
    }

    // Power features usage (10% weight)
    if (metrics.power_features_used >= 3) {
      upgradeScore += 0.1;
    } else if (metrics.power_features_used >= 1) {
      upgradeScore += 0.05;
    }

    // Bonuses for specific behaviors
    if (metrics.pricing_page_views > 0) upgradeScore += 0.1;
    if (metrics.upgrade_attempts > 0) upgradeScore += 0.15;

    // Penalties
    if (metrics.payment_failures > 0) upgradeScore -= 0.2;
    if (metrics.support_tickets > 3) upgradeScore -= 0.1;

    const finalScore = Math.max(0, Math.min(upgradeScore, 1.0));
    const confidence = this.calculateRuleBasedConfidence(metrics);

    return {
      prediction: finalScore,
      confidence,
      features,
      modelUsed: 'rule_based'
    };
  }

  // Feature Extraction Methods

  private extractChurnFeatures(metrics: BehavioralMetrics): ModelFeatures {
    return {
      daily_active_rate: metrics.daily_active_rate,
      weekly_active_days: metrics.weekly_active_days,
      session_frequency: metrics.session_frequency,
      avg_session_duration: metrics.avg_session_duration,
      total_events: metrics.total_events,
      unique_event_types: metrics.unique_event_types,
      feature_adoption_rate: metrics.feature_adoption_rate,
      power_features_used: metrics.power_features_used,
      days_since_signup: metrics.days_since_signup,
      days_since_last_active: metrics.days_since_last_active,
      engagement_trend: metrics.engagement_trend,
      error_rate: metrics.error_rate,
      support_tickets: metrics.support_tickets,
      payment_failures: metrics.payment_failures,
      monetization_score: metrics.monetization_score,
      pricing_page_views: metrics.pricing_page_views,
      upgrade_attempts: metrics.upgrade_attempts
    };
  }

  private extractLTVFeatures(metrics: BehavioralMetrics): ModelFeatures {
    return {
      daily_active_rate: metrics.daily_active_rate,
      weekly_active_days: metrics.weekly_active_days,
      session_frequency: metrics.session_frequency,
      avg_session_duration: metrics.avg_session_duration,
      total_events: metrics.total_events,
      unique_event_types: metrics.unique_event_types,
      feature_adoption_rate: metrics.feature_adoption_rate,
      power_features_used: metrics.power_features_used,
      days_since_signup: metrics.days_since_signup,
      days_since_last_active: metrics.days_since_last_active,
      engagement_trend: metrics.engagement_trend,
      error_rate: metrics.error_rate,
      support_tickets: metrics.support_tickets,
      payment_failures: metrics.payment_failures,
      monetization_score: metrics.monetization_score,
      pricing_page_views: metrics.pricing_page_views,
      upgrade_attempts: metrics.upgrade_attempts
    };
  }

  private extractUpgradeFeatures(metrics: BehavioralMetrics): ModelFeatures {
    return this.extractLTVFeatures(metrics); // Same features for now
  }

  // Helper Methods

  private calculateSimpleChurnRisk(metrics: BehavioralMetrics): number {
    // Simple churn risk calculation for internal use
    let risk = 0;
    
    risk += (1 - metrics.engagement_score) * 0.4;
    risk += Math.min(metrics.days_since_last_active / 30, 1) * 0.3;
    risk += Math.min(metrics.support_tickets / 5, 1) * 0.2;
    risk += metrics.payment_failures > 0 ? 0.1 : 0;
    
    return Math.min(risk, 1.0);
  }

  private calculateRuleBasedConfidence(metrics: BehavioralMetrics): number {
    let confidence = 1.0;

    // Reduce confidence if insufficient data
    if (metrics.total_events < 10) confidence *= 0.7;
    if (metrics.days_since_signup < 7) confidence *= 0.8;
    if (metrics.total_sessions < 3) confidence *= 0.9;

    return Math.max(confidence, 0.1); // Minimum 10% confidence
  }

  private determineRiskLevel(churnProbability: number): 'low' | 'medium' | 'high' | 'critical' {
    if (churnProbability >= 0.8) return 'critical';
    if (churnProbability >= 0.6) return 'high';
    if (churnProbability >= 0.4) return 'medium';
    return 'low';
  }

  private identifyChurnFactors(metrics: BehavioralMetrics, features: ModelFeatures): string[] {
    const factors: string[] = [];

    if (metrics.engagement_score < 0.3) factors.push('Low engagement');
    if (metrics.days_since_last_active >= 7) factors.push('Extended inactivity');
    if (metrics.engagement_trend < -0.3) factors.push('Declining usage');
    if (metrics.support_tickets >= 3) factors.push('Support issues');
    if (metrics.payment_failures > 0) factors.push('Payment problems');
    if (metrics.error_rate > 0.1) factors.push('High error rate');
    if (metrics.cancellation_signals > 0) factors.push('Cancellation interest');

    return factors;
  }

  private generateChurnActions(
    riskLevel: 'low' | 'medium' | 'high' | 'critical',
    factors: string[]
  ): string[] {
    const actions: string[] = [];

    switch (riskLevel) {
      case 'critical':
        actions.push('Immediate personal outreach');
        actions.push('Executive escalation');
        actions.push('Custom retention offer');
        break;
      case 'high':
        actions.push('Priority customer success call');
        actions.push('Targeted retention campaign');
        actions.push('Feature training session');
        break;
      case 'medium':
        actions.push('Re-engagement email series');
        actions.push('Usage tip notifications');
        actions.push('Check-in survey');
        break;
      case 'low':
        actions.push('Continue monitoring');
        actions.push('Standard nurture campaign');
        break;
    }

    // Add factor-specific actions
    if (factors.includes('Support issues')) {
      actions.push('Proactive support check-in');
    }
    if (factors.includes('Payment problems')) {
      actions.push('Billing support contact');
    }

    return actions;
  }

  private identifyLTVFactors(metrics: BehavioralMetrics, features: ModelFeatures): Record<string, number> {
    return {
      engagement_impact: metrics.engagement_score * 0.3,
      feature_adoption_impact: metrics.feature_adoption_rate * 0.25,
      tenure_impact: Math.min(metrics.days_since_signup / 365, 1) * 0.2,
      monetization_impact: metrics.monetization_score * 0.15,
      risk_impact: -(this.calculateSimpleChurnRisk(metrics) * 0.1)
    };
  }

  private calculateLTVConfidenceInterval(prediction: number, confidence: number): [number, number] {
    const margin = prediction * (1 - confidence) * 0.5;
    return [
      Math.max(0, prediction - margin),
      prediction + margin
    ];
  }

  private identifyUpgradeMotivations(metrics: BehavioralMetrics, features: ModelFeatures): string[] {
    const motivations: string[] = [];

    if (metrics.feature_adoption_rate > 0.7) motivations.push('High feature usage');
    if (metrics.power_features_used > 0) motivations.push('Power user behavior');
    if (metrics.pricing_page_views > 0) motivations.push('Pricing interest');
    if (metrics.engagement_score > 0.8) motivations.push('High engagement');
    if (metrics.monetization_score > 0.5) motivations.push('Monetization signals');

    return motivations;
  }

  private calculateOptimalUpgradeTiming(metrics: BehavioralMetrics): number {
    // Days until optimal upgrade timing
    let optimalDays = 7; // Default to 1 week

    if (metrics.engagement_score > 0.8) optimalDays = 3; // Strike while hot
    if (metrics.pricing_page_views > 0) optimalDays = 1; // Very interested
    if (metrics.upgrade_attempts > 0) optimalDays = 0; // Immediate

    return optimalDays;
  }

  private recommendUpgradePlan(metrics: BehavioralMetrics, features: ModelFeatures): string {
    if (metrics.power_features_used >= 3) return 'Enterprise';
    if (metrics.feature_adoption_rate > 0.6) return 'Professional';
    return 'Premium';
  }

  // Safe Fallback Methods

  private createSafeChurnPrediction(userId: string): ChurnPrediction {
    return {
      user_id: userId,
      churn_probability: 0.5,
      risk_level: 'medium',
      contributing_factors: ['Insufficient data'],
      recommended_actions: ['Collect more user data'],
      prediction_date: new Date(),
      model_version: 'fallback-1.0'
    };
  }

  private createSafeLTVPrediction(userId: string): LTVPrediction {
    return {
      user_id: userId,
      predicted_ltv: this.config.ltvModel.baseValue,
      confidence_interval: [this.config.ltvModel.baseValue * 0.5, this.config.ltvModel.baseValue * 1.5],
      time_horizon_days: this.config.ltvModel.timeHorizonDays,
      contributing_factors: { insufficient_data: 1.0 },
      prediction_date: new Date(),
      model_version: 'fallback-1.0'
    };
  }

  private createSafeUpgradePrediction(userId: string): UpgradePrediction {
    return {
      user_id: userId,
      upgrade_probability: 0.3,
      optimal_timing_days: 14,
      recommended_plan: 'Premium',
      motivation_factors: ['Insufficient data'],
      prediction_date: new Date(),
      model_version: 'fallback-1.0'
    };
  }
} 