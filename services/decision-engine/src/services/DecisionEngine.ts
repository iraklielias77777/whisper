import {
  DecisionContext,
  InterventionDecision,
  InterventionType,
  UrgencyLevel,
  Channel,
  Strategy,
  Action,
  MLPrediction,
  DecisionEngineConfig,
  ContentStrategy,
  FatigueCheck
} from '../types';
import { logger } from '@userwhisperer/shared';
import { StrategySelector } from './StrategySelector';
import { ChannelSelector } from './ChannelSelector';
import { TimingOptimizer } from './TimingOptimizer';
import { FatigueManager } from './FatigueManager';
import { v4 as uuidv4 } from 'uuid';

export class DecisionEngine {
  private config: DecisionEngineConfig;
  private strategySelector: StrategySelector;
  private channelSelector: ChannelSelector;
  private timingOptimizer: TimingOptimizer;
  private fatigueManager: FatigueManager;
  private mlModel: any = null; // Placeholder for ML model

  constructor(config: DecisionEngineConfig) {
    this.config = config;
    this.strategySelector = new StrategySelector();
    this.channelSelector = new ChannelSelector(config.channel_costs);
    this.timingOptimizer = new TimingOptimizer(config.default_timezone);
    this.fatigueManager = new FatigueManager(config.fatigue_limits);
    
    if (config.ml_model_enabled) {
      this.initializeMLModel();
    }
  }

  /**
   * Main decision-making pipeline
   */
  public async makeDecision(context: DecisionContext, aiInsights?: any): Promise<InterventionDecision> {
    const startTime = Date.now();
    const decisionId = uuidv4();

    try {
      logger.info('Starting decision-making process', {
        userId: context.user_id,
        decisionId,
        triggerEvent: context.trigger_event.event_type
      });

      // Step 1: Check if intervention is warranted
      const interventionScore = this.calculateInterventionScore(context);
      
      if (interventionScore < this.config.intervention_threshold) {
        return this.createNoInterventionDecision(
          decisionId,
          'Intervention score below threshold',
          interventionScore
        );
      }

      // Step 2: Check for message fatigue
      const fatigueCheck = await this.fatigueManager.checkFatigue(
        context.user_id,
        context.message_history
      );

      if (fatigueCheck.is_fatigued) {
        return this.createNoInterventionDecision(
          decisionId,
          `User fatigued: ${fatigueCheck.reason}`,
          interventionScore
        );
      }

      // Step 3: Determine intervention type and strategy
      let strategy = this.strategySelector.selectStrategy(context);
      
      // Step 3a: Enhance strategy with AI orchestration insights if available
      if (aiInsights && aiInsights.strategy_decisions) {
        strategy = {
          ...strategy,
          intervention_type: aiInsights.strategy_decisions.intervention_type || strategy.intervention_type,
          recommended_channel: aiInsights.strategy_decisions.channel || strategy.recommended_channel,
          ai_confidence: aiInsights.ai_insights.confidence,
          ai_reasoning: aiInsights.ai_insights.reasoning,
          ai_enhanced: true
        };
        
        logger.info('Strategy enhanced with AI orchestration insights', {
          decisionId,
          userId: context.user_id,
          originalStrategy: this.strategySelector.selectStrategy(context).intervention_type,
          aiRecommendedStrategy: aiInsights.strategy_decisions.intervention_type,
          aiConfidence: aiInsights.ai_insights.confidence
        });
      }

      // Step 4: Use ML to predict best approach (if enabled)
      const mlPrediction = await this.predictBestApproach(context, strategy);

      // Step 5: Select optimal channel (prioritize AI recommendation if available)
      const channel = await this.channelSelector.selectChannel(
        context,
        strategy,
        mlPrediction
      );

      // Step 6: Optimize timing
      const sendTime = await this.timingOptimizer.optimizeTiming(
        context,
        strategy,
        channel
      );

      // Step 7: Prepare content strategy
      const contentStrategy = this.prepareContentStrategy(
        context,
        strategy,
        mlPrediction
      );

      const decision: InterventionDecision = {
        should_intervene: true,
        intervention_type: strategy.intervention_type,
        urgency: strategy.urgency,
        channel,
        send_time: sendTime,
        content_strategy: contentStrategy,
        confidence_score: mlPrediction.confidence,
        reasoning: this.buildReasoning(context, strategy, mlPrediction),
        decision_id: decisionId,
        created_at: new Date()
      };

      const processingTime = Date.now() - startTime;

      logger.info('Decision completed successfully', {
        userId: context.user_id,
        decisionId,
        shouldIntervene: true,
        interventionType: strategy.intervention_type,
        channel,
        urgency: strategy.urgency,
        confidence: mlPrediction.confidence,
        processingTime: `${processingTime}ms`
      });

      return decision;

    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      logger.error('Decision-making failed', {
        userId: context.user_id,
        decisionId,
        error: error instanceof Error ? error.message : String(error),
        processingTime: `${processingTime}ms`
      });

      // Return safe fallback decision
      return this.createNoInterventionDecision(
        decisionId,
        `Error in decision-making: ${error instanceof Error ? error.message : 'Unknown error'}`,
        0.5
      );
    }
  }

  /**
   * Calculate overall intervention score based on behavioral signals
   */
  private calculateInterventionScore(context: DecisionContext): number {
    let score = 0.0;
    const weights = this.config.intervention_weights;

    // Churn risk component (highest priority)
    const churnRisk = context.behavioral_scores.churn_risk || 0;
    if (churnRisk > 0.7) {
      score += 1.0 * weights.churn; // Critical
    } else if (churnRisk > 0.5) {
      score += 0.7 * weights.churn; // High
    } else if (churnRisk > 0.3) {
      score += 0.4 * weights.churn; // Medium
    }

    // Monetization opportunity
    const upgradeProbability = context.behavioral_scores.upgrade_probability || 0;
    score += upgradeProbability * weights.monetization;

    // Onboarding needs
    if (context.user_profile.lifecycle_stage === 'onboarding') {
      const daysSinceSignup = context.user_profile.days_since_signup;
      if (daysSinceSignup <= 3) {
        score += 0.8 * weights.onboarding;
      } else if (daysSinceSignup <= 7) {
        score += 0.5 * weights.onboarding;
      }
    }

    // Support needs
    const supportTickets = context.behavioral_scores.support_tickets || 0;
    if (supportTickets > 0) {
      score += Math.min(supportTickets * 0.2, 0.6) * weights.support;
    }

    // Celebration opportunities
    if (this.hasAchievement(context)) {
      score += 0.4 * weights.celebration;
    }

    // Event-driven triggers (critical events get immediate attention)
    if (this.isCriticalEvent(context.trigger_event)) {
      score += 0.9 * weights.event_trigger;
    }

    return Math.min(score, 1.0);
  }

  /**
   * Check if user has reached an achievement worth celebrating
   */
  private hasAchievement(context: DecisionContext): boolean {
    const achievements = [
      // First week completion
      context.user_profile.days_since_signup === 7,
      // Power user status
      (context.behavioral_scores.engagement_score || 0) > 0.8,
      // Feature mastery
      (context.behavioral_scores.feature_adoption_rate || 0) > 0.7,
      // Milestone event
      context.trigger_event.event_type === 'milestone_reached',
      // Upgrade completion
      context.trigger_event.event_type === 'subscription_upgraded',
      // Goal achievement
      context.trigger_event.event_type === 'goal_completed'
    ];

    return achievements.some(achieved => achieved);
  }

  /**
   * Check if event requires immediate intervention
   */
  private isCriticalEvent(event: any): boolean {
    const criticalEvents = [
      'payment_failed',
      'subscription_cancelled',
      'subscription_cancel_clicked',
      'error_critical',
      'limit_reached',
      'trial_ending',
      'account_downgrade',
      'multiple_login_failures',
      'data_loss_detected',
      'security_breach_detected'
    ];

    return criticalEvents.includes(event.event_type);
  }

  /**
   * Use ML model to predict best intervention approach
   */
  private async predictBestApproach(
    context: DecisionContext,
    strategy: Strategy
  ): Promise<MLPrediction> {
    try {
      // Extract features for ML model
      const features = this.extractFeatures(context, strategy);
      
      // Get possible actions for this strategy
      const possibleActions = this.getPossibleActions(strategy);

      if (this.config.ml_model_enabled && this.mlModel) {
        // Use ML model for prediction
        return await this.mlPredict(features, possibleActions);
      } else {
        // Use heuristic prediction as fallback
        return this.heuristicPrediction(features, possibleActions);
      }

    } catch (error) {
      logger.error('ML prediction failed, using heuristic fallback', {
        userId: context.user_id,
        error: error instanceof Error ? error.message : String(error)
      });

      // Fallback to heuristics
      const features = this.extractFeatures(context, strategy);
      const possibleActions = this.getPossibleActions(strategy);
      return this.heuristicPrediction(features, possibleActions);
    }
  }

  /**
   * Extract features for ML model or heuristic prediction
   */
  private extractFeatures(context: DecisionContext, strategy: Strategy): Record<string, any> {
    return {
      // User characteristics
      days_since_signup: context.user_profile.days_since_signup,
      lifecycle_stage: context.user_profile.lifecycle_stage,
      subscription_plan: context.user_profile.subscription_plan || 'free',
      
      // Behavioral scores
      churn_risk: context.behavioral_scores.churn_risk || 0,
      engagement_score: context.behavioral_scores.engagement_score || 0,
      feature_adoption_rate: context.behavioral_scores.feature_adoption_rate || 0,
      upgrade_probability: context.behavioral_scores.upgrade_probability || 0,
      monetization_score: context.behavioral_scores.monetization_score || 0,
      days_since_last_active: context.behavioral_scores.days_since_last_active || 0,
      
      // Message history features
      total_messages_sent: context.message_history.length,
      recent_open_rate: this.calculateRecentOpenRate(context.message_history),
      recent_click_rate: this.calculateRecentClickRate(context.message_history),
      hours_since_last_message: this.hoursSinceLastMessage(context.message_history),
      
      // Event context
      event_type: context.trigger_event.event_type,
      is_critical_event: this.isCriticalEvent(context.trigger_event),
      
      // Strategy context
      intervention_type: strategy.intervention_type,
      urgency: strategy.urgency,
      
      // Timing features
      hour_of_day: context.current_time.getHours(),
      day_of_week: context.current_time.getDay(),
      is_weekend: context.current_time.getDay() === 0 || context.current_time.getDay() === 6
    };
  }

  /**
   * Get possible actions for a strategy
   */
  private getPossibleActions(strategy: Strategy): Action[] {
    const baseActions = {
      [InterventionType.RETENTION]: [
        { id: 'value_reminder', name: 'Value Reminder', type: 'educational', value: 0.8 },
        { id: 'feature_highlight', name: 'Feature Highlight', type: 'feature_focused', value: 0.6 },
        { id: 'personal_check_in', name: 'Personal Check-in', type: 'personal', value: 0.9 },
        { id: 'incentive_offer', name: 'Incentive Offer', type: 'promotional', value: 0.7 }
      ],
      [InterventionType.MONETIZATION]: [
        { id: 'upgrade_benefits', name: 'Upgrade Benefits', type: 'promotional', value: 0.9 },
        { id: 'limited_time_offer', name: 'Limited Time Offer', type: 'promotional', value: 0.7 },
        { id: 'usage_limit_warning', name: 'Usage Limit Warning', type: 'informational', value: 0.6 },
        { id: 'success_story', name: 'Success Story', type: 'social_proof', value: 0.5 }
      ],
      [InterventionType.ONBOARDING]: [
        { id: 'welcome_series', name: 'Welcome Series', type: 'educational', value: 0.8 },
        { id: 'quick_start_guide', name: 'Quick Start Guide', type: 'tutorial', value: 0.7 },
        { id: 'feature_tour', name: 'Feature Tour', type: 'tutorial', value: 0.6 },
        { id: 'personal_setup', name: 'Personal Setup', type: 'personal', value: 0.9 }
      ],
      [InterventionType.SUPPORT]: [
        { id: 'help_article', name: 'Help Article', type: 'informational', value: 0.6 },
        { id: 'personal_assistance', name: 'Personal Assistance', type: 'personal', value: 0.9 },
        { id: 'video_tutorial', name: 'Video Tutorial', type: 'tutorial', value: 0.7 },
        { id: 'community_forum', name: 'Community Forum', type: 'community', value: 0.5 }
      ],
      [InterventionType.CELEBRATION]: [
        { id: 'achievement_badge', name: 'Achievement Badge', type: 'recognition', value: 0.7 },
        { id: 'milestone_message', name: 'Milestone Message', type: 'personal', value: 0.8 },
        { id: 'social_sharing', name: 'Social Sharing', type: 'social_proof', value: 0.6 },
        { id: 'special_reward', name: 'Special Reward', type: 'promotional', value: 0.9 }
      ],
      [InterventionType.REACTIVATION]: [
        { id: 'winback_offer', name: 'Win-back Offer', type: 'promotional', value: 0.8 },
        { id: 'what_you_missed', name: 'What You Missed', type: 'informational', value: 0.6 },
        { id: 'personal_invitation', name: 'Personal Invitation', type: 'personal', value: 0.9 },
        { id: 'limited_time_return', name: 'Limited Time Return Offer', type: 'promotional', value: 0.7 }
      ],
      [InterventionType.EDUCATION]: [
        { id: 'best_practices', name: 'Best Practices Guide', type: 'educational', value: 0.7 },
        { id: 'advanced_features', name: 'Advanced Features', type: 'tutorial', value: 0.6 },
        { id: 'tips_and_tricks', name: 'Tips and Tricks', type: 'educational', value: 0.8 },
        { id: 'workflow_optimization', name: 'Workflow Optimization', type: 'educational', value: 0.9 }
      ]
    };

    const actions = baseActions[strategy.intervention_type] || baseActions[InterventionType.RETENTION];
    
    return actions.map((action: any) => ({
      ...action,
      features: { strategy_type: strategy.intervention_type, ...action }
    }));
  }

  /**
   * Heuristic prediction when ML model is not available
   */
  private heuristicPrediction(
    features: Record<string, any>,
    possibleActions: Action[]
  ): MLPrediction {
    const predictions = possibleActions.map(action => {
      let probability = action.value; // Base probability

      // Adjust based on user characteristics
      if (features.engagement_score > 0.7 && action.type === 'personal') {
        probability *= 1.2; // High engagement users respond well to personal touch
      }

      if (features.churn_risk > 0.6 && action.type === 'promotional') {
        probability *= 1.3; // At-risk users need stronger incentives
      }

      if (features.days_since_signup < 7 && action.type === 'tutorial') {
        probability *= 1.2; // New users benefit from tutorials
      }

      if (features.recent_open_rate > 0.3 && action.type === 'educational') {
        probability *= 1.1; // Engaged users like educational content
      }

      // Adjust for timing
      if (features.is_weekend && action.type === 'personal') {
        probability *= 0.9; // Personal messages slightly less effective on weekends
      }

      // Apply urgency modifiers
      if (features.urgency === UrgencyLevel.CRITICAL) {
        if (action.type === 'personal') {
          probability *= 1.5; // Personal touch critical for urgent situations
        }
      }

      const expectedValue = Math.min(probability, 1.0) * action.value;

      return {
        action,
        probability: Math.min(probability, 1.0),
        expected_value: expectedValue
      };
    });

    // Select best action
    const bestPrediction = predictions.reduce((prev, current) => 
      current.expected_value > prev.expected_value ? current : prev
    );

    return {
      action: bestPrediction.action,
      confidence: bestPrediction.probability,
      alternatives: predictions.filter(p => p.action.id !== bestPrediction.action.id)
    };
  }

  /**
   * Prepare content strategy based on decision context
   */
  private prepareContentStrategy(
    context: DecisionContext,
    strategy: Strategy,
    mlPrediction: MLPrediction
  ): ContentStrategy {
    return {
      personalization_level: this.determinePersonalizationLevel(context),
      tone: this.determineTone(context, strategy),
      approach: mlPrediction.action.type,
      key_messages: this.generateKeyMessages(context, strategy, mlPrediction),
      cta_type: this.determineCTAType(strategy, mlPrediction),
      urgency_indicators: this.generateUrgencyIndicators(strategy),
      social_proof: this.shouldIncludeSocialProof(context, mlPrediction),
      special_offer: this.generateSpecialOffer(context, strategy)
    };
  }

  /**
   * Build reasoning explanation for the decision
   */
  private buildReasoning(
    context: DecisionContext,
    strategy: Strategy,
    mlPrediction: MLPrediction
  ): string {
    const reasons = [];

    // Primary driver
    if (context.behavioral_scores.churn_risk > 0.6) {
      reasons.push(`High churn risk detected (${(context.behavioral_scores.churn_risk * 100).toFixed(0)}%)`);
    } else if (context.behavioral_scores.upgrade_probability > 0.6) {
      reasons.push(`Strong monetization opportunity (${(context.behavioral_scores.upgrade_probability * 100).toFixed(0)}% upgrade probability)`);
    } else if (this.isCriticalEvent(context.trigger_event)) {
      reasons.push(`Critical event triggered: ${context.trigger_event.event_type}`);
    }

    // Supporting factors
    if (context.user_profile.lifecycle_stage === 'onboarding') {
      reasons.push(`New user in onboarding phase (${context.user_profile.days_since_signup} days since signup)`);
    }

    if (context.behavioral_scores.support_tickets > 0) {
      reasons.push(`User has active support needs (${context.behavioral_scores.support_tickets} tickets)`);
    }

    // Strategy selection
    reasons.push(`Selected ${strategy.intervention_type} strategy with ${strategy.urgency} urgency`);

    // ML/Heuristic prediction
    reasons.push(`Predicted best approach: ${mlPrediction.action.name} (${(mlPrediction.confidence * 100).toFixed(0)}% confidence)`);

    return reasons.join('; ');
  }

  // Helper methods

  private createNoInterventionDecision(
    decisionId: string,
    reason: string,
    score: number
  ): InterventionDecision {
    return {
      should_intervene: false,
      urgency: UrgencyLevel.NONE,
      confidence_score: score,
      reasoning: reason,
      decision_id: decisionId,
      created_at: new Date()
    };
  }

  private calculateRecentOpenRate(messageHistory: any[]): number {
    const recentMessages = messageHistory.filter(msg => {
      const daysSince = (Date.now() - new Date(msg.sent_at).getTime()) / (1000 * 60 * 60 * 24);
      return daysSince <= 30;
    });

    if (recentMessages.length === 0) return 0.3; // Default rate

    const opened = recentMessages.filter(msg => msg.opened_at).length;
    return opened / recentMessages.length;
  }

  private calculateRecentClickRate(messageHistory: any[]): number {
    const recentMessages = messageHistory.filter(msg => {
      const daysSince = (Date.now() - new Date(msg.sent_at).getTime()) / (1000 * 60 * 60 * 24);
      return daysSince <= 30;
    });

    if (recentMessages.length === 0) return 0.1; // Default rate

    const clicked = recentMessages.filter(msg => msg.clicked_at).length;
    return clicked / recentMessages.length;
  }

  private hoursSinceLastMessage(messageHistory: any[]): number {
    if (messageHistory.length === 0) return 999;

    const lastMessage = messageHistory.reduce((latest, msg) => {
      return new Date(msg.sent_at) > new Date(latest.sent_at) ? msg : latest;
    });

    return (Date.now() - new Date(lastMessage.sent_at).getTime()) / (1000 * 60 * 60);
  }

  private determinePersonalizationLevel(context: DecisionContext): 'low' | 'medium' | 'high' {
    const engagementScore = context.behavioral_scores.engagement_score || 0;
    const churnRisk = context.behavioral_scores.churn_risk || 0;

    if (churnRisk > 0.7 || engagementScore > 0.8) {
      return 'high'; // High-value or at-risk users get maximum personalization
    } else if (engagementScore > 0.5 || context.user_profile.lifecycle_stage === 'onboarding') {
      return 'medium';
    } else {
      return 'low';
    }
  }

  private determineTone(context: DecisionContext, strategy: Strategy): string {
    if (strategy.urgency === UrgencyLevel.CRITICAL) {
      return 'urgent but supportive';
    } else if (strategy.intervention_type === InterventionType.CELEBRATION) {
      return 'enthusiastic and congratulatory';
    } else if (strategy.intervention_type === InterventionType.SUPPORT) {
      return 'helpful and reassuring';
    } else if (context.user_profile.lifecycle_stage === 'onboarding') {
      return 'welcoming and encouraging';
    } else {
      return 'friendly and professional';
    }
  }

  private generateKeyMessages(
    context: DecisionContext,
    strategy: Strategy,
    mlPrediction: MLPrediction
  ): string[] {
    const messages = [];

    switch (strategy.intervention_type) {
      case InterventionType.RETENTION:
        messages.push('We value you as a customer');
        messages.push('Here\'s what you\'re missing');
        break;
      case InterventionType.MONETIZATION:
        messages.push('Unlock your full potential');
        messages.push('Join successful users');
        break;
      case InterventionType.ONBOARDING:
        messages.push('Welcome to our community');
        messages.push('Let\'s get you started');
        break;
      case InterventionType.SUPPORT:
        messages.push('We\'re here to help');
        messages.push('Quick solution available');
        break;
      case InterventionType.CELEBRATION:
        messages.push('Congratulations on your achievement');
        messages.push('You\'re making great progress');
        break;
    }

    return messages;
  }

  private determineCTAType(strategy: Strategy, mlPrediction: MLPrediction): string {
    if (strategy.intervention_type === InterventionType.MONETIZATION) {
      return 'upgrade';
    } else if (strategy.intervention_type === InterventionType.SUPPORT) {
      return 'get_help';
    } else if (strategy.intervention_type === InterventionType.ONBOARDING) {
      return 'continue_setup';
    } else {
      return 'explore';
    }
  }

  private generateUrgencyIndicators(strategy: Strategy): string[] {
    const indicators = [];

    if (strategy.urgency === UrgencyLevel.CRITICAL) {
      indicators.push('Immediate action required');
      indicators.push('Time-sensitive');
    } else if (strategy.urgency === UrgencyLevel.HIGH) {
      indicators.push('Don\'t miss out');
      indicators.push('Limited time');
    }

    return indicators;
  }

  private shouldIncludeSocialProof(context: DecisionContext, mlPrediction: MLPrediction): boolean {
    return (
      mlPrediction.action.type === 'social_proof' ||
      context.behavioral_scores.engagement_score > 0.6 ||
      context.user_profile.lifecycle_stage === 'onboarding'
    );
  }

  private generateSpecialOffer(context: DecisionContext, strategy: Strategy): any {
    if (strategy.intervention_type === InterventionType.MONETIZATION) {
      return {
        type: 'discount',
        value: '20% off',
        expiry_date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days
        conditions: ['First-time upgrade only']
      };
    }
    return undefined;
  }

  private async initializeMLModel(): Promise<void> {
    try {
      // Placeholder for ML model initialization
      logger.info('ML model initialization skipped (using heuristics)');
    } catch (error) {
      logger.error('ML model initialization failed', {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async mlPredict(features: Record<string, any>, actions: Action[]): Promise<MLPrediction> {
    // Placeholder for actual ML prediction
    // For now, fall back to heuristic prediction
    return this.heuristicPrediction(features, actions);
  }

  /**
   * Health check for the decision engine
   */
  public async healthCheck(): Promise<{ healthy: boolean; details: Record<string, any> }> {
    const details: Record<string, any> = {};
    let healthy = true;

    try {
      // Check component health
      details.components = {
        strategySelector: !!this.strategySelector,
        channelSelector: !!this.channelSelector,
        timingOptimizer: !!this.timingOptimizer,
        fatigueManager: !!this.fatigueManager
      };

      details.config = {
        mlEnabled: this.config.ml_model_enabled,
        interventionThreshold: this.config.intervention_threshold
      };

      // Check if all components are initialized
      if (!Object.values(details.components).every(Boolean)) {
        healthy = false;
        details.error = 'Some components not initialized';
      }

    } catch (error) {
      healthy = false;
      details.error = error instanceof Error ? error.message : String(error);
    }

    return { healthy, details };
  }
} 