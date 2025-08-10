import { 
  DecisionContext, 
  Strategy, 
  InterventionType, 
  UrgencyLevel 
} from '../types';
import { logger } from '@userwhisperer/shared';

export class StrategySelector {
  /**
   * Select the most appropriate intervention strategy based on context
   */
  public selectStrategy(context: DecisionContext): Strategy {
    logger.debug('Selecting intervention strategy', {
      userId: context.user_id,
      triggerEvent: context.trigger_event.event_type,
      lifecycleStage: context.user_profile.lifecycle_stage,
      churnRisk: context.behavioral_scores.churn_risk
    });

    // Priority order for strategy selection
    const strategies = [
      this.checkCriticalIntervention(context),
      this.checkRetentionIntervention(context),
      this.checkMonetizationIntervention(context),
      this.checkOnboardingIntervention(context),
      this.checkReactivationIntervention(context),
      this.checkSupportIntervention(context),
      this.checkEducationIntervention(context),
      this.checkCelebrationIntervention(context)
    ];

    // Return first applicable strategy
    for (const strategy of strategies) {
      if (strategy) {
        logger.info('Strategy selected', {
          userId: context.user_id,
          interventionType: strategy.intervention_type,
          urgency: strategy.urgency,
          goal: strategy.goal
        });
        return strategy;
      }
    }

    // Default strategy
    return this.getDefaultStrategy(context);
  }

  /**
   * Check for critical interventions that require immediate attention
   */
  private checkCriticalIntervention(context: DecisionContext): Strategy | null {
    const eventType = context.trigger_event.event_type;

    // Payment failure - highest priority
    if (eventType === 'payment_failed') {
      return {
        intervention_type: InterventionType.SUPPORT,
        urgency: UrgencyLevel.CRITICAL,
        goal: 'recover_payment',
        approach: 'payment_recovery',
        max_attempts: 3,
        success_metrics: ['payment_succeeded', 'payment_method_updated'],
        content_hints: {
          focus: 'payment_resolution',
          tone: 'helpful_urgent',
          include_support_contact: true,
          offer_assistance: true
        }
      };
    }

    // Subscription cancellation
    if (eventType === 'subscription_cancelled' || eventType === 'subscription_cancel_clicked') {
      return {
        intervention_type: InterventionType.RETENTION,
        urgency: UrgencyLevel.CRITICAL,
        goal: 'prevent_churn',
        approach: 'win_back',
        max_attempts: 2,
        success_metrics: ['subscription_reactivated', 'cancellation_reverted'],
        content_hints: {
          focus: 'value_retention',
          tone: 'understanding_urgent',
          include_special_offer: true,
          personal_touch: true
        }
      };
    }

    // Critical errors
    if (eventType === 'error_critical' || eventType === 'data_loss_detected') {
      return {
        intervention_type: InterventionType.SUPPORT,
        urgency: UrgencyLevel.CRITICAL,
        goal: 'resolve_critical_issue',
        approach: 'immediate_support',
        max_attempts: 1,
        success_metrics: ['issue_resolved', 'support_contact_made'],
        content_hints: {
          focus: 'immediate_resolution',
          tone: 'supportive_urgent',
          escalate_to_human: true,
          priority_support: true
        }
      };
    }

    // Trial ending soon
    if (eventType === 'trial_ending' || eventType === 'trial_expires_tomorrow') {
      const daysUntilExpiry = this.getDaysUntilTrialExpiry(context);
      if (daysUntilExpiry <= 1) {
        return {
          intervention_type: InterventionType.MONETIZATION,
          urgency: UrgencyLevel.CRITICAL,
          goal: 'convert_trial',
          approach: 'urgent_conversion',
          max_attempts: 2,
          success_metrics: ['subscription_started', 'trial_extended'],
          content_hints: {
            focus: 'immediate_conversion',
            tone: 'urgent_but_friendly',
            time_sensitivity: true,
            special_offer: true
          }
        };
      }
    }

    return null;
  }

  /**
   * Check for retention interventions
   */
  private checkRetentionIntervention(context: DecisionContext): Strategy | null {
    const churnRisk = context.behavioral_scores.churn_risk || 0;
    const daysSinceLastActive = context.behavioral_scores.days_since_last_active || 0;
    const engagementScore = context.behavioral_scores.engagement_score || 0;

    // High churn risk
    if (churnRisk > 0.7) {
      return {
        intervention_type: InterventionType.RETENTION,
        urgency: UrgencyLevel.HIGH,
        goal: 'prevent_churn',
        approach: 'proactive_retention',
        max_attempts: 3,
        success_metrics: ['engagement_increased', 'session_started', 'feature_used'],
        content_hints: {
          focus: 'value_demonstration',
          tone: 'caring_concerned',
          personalization: 'high',
          include_success_metrics: true
        }
      };
    }

    // Medium churn risk
    if (churnRisk > 0.5 || daysSinceLastActive >= 7) {
      return {
        intervention_type: InterventionType.RETENTION,
        urgency: UrgencyLevel.MEDIUM,
        goal: 're_engage_user',
        approach: 'gentle_reminder',
        max_attempts: 2,
        success_metrics: ['session_started', 'feature_used', 'content_viewed'],
        content_hints: {
          focus: 'gentle_reengagement',
          tone: 'friendly_inviting',
          show_missed_updates: true,
          low_pressure: true
        }
      };
    }

    // Declining engagement
    if (engagementScore < 0.3 && context.user_profile.lifecycle_stage !== 'new') {
      return {
        intervention_type: InterventionType.RETENTION,
        urgency: UrgencyLevel.MEDIUM,
        goal: 'boost_engagement',
        approach: 'engagement_boost',
        max_attempts: 2,
        success_metrics: ['engagement_score_improved', 'feature_explored'],
        content_hints: {
          focus: 'feature_discovery',
          tone: 'encouraging',
          highlight_unused_features: true,
          educational: true
        }
      };
    }

    return null;
  }

  /**
   * Check for monetization opportunities
   */
  private checkMonetizationIntervention(context: DecisionContext): Strategy | null {
    const upgradeProbability = context.behavioral_scores.upgrade_probability || 0;
    const featureAdoptionRate = context.behavioral_scores.feature_adoption_rate || 0;
    const hittingLimits = context.behavioral_scores.hitting_limits || false;
    const eventType = context.trigger_event.event_type;

    // User hit usage limits
    if (eventType === 'limit_reached' || hittingLimits) {
      return {
        intervention_type: InterventionType.MONETIZATION,
        urgency: UrgencyLevel.HIGH,
        goal: 'convert_due_to_limits',
        approach: 'limit_based_conversion',
        max_attempts: 2,
        success_metrics: ['upgrade_started', 'pricing_viewed', 'trial_started'],
        content_hints: {
          focus: 'limit_relief',
          tone: 'understanding_helpful',
          show_upgrade_benefits: true,
          immediate_value: true
        }
      };
    }

    // High upgrade probability
    if (upgradeProbability > 0.7) {
      return {
        intervention_type: InterventionType.MONETIZATION,
        urgency: UrgencyLevel.MEDIUM,
        goal: 'drive_upgrade',
        approach: 'direct_upgrade_pitch',
        max_attempts: 2,
        success_metrics: ['upgrade_started', 'pricing_viewed', 'trial_started'],
        content_hints: {
          focus: 'upgrade_benefits',
          tone: 'confident_persuasive',
          social_proof: true,
          roi_calculation: true
        }
      };
    }

    // High feature adoption with pricing page views
    if (featureAdoptionRate > 0.6 && this.hasRecentPricingViews(context)) {
      return {
        intervention_type: InterventionType.MONETIZATION,
        urgency: UrgencyLevel.MEDIUM,
        goal: 'nurture_upgrade_intent',
        approach: 'educational_nurture',
        max_attempts: 2,
        success_metrics: ['upgrade_started', 'demo_requested', 'sales_contact'],
        content_hints: {
          focus: 'value_education',
          tone: 'educational_supportive',
          case_studies: true,
          feature_comparison: true
        }
      };
    }

    // Medium upgrade probability with good engagement
    if (upgradeProbability > 0.5 && context.behavioral_scores.engagement_score > 0.6) {
      return {
        intervention_type: InterventionType.MONETIZATION,
        urgency: UrgencyLevel.LOW,
        goal: 'educate_on_benefits',
        approach: 'soft_monetization',
        max_attempts: 1,
        success_metrics: ['pricing_viewed', 'upgrade_considered', 'feature_explored'],
        content_hints: {
          focus: 'soft_education',
          tone: 'friendly_informative',
          no_pressure: true,
          highlight_power_features: true
        }
      };
    }

    return null;
  }

  /**
   * Check for onboarding interventions
   */
  private checkOnboardingIntervention(context: DecisionContext): Strategy | null {
    const lifecycleStage = context.user_profile.lifecycle_stage;
    const daysSinceSignup = context.user_profile.days_since_signup;
    const featureAdoptionRate = context.behavioral_scores.feature_adoption_rate || 0;

    if (lifecycleStage === 'new' || lifecycleStage === 'onboarding') {
      // Early onboarding (first 3 days)
      if (daysSinceSignup <= 3) {
        return {
          intervention_type: InterventionType.ONBOARDING,
          urgency: UrgencyLevel.HIGH,
          goal: 'complete_setup',
          approach: 'guided_onboarding',
          max_attempts: 3,
          success_metrics: ['profile_completed', 'first_action_taken', 'tutorial_completed'],
          content_hints: {
            focus: 'setup_completion',
            tone: 'welcoming_encouraging',
            step_by_step: true,
            quick_wins: true
          }
        };
      }

      // Mid onboarding (4-7 days)
      if (daysSinceSignup <= 7 && featureAdoptionRate < 0.3) {
        return {
          intervention_type: InterventionType.ONBOARDING,
          urgency: UrgencyLevel.MEDIUM,
          goal: 'drive_activation',
          approach: 'activation_focused',
          max_attempts: 2,
          success_metrics: ['key_feature_used', 'value_realized', 'engagement_increased'],
          content_hints: {
            focus: 'value_realization',
            tone: 'helpful_guiding',
            feature_highlights: true,
            use_case_examples: true
          }
        };
      }

      // Extended onboarding (8-14 days)
      if (daysSinceSignup <= 14 && featureAdoptionRate < 0.5) {
        return {
          intervention_type: InterventionType.ONBOARDING,
          urgency: UrgencyLevel.MEDIUM,
          goal: 'prevent_early_churn',
          approach: 'retention_onboarding',
          max_attempts: 2,
          success_metrics: ['feature_adoption_improved', 'session_frequency_increased'],
          content_hints: {
            focus: 'overcoming_obstacles',
            tone: 'supportive_patient',
            address_pain_points: true,
            alternative_approaches: true
          }
        };
      }
    }

    return null;
  }

  /**
   * Check for reactivation interventions
   */
  private checkReactivationIntervention(context: DecisionContext): Strategy | null {
    const daysSinceLastActive = context.behavioral_scores.days_since_last_active || 0;
    const lifecycleStage = context.user_profile.lifecycle_stage;
    const previousEngagement = this.getPreviousEngagementLevel(context);

    // Dormant users with previous high engagement
    if (daysSinceLastActive >= 30 && previousEngagement > 0.6) {
      return {
        intervention_type: InterventionType.REACTIVATION,
        urgency: UrgencyLevel.MEDIUM,
        goal: 'reactivate_dormant_user',
        approach: 'win_back_campaign',
        max_attempts: 3,
        success_metrics: ['session_started', 'feature_used', 'engagement_resumed'],
        content_hints: {
          focus: 'what_they_missed',
          tone: 'nostalgic_inviting',
          show_improvements: true,
          special_welcome_back: true
        }
      };
    }

    // Recently dormant users
    if (daysSinceLastActive >= 14 && daysSinceLastActive < 30) {
      return {
        intervention_type: InterventionType.REACTIVATION,
        urgency: UrgencyLevel.LOW,
        goal: 'gentle_reactivation',
        approach: 'soft_reminder',
        max_attempts: 2,
        success_metrics: ['session_started', 'notification_enabled'],
        content_hints: {
          focus: 'gentle_reminder',
          tone: 'friendly_casual',
          low_commitment: true,
          easy_return: true
        }
      };
    }

    return null;
  }

  /**
   * Check for support interventions
   */
  private checkSupportIntervention(context: DecisionContext): Strategy | null {
    const supportTickets = context.behavioral_scores.support_tickets || 0;
    const errorRate = context.behavioral_scores.error_rate || 0;
    const eventType = context.trigger_event.event_type;

    // Multiple support tickets
    if (supportTickets >= 3) {
      return {
        intervention_type: InterventionType.SUPPORT,
        urgency: UrgencyLevel.HIGH,
        goal: 'resolve_support_issues',
        approach: 'escalated_support',
        max_attempts: 2,
        success_metrics: ['issue_resolved', 'satisfaction_improved'],
        content_hints: {
          focus: 'issue_resolution',
          tone: 'apologetic_helpful',
          escalate_priority: true,
          personal_attention: true
        }
      };
    }

    // High error rate
    if (errorRate > 0.1) {
      return {
        intervention_type: InterventionType.SUPPORT,
        urgency: UrgencyLevel.MEDIUM,
        goal: 'reduce_errors',
        approach: 'proactive_support',
        max_attempts: 2,
        success_metrics: ['error_rate_reduced', 'help_content_viewed'],
        content_hints: {
          focus: 'error_prevention',
          tone: 'helpful_educational',
          troubleshooting_guide: true,
          prevent_future_issues: true
        }
      };
    }

    // Support-related events
    if (eventType === 'help_requested' || eventType === 'error_encountered') {
      return {
        intervention_type: InterventionType.SUPPORT,
        urgency: UrgencyLevel.MEDIUM,
        goal: 'provide_immediate_help',
        approach: 'reactive_support',
        max_attempts: 1,
        success_metrics: ['help_content_accessed', 'issue_self_resolved'],
        content_hints: {
          focus: 'immediate_assistance',
          tone: 'responsive_helpful',
          contextual_help: true,
          quick_resolution: true
        }
      };
    }

    return null;
  }

  /**
   * Check for education interventions
   */
  private checkEducationIntervention(context: DecisionContext): Strategy | null {
    const featureAdoptionRate = context.behavioral_scores.feature_adoption_rate || 0;
    const engagementScore = context.behavioral_scores.engagement_score || 0;

    // Low feature adoption with good engagement
    if (featureAdoptionRate < 0.4 && engagementScore > 0.5) {
      return {
        intervention_type: InterventionType.EDUCATION,
        urgency: UrgencyLevel.LOW,
        goal: 'increase_feature_adoption',
        approach: 'educational_content',
        max_attempts: 2,
        success_metrics: ['feature_adoption_increased', 'tutorial_completed'],
        content_hints: {
          focus: 'feature_education',
          tone: 'educational_encouraging',
          step_by_step_guides: true,
          practical_examples: true
        }
      };
    }

    return null;
  }

  /**
   * Check for celebration interventions
   */
  private checkCelebrationIntervention(context: DecisionContext): Strategy | null {
    const eventType = context.trigger_event.event_type;
    const daysSinceSignup = context.user_profile.days_since_signup;
    const engagementScore = context.behavioral_scores.engagement_score || 0;

    // Achievement events
    if (eventType === 'milestone_reached' || eventType === 'goal_completed') {
      return {
        intervention_type: InterventionType.CELEBRATION,
        urgency: UrgencyLevel.LOW,
        goal: 'celebrate_achievement',
        approach: 'achievement_recognition',
        max_attempts: 1,
        success_metrics: ['achievement_shared', 'continued_engagement'],
        content_hints: {
          focus: 'celebration',
          tone: 'congratulatory_enthusiastic',
          achievement_details: true,
          encourage_sharing: true
        }
      };
    }

    // Upgrade completion
    if (eventType === 'subscription_upgraded') {
      return {
        intervention_type: InterventionType.CELEBRATION,
        urgency: UrgencyLevel.LOW,
        goal: 'welcome_premium_user',
        approach: 'premium_welcome',
        max_attempts: 1,
        success_metrics: ['premium_feature_used', 'satisfaction_confirmed'],
        content_hints: {
          focus: 'premium_benefits',
          tone: 'welcoming_exclusive',
          highlight_new_features: true,
          vip_treatment: true
        }
      };
    }

    // First week anniversary
    if (daysSinceSignup === 7 && engagementScore > 0.5) {
      return {
        intervention_type: InterventionType.CELEBRATION,
        urgency: UrgencyLevel.LOW,
        goal: 'first_week_celebration',
        approach: 'milestone_celebration',
        max_attempts: 1,
        success_metrics: ['continued_engagement', 'feature_exploration'],
        content_hints: {
          focus: 'progress_celebration',
          tone: 'proud_encouraging',
          show_progress: true,
          future_potential: true
        }
      };
    }

    return null;
  }

  /**
   * Get default strategy when no specific strategy applies
   */
  private getDefaultStrategy(context: DecisionContext): Strategy {
    return {
      intervention_type: InterventionType.RETENTION,
      urgency: UrgencyLevel.LOW,
      goal: 'maintain_engagement',
      approach: 'general_touchpoint',
      max_attempts: 1,
      success_metrics: ['session_started', 'content_viewed'],
      content_hints: {
        focus: 'general_value',
        tone: 'friendly_casual',
        low_pressure: true,
        value_reminder: true
      }
    };
  }

  // Helper methods

  private getDaysUntilTrialExpiry(context: DecisionContext): number {
    // This would calculate based on trial start date and duration
    // For now, return a mock value
    return 1;
  }

  private hasRecentPricingViews(context: DecisionContext): boolean {
    // Check if user has viewed pricing in last 7 days
    const pricingEvents = context.message_history.filter(msg => 
      msg.metadata.page_viewed === 'pricing' || 
      msg.metadata.event_type === 'pricing_viewed'
    );
    
    return pricingEvents.some(event => {
      const daysSince = (Date.now() - event.sent_at.getTime()) / (1000 * 60 * 60 * 24);
      return daysSince <= 7;
    });
  }

  private getPreviousEngagementLevel(context: DecisionContext): number {
    // This would analyze historical engagement data
    // For now, return a mock value based on user profile
    if (context.user_profile.lifecycle_stage === 'churned' || 
        context.user_profile.lifecycle_stage === 'dormant') {
      return 0.7; // Assume they were previously engaged
    }
    return 0.3;
  }
} 