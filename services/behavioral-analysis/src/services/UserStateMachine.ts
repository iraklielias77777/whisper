import { UserLifecycleStage, UserEvent, BehavioralMetrics } from '../types';
import { logger } from '@userwhisperer/shared';

export interface StateTransitionRule {
  from: UserLifecycleStage;
  to: UserLifecycleStage;
  condition: (metrics: BehavioralMetrics, events: UserEvent[]) => boolean;
  confidence: number;
  description: string;
}

export interface StateTransitionResult {
  shouldTransition: boolean;
  newState?: UserLifecycleStage;
  confidence: number;
  reason: string;
  triggerEvents: string[];
}

export class UserStateMachine {
  private transitionRules: StateTransitionRule[];

  constructor() {
    this.transitionRules = this.initializeTransitionRules();
  }

  /**
   * Determine if user should transition to a new state
   */
  public determineStateTransition(
    currentState: UserLifecycleStage,
    metrics: BehavioralMetrics,
    recentEvents: UserEvent[]
  ): StateTransitionResult {
    // Get applicable transition rules
    const applicableRules = this.transitionRules.filter(
      rule => rule.from === currentState
    );

    // Evaluate each rule
    for (const rule of applicableRules) {
      try {
        if (rule.condition(metrics, recentEvents)) {
          const triggerEvents = this.extractTriggerEvents(recentEvents, rule.to);
          
          logger.info('State transition detected', {
            userId: metrics.user_id,
            fromState: currentState,
            toState: rule.to,
            confidence: rule.confidence,
            reason: rule.description,
            triggerEvents
          });

          return {
            shouldTransition: true,
            newState: rule.to,
            confidence: rule.confidence,
            reason: rule.description,
            triggerEvents
          };
        }
      } catch (error) {
        logger.error('Error evaluating transition rule', {
          rule: rule.description,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    return {
      shouldTransition: false,
      confidence: 0,
      reason: 'No transition conditions met',
      triggerEvents: []
    };
  }

  /**
   * Get all possible transitions from current state
   */
  public getPossibleTransitions(currentState: UserLifecycleStage): UserLifecycleStage[] {
    return this.transitionRules
      .filter(rule => rule.from === currentState)
      .map(rule => rule.to);
  }

  /**
   * Initialize all state transition rules
   */
  private initializeTransitionRules(): StateTransitionRule[] {
    return [
      // NEW -> ONBOARDING
      {
        from: UserLifecycleStage.NEW,
        to: UserLifecycleStage.ONBOARDING,
        condition: (metrics, events) => {
          return (
            metrics.days_since_signup <= 1 &&
            metrics.total_events > 0 &&
            this.hasRecentActivity(events, 24) // Activity in last 24 hours
          );
        },
        confidence: 0.9,
        description: 'User has completed signup and started using the product'
      },

      // NEW -> CHURNED (early churn)
      {
        from: UserLifecycleStage.NEW,
        to: UserLifecycleStage.CHURNED,
        condition: (metrics, events) => {
          return (
            metrics.days_since_signup >= 7 &&
            metrics.total_events <= 1 &&
            metrics.days_since_last_active >= 7
          );
        },
        confidence: 0.8,
        description: 'User churned without meaningful engagement'
      },

      // ONBOARDING -> ACTIVATED
      {
        from: UserLifecycleStage.ONBOARDING,
        to: UserLifecycleStage.ACTIVATED,
        condition: (metrics, events) => {
          return (
            metrics.feature_adoption_rate >= 0.3 &&
            metrics.total_sessions >= 3 &&
            metrics.days_since_signup <= 7 &&
            this.hasActivationEvents(events)
          );
        },
        confidence: 0.85,
        description: 'User has completed key activation actions'
      },

      // ONBOARDING -> AT_RISK
      {
        from: UserLifecycleStage.ONBOARDING,
        to: UserLifecycleStage.AT_RISK,
        condition: (metrics, events) => {
          return (
            metrics.days_since_last_active >= 3 &&
            metrics.engagement_score < 0.3 &&
            metrics.support_tickets > 0
          );
        },
        confidence: 0.7,
        description: 'Onboarding user showing signs of struggle'
      },

      // ONBOARDING -> CHURNED
      {
        from: UserLifecycleStage.ONBOARDING,
        to: UserLifecycleStage.CHURNED,
        condition: (metrics, events) => {
          return (
            metrics.days_since_signup >= 14 &&
            metrics.feature_adoption_rate < 0.2 &&
            metrics.days_since_last_active >= 7
          );
        },
        confidence: 0.75,
        description: 'User failed to activate after onboarding period'
      },

      // ACTIVATED -> ENGAGED
      {
        from: UserLifecycleStage.ACTIVATED,
        to: UserLifecycleStage.ENGAGED,
        condition: (metrics, events) => {
          return (
            metrics.weekly_active_days >= 3 &&
            metrics.feature_adoption_rate >= 0.5 &&
            metrics.session_frequency >= 0.5 &&
            metrics.engagement_score >= 0.6
          );
        },
        confidence: 0.8,
        description: 'User showing consistent engagement patterns'
      },

      // ACTIVATED -> AT_RISK
      {
        from: UserLifecycleStage.ACTIVATED,
        to: UserLifecycleStage.AT_RISK,
        condition: (metrics, events) => {
          return this.isAtRisk(metrics, events);
        },
        confidence: 0.7,
        description: 'Activated user showing decline in engagement'
      },

      // ACTIVATED -> DORMANT
      {
        from: UserLifecycleStage.ACTIVATED,
        to: UserLifecycleStage.DORMANT,
        condition: (metrics, events) => {
          return metrics.days_since_last_active >= 14;
        },
        confidence: 0.9,
        description: 'User has been inactive for extended period'
      },

      // ENGAGED -> POWER_USER
      {
        from: UserLifecycleStage.ENGAGED,
        to: UserLifecycleStage.POWER_USER,
        condition: (metrics, events) => {
          return (
            metrics.daily_active_rate >= 0.8 &&
            metrics.feature_adoption_rate >= 0.7 &&
            metrics.power_features_used >= 2 &&
            this.hasContentCreationActivity(events)
          );
        },
        confidence: 0.9,
        description: 'User exhibits power user behavior patterns'
      },

      // ENGAGED -> AT_RISK
      {
        from: UserLifecycleStage.ENGAGED,
        to: UserLifecycleStage.AT_RISK,
        condition: (metrics, events) => {
          return this.isAtRisk(metrics, events);
        },
        confidence: 0.75,
        description: 'Engaged user showing warning signs'
      },

      // ENGAGED -> DORMANT
      {
        from: UserLifecycleStage.ENGAGED,
        to: UserLifecycleStage.DORMANT,
        condition: (metrics, events) => {
          return metrics.days_since_last_active >= 14;
        },
        confidence: 0.85,
        description: 'Previously engaged user became inactive'
      },

      // POWER_USER -> ENGAGED (regression)
      {
        from: UserLifecycleStage.POWER_USER,
        to: UserLifecycleStage.ENGAGED,
        condition: (metrics, events) => {
          return (
            metrics.daily_active_rate < 0.6 ||
            metrics.power_features_used < 2
          );
        },
        confidence: 0.7,
        description: 'Power user activity declined to engaged level'
      },

      // POWER_USER -> AT_RISK
      {
        from: UserLifecycleStage.POWER_USER,
        to: UserLifecycleStage.AT_RISK,
        condition: (metrics, events) => {
          return this.isAtRisk(metrics, events);
        },
        confidence: 0.8,
        description: 'Power user showing risk indicators'
      },

      // AT_RISK -> ENGAGED (recovery)
      {
        from: UserLifecycleStage.AT_RISK,
        to: UserLifecycleStage.ENGAGED,
        condition: (metrics, events) => {
          return (
            metrics.engagement_score >= 0.6 &&
            metrics.weekly_active_days >= 3 &&
            this.hasRecentActivity(events, 48)
          );
        },
        confidence: 0.8,
        description: 'At-risk user recovered engagement'
      },

      // AT_RISK -> REACTIVATED
      {
        from: UserLifecycleStage.AT_RISK,
        to: UserLifecycleStage.REACTIVATED,
        condition: (metrics, events) => {
          return (
            metrics.days_since_last_active <= 1 &&
            this.hasRecentActivity(events, 24) &&
            metrics.engagement_score > 0.4
          );
        },
        confidence: 0.75,
        description: 'At-risk user returned with meaningful activity'
      },

      // AT_RISK -> DORMANT
      {
        from: UserLifecycleStage.AT_RISK,
        to: UserLifecycleStage.DORMANT,
        condition: (metrics, events) => {
          return metrics.days_since_last_active >= 14;
        },
        confidence: 0.9,
        description: 'At-risk user became dormant'
      },

      // AT_RISK -> CHURNED
      {
        from: UserLifecycleStage.AT_RISK,
        to: UserLifecycleStage.CHURNED,
        condition: (metrics, events) => {
          return (
            metrics.days_since_last_active >= 30 ||
            this.hasCancellationSignals(events) ||
            metrics.cancellation_signals > 0
          );
        },
        confidence: 0.85,
        description: 'At-risk user churned'
      },

      // DORMANT -> REACTIVATED
      {
        from: UserLifecycleStage.DORMANT,
        to: UserLifecycleStage.REACTIVATED,
        condition: (metrics, events) => {
          return (
            this.hasRecentActivity(events, 24) &&
            metrics.total_events > 0
          );
        },
        confidence: 0.9,
        description: 'Dormant user returned with activity'
      },

      // DORMANT -> CHURNED
      {
        from: UserLifecycleStage.DORMANT,
        to: UserLifecycleStage.CHURNED,
        condition: (metrics, events) => {
          return (
            metrics.days_since_last_active >= 60 ||
            this.hasCancellationSignals(events)
          );
        },
        confidence: 0.8,
        description: 'Dormant user transitioned to churned'
      },

      // CHURNED -> REACTIVATED
      {
        from: UserLifecycleStage.CHURNED,
        to: UserLifecycleStage.REACTIVATED,
        condition: (metrics, events) => {
          return (
            this.hasRecentActivity(events, 24) &&
            events.length > 1 // More than just a single login
          );
        },
        confidence: 0.85,
        description: 'Churned user returned with meaningful activity'
      },

      // REACTIVATED -> ENGAGED
      {
        from: UserLifecycleStage.REACTIVATED,
        to: UserLifecycleStage.ENGAGED,
        condition: (metrics, events) => {
          return (
            metrics.weekly_active_days >= 3 &&
            metrics.engagement_score >= 0.5 &&
            metrics.feature_adoption_rate >= 0.4
          );
        },
        confidence: 0.8,
        description: 'Reactivated user achieved sustained engagement'
      },

      // REACTIVATED -> AT_RISK
      {
        from: UserLifecycleStage.REACTIVATED,
        to: UserLifecycleStage.AT_RISK,
        condition: (metrics, events) => {
          return this.isAtRisk(metrics, events);
        },
        confidence: 0.7,
        description: 'Reactivated user showing risk signs again'
      },

      // REACTIVATED -> CHURNED
      {
        from: UserLifecycleStage.REACTIVATED,
        to: UserLifecycleStage.CHURNED,
        condition: (metrics, events) => {
          return (
            metrics.days_since_last_active >= 14 &&
            metrics.engagement_score < 0.3
          );
        },
        confidence: 0.75,
        description: 'Reactivated user churned again'
      }
    ];
  }

  /**
   * Check if user is showing at-risk indicators
   */
  private isAtRisk(metrics: BehavioralMetrics, events: UserEvent[]): boolean {
    return (
      metrics.engagement_trend <= -0.3 || // 30% decline in engagement
      metrics.days_since_last_active >= 7 ||
      metrics.support_tickets >= 3 ||
      metrics.error_rate > 0.1 ||
      metrics.payment_failures > 0 ||
      this.hasCancellationSignals(events)
    );
  }

  /**
   * Check if user has recent activity within specified hours
   */
  private hasRecentActivity(events: UserEvent[], withinHours: number): boolean {
    const cutoffTime = new Date(Date.now() - withinHours * 60 * 60 * 1000);
    return events.some(event => new Date(event.timestamp) > cutoffTime);
  }

  /**
   * Check if user has activation events
   */
  private hasActivationEvents(events: UserEvent[]): boolean {
    const activationEvents = [
      'profile_completed',
      'first_action_completed',
      'tutorial_completed',
      'feature_used',
      'content_created'
    ];

    return events.some(event => 
      activationEvents.some(activationEvent => 
        event.event_type.includes(activationEvent)
      )
    );
  }

  /**
   * Check if user has content creation activity
   */
  private hasContentCreationActivity(events: UserEvent[]): boolean {
    const contentEvents = [
      'content_created',
      'document_created',
      'project_created',
      'post_published',
      'file_uploaded'
    ];

    const contentEventsCount = events.filter(event =>
      contentEvents.some(contentEvent =>
        event.event_type.includes(contentEvent)
      )
    ).length;

    return contentEventsCount >= 10; // At least 10 content creation events
  }

  /**
   * Check if user has cancellation signals
   */
  private hasCancellationSignals(events: UserEvent[]): boolean {
    const cancellationEvents = [
      'subscription_cancel_clicked',
      'cancellation_reason_viewed',
      'downgrade_clicked',
      'account_deletion_requested',
      'subscription_cancelled'
    ];

    return events.some(event =>
      cancellationEvents.includes(event.event_type)
    );
  }

  /**
   * Extract trigger events that influenced the transition
   */
  private extractTriggerEvents(events: UserEvent[], toState: UserLifecycleStage): string[] {
    const recentEvents = events
      .filter(event => {
        const eventTime = new Date(event.timestamp);
        const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
        return eventTime > oneDayAgo;
      })
      .map(event => event.event_type);

    // Return unique event types
    return [...new Set(recentEvents)];
  }

  /**
   * Get transition confidence based on metrics quality
   */
  public getTransitionConfidence(metrics: BehavioralMetrics): number {
    let confidence = 1.0;

    // Reduce confidence if insufficient data
    if (metrics.total_events < 10) confidence *= 0.7;
    if (metrics.days_since_signup < 7) confidence *= 0.8;
    if (metrics.total_sessions < 3) confidence *= 0.9;

    return Math.max(confidence, 0.1); // Minimum 10% confidence
  }
} 