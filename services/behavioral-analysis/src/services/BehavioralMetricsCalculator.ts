import { BehavioralMetrics, UserEvent, SessionMetrics } from '../types';
import { logger } from '@userwhisperer/shared';
import { differenceInDays, differenceInHours, parseISO } from 'date-fns';

export interface EventDataFrame {
  events: UserEvent[];
  startDate: Date;
  endDate: Date;
  totalEvents: number;
  uniqueEventTypes: number;
}

export class BehavioralMetricsCalculator {
  private readonly SESSION_BOUNDARY_MINUTES = 30;
  private readonly TOTAL_FEATURES = 50; // This would come from config
  private readonly POWER_FEATURES = ['advanced_export', 'bulk_edit', 'api_access', 'automation'];

  /**
   * Calculate all behavioral metrics for a user
   */
  public async calculateAllMetrics(
    userId: string,
    events: UserEvent[],
    messageHistory?: any[]
  ): Promise<BehavioralMetrics> {
    if (!events || events.length === 0) {
      return this.createEmptyMetrics(userId);
    }

    // Sort events by timestamp
    const sortedEvents = events.sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    const dataFrame = this.createEventDataFrame(sortedEvents);
    const now = new Date();

    logger.debug('Calculating metrics for user', {
      userId,
      eventsCount: events.length,
      dateRange: `${dataFrame.startDate.toISOString()} - ${dataFrame.endDate.toISOString()}`
    });

    // Calculate all metric categories
    const engagementMetrics = this.calculateEngagementMetrics(dataFrame, now);
    const sessionMetrics = this.calculateSessionMetrics(dataFrame);
    const featureMetrics = this.calculateFeatureMetrics(dataFrame);
    const monetizationMetrics = this.calculateMonetizationMetrics(dataFrame, now);
    const riskIndicators = this.calculateRiskIndicators(dataFrame, now);
    const communicationMetrics = this.calculateCommunicationMetrics(messageHistory || []);

    // Calculate derived metrics
    const derivedMetrics = this.calculateDerivedMetrics({
      ...engagementMetrics,
      ...sessionMetrics,
      ...featureMetrics,
      ...monetizationMetrics,
      ...riskIndicators,
      ...communicationMetrics
    }, dataFrame, now);

    const allMetrics: BehavioralMetrics = {
      user_id: userId,
      calculated_at: now,
      // Engagement metrics with defaults
      engagement_score: engagementMetrics.engagement_score || 0,
      daily_active_rate: engagementMetrics.daily_active_rate || 0,
      weekly_active_days: engagementMetrics.weekly_active_days || 0,
      engagement_trend: engagementMetrics.engagement_trend || 0,
      total_events: engagementMetrics.total_events || 0,
      unique_event_types: engagementMetrics.unique_event_types || 0,
      days_since_last_active: engagementMetrics.days_since_last_active || 0,
      
      // Session metrics with defaults
      session_frequency: sessionMetrics.session_frequency || 0,
      avg_session_duration: sessionMetrics.avg_session_duration || 0,
      total_sessions: sessionMetrics.total_sessions || 0,
      avg_events_per_session: sessionMetrics.avg_events_per_session || 0,
      session_regularity: sessionMetrics.session_regularity || 0,
      bounce_rate: sessionMetrics.bounce_rate || 0,
      pages_per_session: sessionMetrics.pages_per_session || 0,
      
      // Feature metrics with defaults
      feature_adoption_rate: featureMetrics.feature_adoption_rate || 0,
      feature_depth: featureMetrics.feature_depth || 0,
      feature_breadth: featureMetrics.feature_breadth || 0,
      power_features_used: featureMetrics.power_features_used || 0,
      unique_features_used: featureMetrics.unique_features_used || 0,
      feature_usage_depth: featureMetrics.feature_usage_depth || 0,
      new_feature_adoption: featureMetrics.new_feature_adoption || 0,
      feature_stickiness: featureMetrics.feature_stickiness || 0,
      most_used_features: featureMetrics.most_used_features || [],
      
      // Monetization metrics with defaults
      monetization_score: monetizationMetrics.monetization_score || 0,
      pricing_page_views: monetizationMetrics.pricing_page_views || 0,
      upgrade_attempts: monetizationMetrics.upgrade_attempts || 0,
      monetization_events_count: monetizationMetrics.monetization_events_count || 0,
      days_since_last_monetization_event: monetizationMetrics.days_since_last_monetization_event || 0,
      
      // Risk indicators with defaults
      error_rate: riskIndicators.error_rate || 0,
      support_tickets: riskIndicators.support_tickets || 0,
      usage_decline: riskIndicators.usage_decline || 0,
      payment_failures: riskIndicators.payment_failures || 0,
      cancellation_signals: riskIndicators.cancellation_signals || 0,
      
      // Derived metrics with defaults
      days_since_signup: derivedMetrics.days_since_signup || 0,
      lifetime_value_score: derivedMetrics.lifetime_value_score || 0,
      upgrade_probability: derivedMetrics.upgrade_probability || 0,
      churn_risk_score: derivedMetrics.churn_risk_score || 0
    };

    logger.debug('Metrics calculated successfully', {
      userId,
      engagementScore: allMetrics.engagement_score,
      churnRiskScore: allMetrics.churn_risk_score || 'N/A',
      ltvScore: allMetrics.lifetime_value_score
    });

    return allMetrics;
  }

  /**
   * Calculate engagement metrics
   */
  private calculateEngagementMetrics(dataFrame: EventDataFrame, now: Date): Partial<BehavioralMetrics> {
    const { events } = dataFrame;
    
    if (events.length === 0) {
      return {
        engagement_score: 0,
        daily_active_rate: 0,
        weekly_active_days: 0,
        engagement_trend: 0,
        total_events: 0,
        unique_event_types: 0,
        days_since_last_active: 999
      };
    }

    // Daily active rate (last 30 days)
    const thirtyDaysAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
    const last30DaysEvents = events.filter(e => new Date(e.timestamp) > thirtyDaysAgo);
    const uniqueDaysLast30 = new Set(
      last30DaysEvents.map(e => new Date(e.timestamp).toDateString())
    ).size;
    const dailyActiveRate = uniqueDaysLast30 / 30;

    // Weekly active days
    const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const lastWeekEvents = events.filter(e => new Date(e.timestamp) > oneWeekAgo);
    const weeklyActiveDays = new Set(
      lastWeekEvents.map(e => new Date(e.timestamp).toDateString())
    ).size;

    // Engagement trend (slope of daily events)
    const engagementTrend = this.calculateEngagementTrend(events);

    // Composite engagement score
    const engagementScore = this.calculateEngagementScore(
      dailyActiveRate,
      weeklyActiveDays,
      engagementTrend,
      events.length
    );

    // Days since last active
    const lastEventTime = new Date(events[events.length - 1].timestamp);
    const daysSinceLastActive = differenceInDays(now, lastEventTime);

    return {
      engagement_score: this.roundTo3Decimals(engagementScore),
      daily_active_rate: this.roundTo3Decimals(dailyActiveRate),
      weekly_active_days: weeklyActiveDays,
      engagement_trend: this.roundTo3Decimals(engagementTrend),
      total_events: events.length,
      unique_event_types: dataFrame.uniqueEventTypes,
      days_since_last_active: daysSinceLastActive
    };
  }

  /**
   * Calculate session-based metrics
   */
  private calculateSessionMetrics(dataFrame: EventDataFrame): Partial<BehavioralMetrics> {
    const { events } = dataFrame;
    
    if (events.length === 0) {
      return {
        total_sessions: 0,
        avg_session_duration: 0,
        avg_events_per_session: 0,
        session_frequency: 0,
        session_regularity: 0
      };
    }

    const sessions = this.identifySessions(events);
    const totalSessions = sessions.length;

    if (totalSessions === 0) {
      return {
        total_sessions: 0,
        avg_session_duration: 0,
        avg_events_per_session: 0,
        session_frequency: 0,
        session_regularity: 0
      };
    }

    // Calculate session durations and event counts
    const sessionDurations: number[] = [];
    const eventsPerSession: number[] = [];

    sessions.forEach(session => {
      if (session.length > 1) {
        const startTime = new Date(session[0].timestamp);
        const endTime = new Date(session[session.length - 1].timestamp);
        const durationMinutes = differenceInHours(endTime, startTime) * 60;
        sessionDurations.push(durationMinutes);
      } else {
        sessionDurations.push(0);
      }
      eventsPerSession.push(session.length);
    });

    const avgSessionDuration = this.average(sessionDurations);
    const avgEventsPerSession = this.average(eventsPerSession);

    // Session frequency (sessions per day)
    const totalDays = Math.max(1, differenceInDays(dataFrame.endDate, dataFrame.startDate) + 1);
    const sessionFrequency = totalSessions / totalDays;

    // Session regularity (coefficient of variation of inter-session intervals)
    const sessionRegularity = this.calculateSessionRegularity(sessions);

    return {
      total_sessions: totalSessions,
      avg_session_duration: this.roundTo2Decimals(avgSessionDuration),
      avg_events_per_session: this.roundTo2Decimals(avgEventsPerSession),
      session_frequency: this.roundTo3Decimals(sessionFrequency),
      session_regularity: this.roundTo3Decimals(sessionRegularity)
    };
  }

  /**
   * Calculate feature adoption and usage metrics
   */
  private calculateFeatureMetrics(dataFrame: EventDataFrame): Partial<BehavioralMetrics> {
    const { events } = dataFrame;
    
    // Feature events (events that start with 'feature_')
    const featureEvents = events.filter(e => e.event_type.startsWith('feature_'));
    
    if (featureEvents.length === 0) {
      return {
        feature_adoption_rate: 0,
        feature_depth: 0,
        feature_breadth: 0,
        power_features_used: 0,
        unique_features_used: 0
      };
    }

    // Unique features used
    const uniqueFeatures = new Set(featureEvents.map(e => e.event_type)).size;
    const featureAdoptionRate = uniqueFeatures / this.TOTAL_FEATURES;

    // Feature depth (average usage per feature)
    const featureUsageCounts = this.countEventTypes(featureEvents);
    const featureDepth = this.average(Object.values(featureUsageCounts));

    // Feature breadth (entropy of feature usage distribution)
    const featureBreadth = this.calculateEntropy(Object.values(featureUsageCounts));

    // Power features used
    const powerFeaturesUsed = this.POWER_FEATURES.filter(feature =>
      featureEvents.some(e => e.event_type.includes(feature))
    ).length;

    return {
      feature_adoption_rate: this.roundTo3Decimals(featureAdoptionRate),
      feature_depth: this.roundTo2Decimals(featureDepth),
      feature_breadth: this.roundTo3Decimals(featureBreadth),
      power_features_used: powerFeaturesUsed,
      unique_features_used: uniqueFeatures
    };
  }

  /**
   * Calculate monetization-related metrics
   */
  private calculateMonetizationMetrics(dataFrame: EventDataFrame, now: Date): Partial<BehavioralMetrics> {
    const { events } = dataFrame;
    
    const monetizationEvents = [
      'pricing_viewed', 'upgrade_clicked', 'trial_started',
      'subscription_started', 'payment_added', 'purchase_completed'
    ];

    const monEvents = events.filter(e => monetizationEvents.includes(e.event_type));
    
    // Count specific monetization events
    const pricingViews = events.filter(e => e.event_type === 'pricing_viewed').length;
    const upgradeAttempts = events.filter(e => e.event_type === 'upgrade_clicked').length;

    // Calculate monetization score
    let monetizationScore = 0;
    if (monEvents.length > 0) {
      const weights: Record<string, number> = {
        'pricing_viewed': 0.1,
        'upgrade_clicked': 0.3,
        'trial_started': 0.5,
        'subscription_started': 1.0,
        'payment_added': 0.4,
        'purchase_completed': 0.8
      };

      monetizationEvents.forEach(eventType => {
        const count = events.filter(e => e.event_type === eventType).length;
        monetizationScore += count * (weights[eventType] || 0);
      });

      // Normalize to 0-1
      monetizationScore = Math.min(monetizationScore / 10, 1.0);
    }

    // Days since last monetization event
    const lastMonEvent = monEvents
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())[0];
    
    const daysSinceLastMonetizationEvent = lastMonEvent 
      ? differenceInDays(now, new Date(lastMonEvent.timestamp))
      : 999;

    return {
      monetization_score: this.roundTo3Decimals(monetizationScore),
      pricing_page_views: pricingViews,
      upgrade_attempts: upgradeAttempts,
      monetization_events_count: monEvents.length,
      days_since_last_monetization_event: daysSinceLastMonetizationEvent
    };
  }

  /**
   * Calculate risk and churn indicators
   */
  private calculateRiskIndicators(dataFrame: EventDataFrame, now: Date): Partial<BehavioralMetrics> {
    const { events } = dataFrame;
    
    // Error rate
    const errorEvents = events.filter(e => e.event_type.includes('error'));
    const errorRate = events.length > 0 ? errorEvents.length / events.length : 0;

    // Support tickets
    const supportEvents = events.filter(e => e.event_type === 'support_ticket_created');
    const supportTickets = supportEvents.length;

    // Usage decline
    let usageDecline = 0;
    if (events.length > 7) {
      const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      const twoWeeksAgo = new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000);
      
      const lastWeekEvents = events.filter(e => {
        const eventTime = new Date(e.timestamp);
        return eventTime > oneWeekAgo;
      });
      
      const prevWeekEvents = events.filter(e => {
        const eventTime = new Date(e.timestamp);
        return eventTime > twoWeeksAgo && eventTime <= oneWeekAgo;
      });

      if (prevWeekEvents.length > 0) {
        usageDecline = (lastWeekEvents.length - prevWeekEvents.length) / prevWeekEvents.length;
      }
    }

    // Payment failures
    const paymentFailures = events.filter(e => e.event_type === 'payment_failed').length;

    // Cancellation signals
    const cancellationEvents = [
      'subscription_cancel_clicked',
      'cancellation_reason_viewed',
      'downgrade_clicked'
    ];
    const cancellationSignals = events.filter(e => 
      cancellationEvents.includes(e.event_type)
    ).length;

    return {
      error_rate: this.roundTo3Decimals(errorRate),
      support_tickets: supportTickets,
      usage_decline: this.roundTo3Decimals(usageDecline),
      payment_failures: paymentFailures,
      cancellation_signals: cancellationSignals
    };
  }

  /**
   * Calculate communication metrics
   */
  private calculateCommunicationMetrics(messageHistory: any[]): Partial<BehavioralMetrics> {
    // This would analyze message history if provided
    // For now, return empty metrics
    return {};
  }

  /**
   * Calculate derived metrics
   */
  private calculateDerivedMetrics(
    baseMetrics: Partial<BehavioralMetrics>,
    dataFrame: EventDataFrame,
    now: Date
  ): Partial<BehavioralMetrics> {
    const daysSinceSignup = dataFrame.events.length > 0
      ? differenceInDays(now, dataFrame.startDate)
      : 0;

    // Simple LTV score calculation
    const lifetimeValueScore = this.calculateLTVScore(baseMetrics, daysSinceSignup);

    // Upgrade probability
    const upgradeProbability = this.calculateUpgradeProbability(baseMetrics);

    return {
      days_since_signup: daysSinceSignup,
      lifetime_value_score: this.roundTo2Decimals(lifetimeValueScore),
      upgrade_probability: this.roundTo3Decimals(upgradeProbability)
    };
  }

  // Helper methods

  private createEventDataFrame(events: UserEvent[]): EventDataFrame {
    const sortedEvents = events.sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    return {
      events: sortedEvents,
      startDate: new Date(sortedEvents[0].timestamp),
      endDate: new Date(sortedEvents[sortedEvents.length - 1].timestamp),
      totalEvents: events.length,
      uniqueEventTypes: new Set(events.map(e => e.event_type)).size
    };
  }

  private createEmptyMetrics(userId: string): BehavioralMetrics {
    const now = new Date();
    return {
      user_id: userId,
      calculated_at: now,
      engagement_score: 0,
      daily_active_rate: 0,
      weekly_active_days: 0,
      engagement_trend: 0,
      total_events: 0,
      unique_event_types: 0,
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
      days_since_signup: 999,
      lifetime_value_score: 0,
      upgrade_probability: 0,
      churn_risk_score: 0
    };
  }

  private identifySessions(events: UserEvent[]): UserEvent[][] {
    if (events.length === 0) return [];

    const sessions: UserEvent[][] = [];
    let currentSession: UserEvent[] = [events[0]];

    for (let i = 1; i < events.length; i++) {
      const currentEvent = events[i];
      const lastEvent = currentSession[currentSession.length - 1];
      
      const timeDiff = new Date(currentEvent.timestamp).getTime() - 
                      new Date(lastEvent.timestamp).getTime();
      const minutesDiff = timeDiff / (1000 * 60);

      if (minutesDiff <= this.SESSION_BOUNDARY_MINUTES) {
        currentSession.push(currentEvent);
      } else {
        sessions.push(currentSession);
        currentSession = [currentEvent];
      }
    }

    if (currentSession.length > 0) {
      sessions.push(currentSession);
    }

    return sessions;
  }

  private calculateEngagementTrend(events: UserEvent[]): number {
    if (events.length < 7) return 0;

    // Group events by day
    const dailyEventCounts: Record<string, number> = {};
    events.forEach(event => {
      const day = new Date(event.timestamp).toDateString();
      dailyEventCounts[day] = (dailyEventCounts[day] || 0) + 1;
    });

    const counts = Object.values(dailyEventCounts);
    if (counts.length < 2) return 0;

    // Calculate simple linear trend
    const n = counts.length;
    const x = Array.from({length: n}, (_, i) => i);
    const y = counts;

    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
    const sumXX = x.reduce((acc, xi) => acc + xi * xi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const meanY = sumY / n;

    return meanY > 0 ? slope / meanY : 0;
  }

  private calculateEngagementScore(
    dailyActiveRate: number,
    weeklyActiveDays: number,
    engagementTrend: number,
    totalEvents: number
  ): number {
    const components = {
      activity: dailyActiveRate * 0.3,
      consistency: Math.min(weeklyActiveDays / 7, 1.0) * 0.3,
      trend: Math.max(Math.min(engagementTrend + 1, 2), 0) / 2 * 0.2,
      volume: Math.min(totalEvents / 100, 1.0) * 0.2
    };

    return Object.values(components).reduce((a, b) => a + b, 0);
  }

  private calculateSessionRegularity(sessions: UserEvent[][]): number {
    if (sessions.length < 2) return 1.0;

    const intervals: number[] = [];
    for (let i = 1; i < sessions.length; i++) {
      const prevSessionEnd = new Date(sessions[i-1][sessions[i-1].length - 1].timestamp);
      const currentSessionStart = new Date(sessions[i][0].timestamp);
      const intervalHours = differenceInHours(currentSessionStart, prevSessionEnd);
      intervals.push(intervalHours);
    }

    if (intervals.length === 0) return 1.0;

    const mean = this.average(intervals);
    const variance = this.variance(intervals, mean);
    const coefficientOfVariation = mean > 0 ? Math.sqrt(variance) / mean : 0;

    // Return inverse of CV (higher regularity = lower CV)
    return Math.max(0, 1 - Math.min(coefficientOfVariation, 1));
  }

  private calculateLTVScore(metrics: Partial<BehavioralMetrics>, daysSinceSignup: number): number {
    const baseLTV = 500;
    const engagementMultiplier = 1 + (metrics.engagement_score || 0.5);
    const featureMultiplier = 1 + (metrics.feature_adoption_rate || 0.3);
    const tenureMultiplier = Math.min(daysSinceSignup / 30, 3);

    return baseLTV * engagementMultiplier * featureMultiplier * tenureMultiplier;
  }

  private calculateUpgradeProbability(metrics: Partial<BehavioralMetrics>): number {
    let upgradeScore = 0;

    // Monetization signals (40% weight)
    upgradeScore += (metrics.monetization_score || 0) * 0.4;

    // Feature adoption (30% weight)
    if ((metrics.feature_adoption_rate || 0) > 0.6) {
      upgradeScore += 0.3;
    } else if ((metrics.feature_adoption_rate || 0) > 0.4) {
      upgradeScore += 0.15;
    }

    // Engagement level (20% weight)
    if ((metrics.engagement_score || 0) > 0.7) {
      upgradeScore += 0.2;
    } else if ((metrics.engagement_score || 0) > 0.5) {
      upgradeScore += 0.1;
    }

    // Power features usage (10% weight)
    if ((metrics.power_features_used || 0) > 0) {
      upgradeScore += 0.1;
    }

    return Math.min(upgradeScore, 1.0);
  }

  // Utility methods
  private countEventTypes(events: UserEvent[]): Record<string, number> {
    const counts: Record<string, number> = {};
    events.forEach(event => {
      counts[event.event_type] = (counts[event.event_type] || 0) + 1;
    });
    return counts;
  }

  private calculateEntropy(values: number[]): number {
    if (values.length === 0) return 0;
    
    const total = values.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;

    const probabilities = values.map(v => v / total);
    return -probabilities.reduce((entropy, p) => {
      return p > 0 ? entropy + p * Math.log2(p) : entropy;
    }, 0);
  }

  private average(numbers: number[]): number {
    return numbers.length > 0 ? numbers.reduce((a, b) => a + b, 0) / numbers.length : 0;
  }

  private variance(numbers: number[], mean?: number): number {
    if (numbers.length === 0) return 0;
    const avg = mean !== undefined ? mean : this.average(numbers);
    const squaredDiffs = numbers.map(n => Math.pow(n - avg, 2));
    return this.average(squaredDiffs);
  }

  private roundTo2Decimals(num: number): number {
    return Math.round(num * 100) / 100;
  }

  private roundTo3Decimals(num: number): number {
    return Math.round(num * 1000) / 1000;
  }
} 