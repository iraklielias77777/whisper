import { UserEvent, BehavioralMetrics, BehavioralPattern } from '../types';
import { logger } from '@userwhisperer/shared';
import { differenceInDays, differenceInHours, parseISO } from 'date-fns';
import { v4 as uuidv4 } from 'uuid';

export interface PatternDetectionConfig {
  powerUserThreshold: {
    engagementScore: number;
    featureAdoptionRate: number;
    dailyActiveRate: number;
    powerFeaturesUsed: number;
  };
  riskThresholds: {
    engagementDecline: number;
    sessionRegularity: number;
    errorRate: number;
    supportTickets: number;
    absenceDays: number;
  };
  anomalyThresholds: {
    usageSpike: number;
    rapidActivityThreshold: number;
    timeActivityDeviation: number;
  };
}

export class PatternDetector {
  private config: PatternDetectionConfig;

  constructor(config?: Partial<PatternDetectionConfig>) {
    this.config = {
      powerUserThreshold: {
        engagementScore: 0.8,
        featureAdoptionRate: 0.6,
        dailyActiveRate: 0.7,
        powerFeaturesUsed: 2,
        ...config?.powerUserThreshold
      },
      riskThresholds: {
        engagementDecline: -0.3,
        sessionRegularity: 0.3,
        errorRate: 0.1,
        supportTickets: 3,
        absenceDays: 7,
        ...config?.riskThresholds
      },
      anomalyThresholds: {
        usageSpike: 5, // 5x daily average
        rapidActivityThreshold: 10, // events within 1 second
        timeActivityDeviation: 6, // hours deviation from normal
        ...config?.anomalyThresholds
      }
    };
  }

  /**
   * Detect all relevant patterns for a user
   */
  public detectPatterns(
    events: UserEvent[],
    metrics: BehavioralMetrics
  ): BehavioralPattern[] {
    const patterns: BehavioralPattern[] = [];

    try {
      // Power user patterns
      if (this.isPowerUser(metrics)) {
        patterns.push(this.createPattern(
          metrics.user_id,
          'power_user',
          'Power User',
          0.9,
          { metrics: this.extractPowerUserMetrics(metrics) }
        ));
      }

      // At-risk patterns
      const riskPatterns = this.detectRiskPatterns(events, metrics);
      patterns.push(...riskPatterns);

      // Monetization patterns
      const monetizationPatterns = this.detectMonetizationPatterns(events, metrics);
      patterns.push(...monetizationPatterns);

      // Usage patterns
      const usagePatterns = this.detectUsagePatterns(events);
      patterns.push(...usagePatterns);

      // Anomalies
      const anomalyPatterns = this.detectAnomalies(events, metrics);
      patterns.push(...anomalyPatterns);

      logger.debug('Pattern detection completed', {
        userId: metrics.user_id,
        patternsDetected: patterns.length,
        patternTypes: patterns.map(p => p.pattern_type)
      });

    } catch (error) {
      logger.error('Pattern detection failed', {
        userId: metrics.user_id,
        error: error instanceof Error ? error.message : String(error)
      });
    }

    return patterns;
  }

  /**
   * Check if user exhibits power user behavior
   */
  private isPowerUser(metrics: BehavioralMetrics): boolean {
    const thresholds = this.config.powerUserThreshold;
    
    return (
      metrics.engagement_score >= thresholds.engagementScore &&
      metrics.feature_adoption_rate >= thresholds.featureAdoptionRate &&
      metrics.daily_active_rate >= thresholds.dailyActiveRate &&
      metrics.power_features_used >= thresholds.powerFeaturesUsed
    );
  }

  /**
   * Detect patterns indicating risk or churn
   */
  private detectRiskPatterns(
    events: UserEvent[],
    metrics: BehavioralMetrics
  ): BehavioralPattern[] {
    const patterns: BehavioralPattern[] = [];
    const thresholds = this.config.riskThresholds;

    // Declining usage
    if (metrics.engagement_trend < thresholds.engagementDecline) {
      patterns.push(this.createPattern(
        metrics.user_id,
        'risk',
        'Declining Usage',
        0.8,
        {
          engagementTrend: metrics.engagement_trend,
          threshold: thresholds.engagementDecline,
          severity: this.calculateDeclineSeverity(metrics.engagement_trend)
        }
      ));
    }

    // Irregular usage
    if (metrics.session_regularity < thresholds.sessionRegularity) {
      patterns.push(this.createPattern(
        metrics.user_id,
        'risk',
        'Irregular Usage',
        0.7,
        {
          sessionRegularity: metrics.session_regularity,
          threshold: thresholds.sessionRegularity,
          recentSessions: metrics.total_sessions
        }
      ));
    }

    // High error rate
    if (metrics.error_rate > thresholds.errorRate) {
      patterns.push(this.createPattern(
        metrics.user_id,
        'risk',
        'High Error Rate',
        0.9,
        {
          errorRate: metrics.error_rate,
          threshold: thresholds.errorRate,
          totalEvents: metrics.total_events
        }
      ));
    }

    // Payment issues
    if (metrics.payment_failures > 0) {
      patterns.push(this.createPattern(
        metrics.user_id,
        'risk',
        'Payment Issues',
        0.95,
        {
          paymentFailures: metrics.payment_failures,
          urgency: 'high'
        }
      ));
    }

    // Support escalation
    if (metrics.support_tickets >= thresholds.supportTickets) {
      patterns.push(this.createPattern(
        metrics.user_id,
        'risk',
        'Support Escalation',
        0.85,
        {
          supportTickets: metrics.support_tickets,
          threshold: thresholds.supportTickets,
          urgency: metrics.support_tickets > 5 ? 'critical' : 'high'
        }
      ));
    }

    // Considering cancellation
    if (metrics.cancellation_signals > 0) {
      patterns.push(this.createPattern(
        metrics.user_id,
        'risk',
        'Considering Cancellation',
        0.9,
        {
          cancellationSignals: metrics.cancellation_signals,
          urgency: 'critical'
        }
      ));
    }

    // Extended absence
    if (metrics.days_since_last_active >= thresholds.absenceDays) {
      const severity = this.calculateAbsenceSeverity(metrics.days_since_last_active);
      patterns.push(this.createPattern(
        metrics.user_id,
        'risk',
        'Extended Absence',
        0.8,
        {
          daysSinceLastActive: metrics.days_since_last_active,
          threshold: thresholds.absenceDays,
          severity
        }
      ));
    }

    return patterns;
  }

  /**
   * Detect patterns indicating monetization readiness
   */
  private detectMonetizationPatterns(
    events: UserEvent[],
    metrics: BehavioralMetrics
  ): BehavioralPattern[] {
    const patterns: BehavioralPattern[] = [];

    // Hitting limits
    const limitEvents = events.filter(e => e.event_type.includes('limit_reached'));
    if (limitEvents.length >= 3) {
      patterns.push(this.createPattern(
        metrics.user_id,
        'monetization',
        'Hitting Limits',
        0.85,
        {
          limitEvents: limitEvents.length,
          recentLimits: limitEvents.slice(-3).map(e => ({
            type: e.event_type,
            timestamp: e.timestamp
          })),
          urgency: 'high'
        }
      ));
    }

    // Exploring premium features
    const premiumEvents = events.filter(e => 
      e.event_type.includes('premium') || 
      e.event_type.includes('upgrade') ||
      e.event_type.includes('pro_feature')
    );
    if (premiumEvents.length >= 2) {
      patterns.push(this.createPattern(
        metrics.user_id,
        'monetization',
        'Exploring Premium',
        0.8,
        {
          premiumEvents: premiumEvents.length,
          explorationLevel: this.calculateExplorationLevel(premiumEvents),
          features: [...new Set(premiumEvents.map(e => e.event_type))]
        }
      ));
    }

    // High value user
    if (metrics.engagement_score >= 0.7 && metrics.feature_adoption_rate >= 0.5) {
      patterns.push(this.createPattern(
        metrics.user_id,
        'monetization',
        'High Value User',
        0.9,
        {
          engagementScore: metrics.engagement_score,
          featureAdoptionRate: metrics.feature_adoption_rate,
          daysSinceSignup: metrics.days_since_signup,
          readinessLevel: 'high'
        }
      ));
    }

    // Trial ending soon
    const trialEvents = events.filter(e => e.event_type === 'trial_started');
    if (trialEvents.length > 0) {
      const latestTrial = trialEvents[trialEvents.length - 1];
      const trialStart = new Date(latestTrial.timestamp);
      const daysInTrial = differenceInDays(new Date(), trialStart);
      
      if (daysInTrial >= 10 && daysInTrial <= 14) { // Assuming 14-day trial
        patterns.push(this.createPattern(
          metrics.user_id,
          'monetization',
          'Trial Ending Soon',
          0.95,
          {
            daysInTrial,
            trialStartDate: trialStart.toISOString(),
            daysRemaining: 14 - daysInTrial,
            urgency: 'critical'
          }
        ));
      }
    }

    return patterns;
  }

  /**
   * Detect specific usage patterns
   */
  private detectUsagePatterns(events: UserEvent[]): BehavioralPattern[] {
    const patterns: BehavioralPattern[] = [];

    if (events.length === 0) return patterns;

    // Time-based patterns
    const timestamps = events.map(e => new Date(e.timestamp));
    const hours = timestamps.map(ts => ts.getHours());

    // Business hours user (9-5)
    const businessHours = hours.filter(h => h >= 9 && h <= 17);
    if (businessHours.length / hours.length > 0.8) {
      patterns.push(this.createPattern(
        events[0].user_id,
        'usage',
        'Business Hours User',
        0.8,
        {
          businessHoursPercentage: (businessHours.length / hours.length) * 100,
          primaryHours: this.getMostActiveHours(hours),
          timezone: 'inferred_business'
        }
      ));
    }

    // Night owl (10pm - 2am)
    const nightHours = hours.filter(h => h >= 22 || h <= 2);
    if (nightHours.length / hours.length > 0.3) {
      patterns.push(this.createPattern(
        events[0].user_id,
        'usage',
        'Night Owl',
        0.7,
        {
          nightHoursPercentage: (nightHours.length / hours.length) * 100,
          lateNightActivity: nightHours.length,
          sleepPattern: 'late'
        }
      ));
    }

    // Weekend warrior
    const weekdays = timestamps.map(ts => ts.getDay());
    const weekendEvents = weekdays.filter(wd => wd === 0 || wd === 6);
    if (weekendEvents.length / weekdays.length > 0.4) {
      patterns.push(this.createPattern(
        events[0].user_id,
        'usage',
        'Weekend Warrior',
        0.7,
        {
          weekendPercentage: (weekendEvents.length / weekdays.length) * 100,
          weekendActivity: weekendEvents.length,
          workLifeBalance: 'weekend_focused'
        }
      ));
    }

    // Binge user
    if (this.detectBingePattern(timestamps)) {
      patterns.push(this.createPattern(
        events[0].user_id,
        'usage',
        'Binge User',
        0.8,
        {
          sessionPattern: 'binge',
          intensity: 'high',
          sustainabilityRisk: 'medium'
        }
      ));
    }

    // Mobile primary
    const mobileEvents = events.filter(e => 
      e.context?.device_type === 'mobile' || 
      e.enrichment?.device?.device_type === 'mobile'
    );
    if (mobileEvents.length / events.length > 0.7) {
      patterns.push(this.createPattern(
        events[0].user_id,
        'usage',
        'Mobile Primary',
        0.9,
        {
          mobilePercentage: (mobileEvents.length / events.length) * 100,
          devicePreference: 'mobile',
          mobilityLevel: 'high'
        }
      ));
    }

    return patterns;
  }

  /**
   * Detect anomalous behavior
   */
  private detectAnomalies(
    events: UserEvent[],
    metrics: BehavioralMetrics
  ): BehavioralPattern[] {
    const patterns: BehavioralPattern[] = [];

    if (events.length === 0) return patterns;

    // Sudden usage spike
    if (events.length > 100) {
      const now = new Date();
      const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      const recentEvents = events.filter(e => new Date(e.timestamp) > oneDayAgo);
      const dailyAverage = events.length / 30; // Assuming 30 days of data
      
      if (recentEvents.length > dailyAverage * this.config.anomalyThresholds.usageSpike) {
        patterns.push(this.createPattern(
          metrics.user_id,
          'anomaly',
          'Usage Spike',
          0.8,
          {
            recentEvents: recentEvents.length,
            dailyAverage,
            spikeMultiplier: recentEvents.length / dailyAverage,
            alertLevel: 'medium'
          }
        ));
      }
    }

    // Unusual time activity
    const timestamps = events.map(e => new Date(e.timestamp));
    const recentActivity = this.getRecentActivityPattern(timestamps);
    const historicalActivity = this.getHistoricalActivityPattern(timestamps);
    
    if (Math.abs(recentActivity.averageHour - historicalActivity.averageHour) > 
        this.config.anomalyThresholds.timeActivityDeviation) {
      patterns.push(this.createPattern(
        metrics.user_id,
        'anomaly',
        'Unusual Time Activity',
        0.7,
        {
          recentAverageHour: recentActivity.averageHour,
          historicalAverageHour: historicalActivity.averageHour,
          deviation: Math.abs(recentActivity.averageHour - historicalActivity.averageHour),
          alertLevel: 'low'
        }
      ));
    }

    // Suspicious rapid activity
    const rapidEvents = this.detectRapidActivity(events);
    if (rapidEvents.length > this.config.anomalyThresholds.rapidActivityThreshold) {
      patterns.push(this.createPattern(
        metrics.user_id,
        'anomaly',
        'Suspicious Rapid Activity',
        0.9,
        {
          rapidEvents: rapidEvents.length,
          threshold: this.config.anomalyThresholds.rapidActivityThreshold,
          suspiciousIntervals: rapidEvents.slice(0, 5), // Show first 5 examples
          alertLevel: 'high'
        }
      ));
    }

    return patterns;
  }

  // Helper methods

  private createPattern(
    userId: string,
    patternType: string,
    patternName: string,
    confidence: number,
    patternData: Record<string, any>
  ): BehavioralPattern {
    return {
      pattern_id: uuidv4(),
      user_id: userId,
      pattern_type: patternType,
      pattern_name: patternName,
      confidence_score: confidence,
      detected_at: new Date(),
      pattern_data: patternData,
      is_active: true
    };
  }

  private extractPowerUserMetrics(metrics: BehavioralMetrics): Record<string, any> {
    return {
      engagementScore: metrics.engagement_score,
      featureAdoptionRate: metrics.feature_adoption_rate,
      dailyActiveRate: metrics.daily_active_rate,
      powerFeaturesUsed: metrics.power_features_used,
      sessionFrequency: metrics.session_frequency,
      daysSinceSignup: metrics.days_since_signup
    };
  }

  private calculateDeclineSeverity(engagementTrend: number): string {
    if (engagementTrend <= -0.5) return 'critical';
    if (engagementTrend <= -0.4) return 'high';
    if (engagementTrend <= -0.3) return 'medium';
    return 'low';
  }

  private calculateAbsenceSeverity(days: number): string {
    if (days >= 30) return 'critical';
    if (days >= 14) return 'high';
    if (days >= 7) return 'medium';
    return 'low';
  }

  private calculateExplorationLevel(premiumEvents: UserEvent[]): string {
    const uniqueFeatures = new Set(premiumEvents.map(e => e.event_type)).size;
    if (uniqueFeatures >= 5) return 'extensive';
    if (uniqueFeatures >= 3) return 'moderate';
    return 'initial';
  }

  private getMostActiveHours(hours: number[]): number[] {
    const hourCounts: Record<number, number> = {};
    hours.forEach(hour => {
      hourCounts[hour] = (hourCounts[hour] || 0) + 1;
    });

    return Object.entries(hourCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([hour]) => parseInt(hour));
  }

  private detectBingePattern(timestamps: Date[]): boolean {
    if (timestamps.length < 10) return false;

    // Group events into sessions (30 min gap)
    const sessions: Date[][] = [];
    let currentSession = [timestamps[0]];

    for (let i = 1; i < timestamps.length; i++) {
      const timeDiff = timestamps[i].getTime() - currentSession[currentSession.length - 1].getTime();
      const minutesDiff = timeDiff / (1000 * 60);

      if (minutesDiff <= 30) {
        currentSession.push(timestamps[i]);
      } else {
        sessions.push(currentSession);
        currentSession = [timestamps[i]];
      }
    }

    if (currentSession.length > 0) {
      sessions.push(currentSession);
    }

    // Check for binge pattern
    const sessionLengths = sessions.map(s => s.length);
    if (sessionLengths.length === 0) return false;

    const averageSessionLength = sessionLengths.reduce((a, b) => a + b, 0) / sessionLengths.length;
    const maxSessionLength = Math.max(...sessionLengths);

    return (
      sessions.length < 10 &&
      maxSessionLength > averageSessionLength * 3 &&
      maxSessionLength > 20
    );
  }

  private getRecentActivityPattern(timestamps: Date[]): { averageHour: number } {
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
    const recentHours = timestamps
      .filter(ts => ts > oneDayAgo)
      .map(ts => ts.getHours());

    const averageHour = recentHours.length > 0
      ? recentHours.reduce((a, b) => a + b, 0) / recentHours.length
      : 12; // Default to noon

    return { averageHour };
  }

  private getHistoricalActivityPattern(timestamps: Date[]): { averageHour: number } {
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
    const historicalHours = timestamps
      .filter(ts => ts <= oneDayAgo)
      .map(ts => ts.getHours());

    const averageHour = historicalHours.length > 0
      ? historicalHours.reduce((a, b) => a + b, 0) / historicalHours.length
      : 12; // Default to noon

    return { averageHour };
  }

  private detectRapidActivity(events: UserEvent[]): Array<{ eventType: string; interval: number }> {
    const rapidEvents: Array<{ eventType: string; interval: number }> = [];

    for (let i = 1; i < events.length; i++) {
      const currentTime = new Date(events[i].timestamp).getTime();
      const previousTime = new Date(events[i - 1].timestamp).getTime();
      const intervalSeconds = (currentTime - previousTime) / 1000;

      if (intervalSeconds < 1) { // Less than 1 second between events
        rapidEvents.push({
          eventType: events[i].event_type,
          interval: intervalSeconds
        });
      }
    }

    return rapidEvents;
  }
} 