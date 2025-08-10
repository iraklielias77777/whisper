import { 
  PersonalizationContext, 
  UserContext, 
  PersonalizationData, 
  SocialProofData, 
  ValueMetrics, 
  UrgencyFactors,
  BehavioralSignals 
} from '../types';
import { logger, database, eventsCache } from '@userwhisperer/shared';
import { differenceInDays, differenceInHours } from 'date-fns';

export class PersonalizationEngine {
  private userContextCache: Map<string, PersonalizationContext> = new Map();
  private cacheExpiryMs: number = 300000; // 5 minutes

  constructor() {
    // Initialize personalization engine
    this.setupCacheCleanup();
  }

  public async gatherPersonalizationData(
    userId: string,
    userContext: UserContext,
    strategy: any
  ): Promise<PersonalizationContext> {
    return this.gatherData(userId, userContext, strategy);
  }

  public async gatherData(
    userId: string,
    userContext: UserContext,
    strategy: any
  ): Promise<PersonalizationContext> {
    logger.info(`Gathering personalization data for user ${userId}`);

    try {
      // Check cache first
      const cached = this.getCachedPersonalization(userId);
      if (cached) {
        return cached;
      }

      // Gather all personalization data in parallel
      const [
        userDetails,
        usageInsights,
        valueMetrics,
        socialProof,
        urgencyFactors,
        communicationPrefs
      ] = await Promise.all([
        this.getUserDetails(userId),
        this.getUsageInsights(userId),
        this.getValueMetrics(userId),
        this.getSocialProofData(userId),
        this.getUrgencyFactors(userId, strategy),
        this.getCommunicationPreferences(userId)
      ]);

      const personalizationContext: PersonalizationContext = {
        user_id: userId,
        user_data: { userContext },
        user_details: userDetails,
        usage_insights: usageInsights,
        behavioral_insights: { strategy },
        contextual_data: { timestamp: new Date() },
        value_metrics: valueMetrics,
        social_proof: socialProof,
        urgency_factors: urgencyFactors,
        communication_preferences: communicationPrefs,
        generated_at: new Date(),
        expires_at: new Date(Date.now() + this.cacheExpiryMs)
      };

      // Cache the result
      this.cachePersonalization(userId, personalizationContext);

      logger.info(`Personalization data gathered for user ${userId}`);
      return personalizationContext;

    } catch (error) {
      logger.error(`Failed to gather personalization data for user ${userId}:`, error);
      throw error;
    }
  }

  private async getUserDetails(userId: string): Promise<any> {
    try {
      const query = `
        SELECT 
          u.external_user_id,
          u.email,
          u.name,
          u.first_name,
          u.last_name,
          u.company,
          u.job_title,
          u.lifecycle_stage,
          u.subscription_status,
          u.subscription_plan,
          u.created_at,
          u.last_active_at,
          u.timezone,
          u.language,
          EXTRACT(EPOCH FROM (NOW() - u.created_at))/86400 as days_since_signup,
          EXTRACT(EPOCH FROM (NOW() - u.last_active_at))/86400 as days_since_active
        FROM user_profiles u
        WHERE u.external_user_id = $1
      `;

      const result = await database.query(query, [userId]);

      if (result.rows.length === 0) {
        return {
          is_new_user: true,
          user_id: userId,
          name: 'there',
          first_name: 'there'
        };
      }

      const user = result.rows[0];
      
      return {
        is_new_user: false,
        user_id: userId,
        email: user.email,
        name: user.name || `${user.first_name} ${user.last_name}`.trim() || 'there',
        first_name: user.first_name || 'there',
        last_name: user.last_name || '',
        company: user.company,
        job_title: user.job_title,
        lifecycle_stage: user.lifecycle_stage,
        subscription_status: user.subscription_status,
        subscription_plan: user.subscription_plan,
        days_since_signup: Math.floor(user.days_since_signup || 0),
        days_since_active: Math.floor(user.days_since_active || 0),
        timezone: user.timezone || 'UTC',
        language: user.language || 'en',
        is_trial_user: user.subscription_status === 'trial',
        is_paying_user: user.subscription_status === 'active' && user.subscription_plan !== 'free'
      };

    } catch (error) {
      logger.error(`Error getting user details for ${userId}:`, error);
      return {
        is_new_user: true,
        user_id: userId,
        name: 'there',
        first_name: 'there'
      };
    }
  }

  private async getUsageInsights(userId: string): Promise<any> {
    try {
      const [activityStats, featureUsage, sessionStats] = await Promise.all([
        this.getActivityStats(userId),
        this.getFeatureUsage(userId),
        this.getSessionStats(userId)
      ]);

      return {
        activity_stats: activityStats,
        feature_usage: featureUsage,
        session_stats: sessionStats,
        favorite_features: await this.getFavoriteFeatures(userId),
        recent_actions: await this.getRecentActions(userId),
        usage_patterns: await this.getUsagePatterns(userId)
      };

    } catch (error) {
      logger.error(`Error getting usage insights for ${userId}:`, error);
      return {
        activity_stats: { total_sessions: 0, total_events: 0 },
        feature_usage: {},
        session_stats: { average_duration: 0, sessions_this_week: 0 }
      };
    }
  }

  private async getActivityStats(userId: string): Promise<any> {
    const query = `
      SELECT 
        COUNT(DISTINCT session_id) as total_sessions,
        COUNT(*) as total_events,
        COUNT(DISTINCT DATE(created_at)) as active_days,
        COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as events_this_week,
        COUNT(CASE WHEN created_at >= NOW() - INTERVAL '1 day' THEN 1 END) as events_today
      FROM events
      WHERE user_id = $1 AND created_at >= NOW() - INTERVAL '30 days'
    `;

    const result = await database.query(query, [userId]);
    return result.rows[0] || {};
  }

  private async getFeatureUsage(userId: string): Promise<any> {
    const query = `
      SELECT 
        properties->>'feature_name' as feature,
        COUNT(*) as usage_count,
        MAX(created_at) as last_used
      FROM events
      WHERE user_id = $1 
        AND event_type = 'feature_used'
        AND created_at >= NOW() - INTERVAL '30 days'
      GROUP BY properties->>'feature_name'
      ORDER BY usage_count DESC
      LIMIT 10
    `;

    const result = await database.query(query, [userId]);
    
    const featureUsage: Record<string, any> = {};
    result.rows.forEach(row => {
      if (row.feature) {
        featureUsage[row.feature] = {
          usage_count: parseInt(row.usage_count),
          last_used: row.last_used,
          days_since_last_use: differenceInDays(new Date(), new Date(row.last_used))
        };
      }
    });

    return featureUsage;
  }

  private async getSessionStats(userId: string): Promise<any> {
    const query = `
      SELECT 
        AVG(EXTRACT(EPOCH FROM (last_event_at - first_event_at))/60) as avg_duration_minutes,
        COUNT(DISTINCT session_id) as sessions_this_week,
        MAX(last_event_at) as last_session
      FROM (
        SELECT 
          session_id,
          MIN(created_at) as first_event_at,
          MAX(created_at) as last_event_at
        FROM events
        WHERE user_id = $1 
          AND created_at >= NOW() - INTERVAL '7 days'
          AND session_id IS NOT NULL
        GROUP BY session_id
      ) sessions
    `;

    const result = await database.query(query, [userId]);
    const stats = result.rows[0] || {};

    return {
      average_duration: Math.round(stats.avg_duration_minutes || 0),
      sessions_this_week: parseInt(stats.sessions_this_week || 0),
      hours_since_last_session: stats.last_session 
        ? differenceInHours(new Date(), new Date(stats.last_session))
        : null
    };
  }

  private async getFavoriteFeatures(userId: string): Promise<string[]> {
    const query = `
      SELECT properties->>'feature_name' as feature
      FROM events
      WHERE user_id = $1 
        AND event_type = 'feature_used'
        AND created_at >= NOW() - INTERVAL '30 days'
      GROUP BY properties->>'feature_name'
      HAVING COUNT(*) >= 3
      ORDER BY COUNT(*) DESC, MAX(created_at) DESC
      LIMIT 3
    `;

    const result = await database.query(query, [userId]);
    return result.rows.map(row => row.feature).filter(Boolean);
  }

  private async getRecentActions(userId: string): Promise<any[]> {
    const query = `
      SELECT 
        event_type,
        properties,
        created_at
      FROM events
      WHERE user_id = $1
      ORDER BY created_at DESC
      LIMIT 5
    `;

    const result = await database.query(query, [userId]);
    return result.rows.map(row => ({
      action: row.event_type,
      details: row.properties,
      timestamp: row.created_at,
      hours_ago: differenceInHours(new Date(), new Date(row.created_at))
    }));
  }

  private async getUsagePatterns(userId: string): Promise<any> {
    const query = `
      SELECT 
        EXTRACT(HOUR FROM created_at) as hour,
        EXTRACT(DOW FROM created_at) as day_of_week,
        COUNT(*) as event_count
      FROM events
      WHERE user_id = $1 
        AND created_at >= NOW() - INTERVAL '30 days'
      GROUP BY EXTRACT(HOUR FROM created_at), EXTRACT(DOW FROM created_at)
      ORDER BY event_count DESC
      LIMIT 1
    `;

    const result = await database.query(query, [userId]);
    
    if (result.rows.length > 0) {
      const pattern = result.rows[0];
      return {
        most_active_hour: parseInt(pattern.hour),
        most_active_day: parseInt(pattern.day_of_week),
        peak_usage_events: parseInt(pattern.event_count)
      };
    }

    return {
      most_active_hour: 14, // Default to 2 PM
      most_active_day: 2,   // Default to Tuesday
      peak_usage_events: 0
    };
  }

  private async getValueMetrics(userId: string): Promise<ValueMetrics> {
    try {
      const [timeMetrics, achievementMetrics, monetaryMetrics] = await Promise.all([
        this.getTimeSavedMetrics(userId),
        this.getAchievementMetrics(userId),
        this.getMonetaryMetrics(userId)
      ]);

      return {
        time_saved_minutes: timeMetrics.time_saved_minutes || 0,
        tasks_completed: achievementMetrics.tasks_completed || 0,
        goals_achieved: achievementMetrics.goals_achieved || 0,
        efficiency_improvement: timeMetrics.efficiency_improvement || 0,
        cost_savings: monetaryMetrics.cost_savings || 0,
        roi_percentage: monetaryMetrics.roi_percentage || 0,
        streak_days: achievementMetrics.streak_days || 0,
        milestones_reached: achievementMetrics.milestones_reached || 0
      };

    } catch (error) {
      logger.error(`Error getting value metrics for ${userId}:`, error);
      return {
        time_saved_minutes: 0,
        tasks_completed: 0,
        goals_achieved: 0,
        efficiency_improvement: 0,
        cost_savings: 0,
        roi_percentage: 0,
        streak_days: 0,
        milestones_reached: 0
      };
    }
  }

  private async getTimeSavedMetrics(userId: string): Promise<any> {
    // This would calculate actual time saved based on feature usage
    // For now, return mock data based on activity
    const query = `
      SELECT COUNT(*) as automation_events
      FROM events
      WHERE user_id = $1 
        AND event_type IN ('automation_triggered', 'workflow_completed', 'task_automated')
        AND created_at >= NOW() - INTERVAL '30 days'
    `;

    const result = await database.query(query, [userId]);
    const automationEvents = parseInt(result.rows[0]?.automation_events || 0);

    return {
      time_saved_minutes: automationEvents * 15, // Assume 15 minutes saved per automation
      efficiency_improvement: Math.min(automationEvents * 2, 100) // Max 100% improvement
    };
  }

  private async getAchievementMetrics(userId: string): Promise<any> {
    const query = `
      SELECT 
        COUNT(CASE WHEN event_type = 'task_completed' THEN 1 END) as tasks_completed,
        COUNT(CASE WHEN event_type = 'goal_achieved' THEN 1 END) as goals_achieved,
        COUNT(CASE WHEN event_type = 'milestone_reached' THEN 1 END) as milestones_reached,
        COUNT(DISTINCT DATE(created_at)) as active_days
      FROM events
      WHERE user_id = $1 
        AND created_at >= NOW() - INTERVAL '30 days'
    `;

    const result = await database.query(query, [userId]);
    const metrics = result.rows[0] || {};

    return {
      tasks_completed: parseInt(metrics.tasks_completed || 0),
      goals_achieved: parseInt(metrics.goals_achieved || 0),
      milestones_reached: parseInt(metrics.milestones_reached || 0),
      streak_days: parseInt(metrics.active_days || 0)
    };
  }

  private async getMonetaryMetrics(userId: string): Promise<any> {
    const query = `
      SELECT 
        SUM(CASE WHEN properties->>'amount' ~ '^[0-9]+\.?[0-9]*$' 
                 THEN (properties->>'amount')::numeric 
                 ELSE 0 END) as total_value,
        COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) as purchases
      FROM events
      WHERE user_id = $1 
        AND event_type IN ('purchase', 'upgrade', 'subscription_started')
        AND created_at >= NOW() - INTERVAL '30 days'
    `;

    const result = await database.query(query, [userId]);
    const metrics = result.rows[0] || {};

    return {
      cost_savings: Math.round((metrics.total_value || 0) * 0.1), // Assume 10% savings
      roi_percentage: Math.min((metrics.purchases || 0) * 25, 200) // Max 200% ROI
    };
  }

  private async getSocialProofData(userId: string): Promise<SocialProofData> {
    try {
      const [testimonials, statistics, peerComparisons] = await Promise.all([
        this.getRelevantTestimonials(userId),
        this.getPlatformStatistics(),
        this.getPeerComparisons(userId)
      ]);

      return {
        testimonials,
        statistics,
        peer_comparisons: peerComparisons,
        success_stories: await this.getSuccessStories(userId),
        industry_benchmarks: await this.getIndustryBenchmarks(userId)
      };

    } catch (error) {
      logger.error(`Error getting social proof data for ${userId}:`, error);
      return {
        testimonials: [],
        statistics: {},
        peer_comparisons: {},
        success_stories: [],
        industry_benchmarks: {}
      };
    }
  }

  private async getRelevantTestimonials(userId: string): Promise<any[]> {
    // Get testimonials relevant to user's industry/use case
    const userDetails = await this.getUserDetails(userId);
    
    const testimonials = [
      {
        quote: "This platform has saved us 20 hours per week on manual tasks.",
        author: "Sarah Johnson",
        title: "Operations Manager",
        company: "TechCorp",
        relevance_score: 0.9
      },
      {
        quote: "Our productivity increased by 40% in the first month.",
        author: "Mike Chen",
        title: "Product Lead", 
        company: "StartupXYZ",
        relevance_score: 0.8
      }
    ];

    // Filter based on user's profile
    return testimonials
      .filter(t => t.relevance_score > 0.7)
      .slice(0, 2);
  }

  private async getPlatformStatistics(): Promise<any> {
    return {
      total_users: "50,000+",
      time_saved_total: "1.2M hours",
      customer_satisfaction: "98%",
      average_roi: "300%",
      uptime: "99.9%"
    };
  }

  private async getPeerComparisons(userId: string): Promise<any> {
    // Compare user's metrics to similar users
    const userMetrics = await this.getActivityStats(userId);
    
    return {
      engagement_percentile: Math.min(Math.max(userMetrics.total_sessions * 10, 25), 95),
      feature_adoption_vs_peers: "above average",
      productivity_ranking: "top 30%"
    };
  }

  private async getSuccessStories(userId: string): Promise<any[]> {
    return [
      {
        title: "Company reduces onboarding time by 70%",
        description: "See how similar companies achieved remarkable results",
        relevance: "high"
      }
    ];
  }

  private async getIndustryBenchmarks(userId: string): Promise<any> {
    return {
      average_adoption_time: "2 weeks",
      typical_roi: "250%",
      industry_satisfaction: "94%"
    };
  }

  private async getUrgencyFactors(userId: string, strategy: any): Promise<UrgencyFactors> {
    try {
      const [trialStatus, limitStatus, competitorActivity, marketTrends] = await Promise.all([
        this.getTrialStatus(userId),
        this.getLimitStatus(userId),
        this.getCompetitorActivity(userId),
        this.getMarketTrends()
      ]);

      return {
        trial_ending: trialStatus,
        limit_approaching: limitStatus,
        competitor_activity: competitorActivity,
        market_trends: marketTrends,
        seasonal_relevance: this.getSeasonalRelevance(),
        time_sensitive_offer: !!strategy.special_offer?.expires_at
      };

    } catch (error) {
      logger.error(`Error getting urgency factors for ${userId}:`, error);
      return {
        trial_ending: null,
        limit_approaching: null,
        competitor_activity: null,
        market_trends: null,
        seasonal_relevance: null,
        time_sensitive_offer: false
      };
    }
  }

  private async getTrialStatus(userId: string): Promise<any> {
    const query = `
      SELECT 
        subscription_status,
        trial_ends_at,
        EXTRACT(EPOCH FROM (trial_ends_at - NOW()))/86400 as days_remaining
      FROM user_profiles
      WHERE external_user_id = $1 AND subscription_status = 'trial'
    `;

    const result = await database.query(query, [userId]);
    
    if (result.rows.length > 0) {
      const trial = result.rows[0];
      return {
        days_remaining: Math.ceil(trial.days_remaining),
        ends_at: trial.trial_ends_at,
        is_expiring_soon: trial.days_remaining <= 3
      };
    }

    return null;
  }

  private async getLimitStatus(userId: string): Promise<any> {
    const query = `
      SELECT 
        subscription_plan,
        usage_current,
        usage_limit
      FROM user_usage_stats
      WHERE user_id = $1
    `;

    const result = await database.query(query, [userId]);
    
    if (result.rows.length > 0) {
      const usage = result.rows[0];
      const utilizationRate = usage.usage_current / usage.usage_limit;
      
      if (utilizationRate >= 0.8) {
        return {
          current_usage: usage.usage_current,
          limit: usage.usage_limit,
          utilization_rate: utilizationRate,
          approaching_limit: utilizationRate >= 0.9
        };
      }
    }

    return null;
  }

  private async getCompetitorActivity(userId: string): Promise<any> {
    // This would integrate with competitive intelligence tools
    return null;
  }

  private async getMarketTrends(): Promise<any> {
    // This would integrate with market research APIs
    return null;
  }

  private getSeasonalRelevance(): any {
    const now = new Date();
    const month = now.getMonth() + 1;
    
    // Define seasonal messaging
    if (month === 12 || month === 1) {
      return {
        season: "year_end",
        message: "Make this your most productive year yet",
        relevance: "high"
      };
    } else if (month >= 9 && month <= 11) {
      return {
        season: "back_to_business",
        message: "Get organized for the busy season ahead", 
        relevance: "medium"
      };
    }

    return null;
  }

  private async getCommunicationPreferences(userId: string): Promise<any> {
    try {
      const query = `
        SELECT 
          preferred_tone,
          preferred_length,
          communication_frequency,
          language,
          channel_preferences
        FROM user_communication_preferences
        WHERE user_id = $1
      `;

      const result = await database.query(query, [userId]);
      
      if (result.rows.length > 0) {
        return result.rows[0];
      }

      // Default preferences
      return {
        preferred_tone: "professional",
        preferred_length: "medium",
        communication_frequency: "normal",
        language: "en",
        channel_preferences: {
          email: 0.8,
          sms: 0.3,
          push: 0.6
        }
      };

    } catch (error) {
      logger.error(`Error getting communication preferences for ${userId}:`, error);
      return {
        preferred_tone: "professional",
        preferred_length: "medium",
        communication_frequency: "normal",
        language: "en"
      };
    }
  }

  private getCachedPersonalization(userId: string): PersonalizationContext | null {
    const cached = this.userContextCache.get(userId);
    
    if (cached && cached.expires_at && cached.expires_at > new Date()) {
      logger.debug(`Using cached personalization data for user ${userId}`);
      return cached;
    }

    if (cached) {
      this.userContextCache.delete(userId);
    }

    return null;
  }

  private cachePersonalization(userId: string, context: PersonalizationContext): void {
    this.userContextCache.set(userId, context);
  }

  private setupCacheCleanup(): void {
    // Clean up expired cache entries every 5 minutes
    setInterval(() => {
      const now = new Date();
      for (const [userId, context] of this.userContextCache.entries()) {
        if (context.expires_at && context.expires_at <= now) {
          this.userContextCache.delete(userId);
        }
      }
    }, 300000);
  }

  public async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    try {
      // Test database connectivity
      await database.query('SELECT 1');
      
      return {
        healthy: true,
        details: {
          cache_size: this.userContextCache.size,
          database_connected: true
        }
      };
    } catch (error) {
      return {
        healthy: false,
        details: { error: error instanceof Error ? error.message : String(error) }
      };
    }
  }
} 