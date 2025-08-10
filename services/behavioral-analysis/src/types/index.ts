export enum UserLifecycleStage {
  NEW = "new",
  ONBOARDING = "onboarding", 
  ACTIVATED = "activated",
  ENGAGED = "engaged",
  POWER_USER = "power_user",
  AT_RISK = "at_risk",
  DORMANT = "dormant",
  CHURNED = "churned",
  REACTIVATED = "reactivated"
}

export interface UserEvent {
  event_id: string;
  app_id: string;
  user_id: string;
  event_type: string;
  timestamp: string;
  properties: Record<string, any>;
  context?: Record<string, any>;
  enrichment?: Record<string, any>;
  category?: string;
}

export interface UserProfile {
  user_id: string;
  external_user_id: string;
  email?: string;
  name?: string;
  lifecycle_stage: UserLifecycleStage;
  engagement_score: number;
  churn_risk_score: number;
  ltv_prediction: number;
  created_at: Date;
  updated_at: Date;
  subscription_status?: string;
  subscription_plan?: string;
  last_active_at?: Date;
  metadata: Record<string, any>;
}

export interface UserSegment {
  segment_id: string;
  name: string;
  description: string;
  criteria: Record<string, any>;
  user_count: number;
  created_at: Date;
  updated_at: Date;
}

export interface BehavioralMetrics {
  user_id: string;
  calculated_at: Date;
  
  // Engagement metrics
  engagement_score: number;
  daily_active_rate: number;
  weekly_active_days: number;
  engagement_trend: number;
  total_events: number;
  unique_event_types: number;
  days_since_last_active: number;
  
  // Session metrics
  total_sessions: number;
  avg_session_duration: number;
  avg_events_per_session: number;
  session_frequency: number;
  session_regularity: number;
  bounce_rate: number;
  pages_per_session: number;
  
  // Feature adoption metrics
  feature_adoption_rate: number;
  feature_depth: number;
  feature_breadth: number;
  power_features_used: number;
  unique_features_used: number;
  feature_usage_depth: number;
  new_feature_adoption: number;
  feature_stickiness: number;
  most_used_features: string[];
  
  // Monetization metrics
  monetization_score: number;
  pricing_page_views: number;
  upgrade_attempts: number;
  monetization_events_count: number;
  days_since_last_monetization_event: number;
  
  // Risk indicators
  error_rate: number;
  support_tickets: number;
  usage_decline: number;
  payment_failures: number;
  cancellation_signals: number;
  
  // Derived metrics
  days_since_signup: number;
  lifetime_value_score: number;
  upgrade_probability: number;
  churn_risk_score: number;
}

export interface BehavioralPattern {
  pattern_id: string;
  user_id: string;
  pattern_type: string;
  pattern_name: string;
  confidence_score: number;
  detected_at: Date;
  pattern_data: Record<string, any>;
  is_active: boolean;
}

export interface EngagementScores {
  user_id: string;
  overall_score: number;
  feature_usage_score: number;
  frequency_score: number;
  depth_score: number;
  recency_score: number;
  trend_score: number;
  calculated_at: Date;
}

export interface UserContext {
  is_new_user: boolean;
  user_id: string;
  external_user_id?: string;
  email?: string;
  name?: string;
  lifecycle_stage?: string;
  engagement_score?: number;
  churn_risk_score?: number;
  ltv_prediction?: number;
  created_at?: Date;
  subscription_status?: string;
  subscription_plan?: string;
  last_active_at?: Date;
  days_since_signup?: number;
  days_since_active?: number;
}

export interface SessionMetrics {
  session_id: string;
  user_id: string;
  started_at: Date;
  ended_at?: Date;
  duration_minutes: number;
  event_count: number;
  pages_visited: number;
  features_used: string[];
  is_active: boolean;
}

export interface ChurnPrediction {
  user_id: string;
  churn_probability: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  contributing_factors: string[];
  recommended_actions: string[];
  prediction_date: Date;
  model_version: string;
}

export interface LTVPrediction {
  user_id: string;
  predicted_ltv: number;
  confidence_interval: [number, number];
  time_horizon_days: number;
  contributing_factors: Record<string, number>;
  prediction_date: Date;
  model_version: string;
}

export interface UpgradePrediction {
  user_id: string;
  upgrade_probability: number;
  optimal_timing_days: number;
  recommended_plan: string;
  motivation_factors: string[];
  prediction_date: Date;
  model_version: string;
}

export interface AnalysisResult {
  user_id: string;
  metrics: BehavioralMetrics;
  patterns: BehavioralPattern[];
  engagement_scores: EngagementScores;
  churn_prediction: ChurnPrediction;
  ltv_prediction: LTVPrediction;
  upgrade_prediction: UpgradePrediction;
  lifecycle_transition?: {
    from: UserLifecycleStage;
    to: UserLifecycleStage;
    confidence: number;
    trigger_events: string[];
  };
  processed_at: Date;
}

export interface AnalysisConfig {
  batch_size: number;
  analysis_window_days: number;
  min_events_required: number;
  pattern_detection_enabled: boolean;
  ml_models_enabled: boolean;
  real_time_processing: boolean;
  feature_calculation_enabled: boolean;
}

export interface ModelFeatures {
  // Engagement features
  daily_active_rate: number;
  weekly_active_days: number;
  session_frequency: number;
  avg_session_duration: number;
  
  // Usage features
  total_events: number;
  unique_event_types: number;
  feature_adoption_rate: number;
  power_features_used: number;
  
  // Temporal features
  days_since_signup: number;
  days_since_last_active: number;
  engagement_trend: number;
  
  // Risk features
  error_rate: number;
  support_tickets: number;
  payment_failures: number;
  
  // Monetization features
  monetization_score: number;
  pricing_page_views: number;
  upgrade_attempts: number;
} 