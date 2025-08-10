export enum InterventionType {
  ONBOARDING = "onboarding",
  RETENTION = "retention",
  MONETIZATION = "monetization",
  REACTIVATION = "reactivation",
  SUPPORT = "support",
  CELEBRATION = "celebration",
  EDUCATION = "education"
}

export enum UrgencyLevel {
  NONE = "none",
  LOW = "low",
  MEDIUM = "medium",
  HIGH = "high",
  CRITICAL = "critical"
}

export enum Channel {
  EMAIL = "email",
  SMS = "sms",
  PUSH = "push",
  IN_APP = "in_app",
  WEBHOOK = "webhook"
}

export interface DecisionContext {
  user_id: string;
  trigger_event: TriggerEvent;
  behavioral_scores: BehavioralScores;
  user_profile: UserProfile;
  message_history: MessageHistory[];
  current_time: Date;
}

export interface TriggerEvent {
  event_id: string;
  event_type: string;
  timestamp: string;
  properties: Record<string, any>;
  context: Record<string, any>;
}

export interface BehavioralScores {
  churn_risk: number;
  upgrade_probability: number;
  engagement_score: number;
  feature_adoption_rate: number;
  support_tickets: number;
  monetization_score: number;
  days_since_last_active: number;
  lifecycle_stage: string;
  hitting_limits?: boolean;
  [key: string]: any;
}

export interface UserProfile {
  user_id: string;
  external_user_id: string;
  email?: string;
  name?: string;
  lifecycle_stage: string;
  subscription_plan?: string;
  subscription_status?: string;
  days_since_signup: number;
  timezone?: string;
  channel_preferences: Record<Channel, number>;
  optimal_send_hours: number[];
  created_at: Date;
  last_active_at?: Date;
  metadata: Record<string, any>;
}

export interface MessageHistory {
  message_id: string;
  user_id: string;
  channel: Channel;
  intervention_type: InterventionType;
  sent_at: Date;
  opened_at?: Date;
  clicked_at?: Date;
  unsubscribed_at?: Date;
  bounced_at?: Date;
  success: boolean;
  metadata: Record<string, any>;
}

export interface InterventionDecision {
  should_intervene: boolean;
  intervention_type?: InterventionType;
  urgency: UrgencyLevel;
  channel?: Channel;
  send_time?: Date;
  content_strategy?: ContentStrategy;
  confidence_score: number;
  reasoning: string;
  decision_id: string;
  created_at: Date;
}

export interface ContentStrategy {
  template_id?: string;
  personalization_level: 'low' | 'medium' | 'high';
  tone: string;
  approach: string;
  key_messages: string[];
  cta_type: string;
  urgency_indicators: string[];
  social_proof?: boolean;
  special_offer?: SpecialOffer;
}

export interface SpecialOffer {
  type: 'discount' | 'trial' | 'feature_unlock' | 'consultation';
  value: string;
  expiry_date?: Date;
  conditions?: string[];
}

export interface Strategy {
  intervention_type: InterventionType;
  urgency: UrgencyLevel;
  goal: string;
  approach: string;
  max_attempts: number;
  success_metrics: string[];
  content_hints: Record<string, any>;
  recommended_channel?: Channel;
  ai_confidence?: number;
  ai_reasoning?: string[];
  ai_enhanced?: boolean;
}

export interface Action {
  id: string;
  name: string;
  type: string;
  value: number; // Expected value score
  features: Record<string, any>;
  success_probability?: number;
}

export interface MLPrediction {
  action: Action;
  confidence: number;
  alternatives: Array<{
    action: Action;
    probability: number;
    expected_value: number;
  }>;
}

export interface FatigueCheck {
  is_fatigued: boolean;
  reason?: string;
  suggested_wait_time?: number; // in hours
  recent_message_count: number;
  last_message_hours_ago?: number;
}

export interface ChannelPerformance {
  channel: Channel;
  user_id: string;
  sent_count: number;
  opened_count: number;
  clicked_count: number;
  unsubscribed_count: number;
  bounced_count: number;
  open_rate: number;
  click_rate: number;
  conversion_rate: number;
  last_updated: Date;
}

export interface TimingOptimization {
  optimal_hours: number[];
  timezone: string;
  avoid_hours: number[];
  preferred_days: number[]; // 0-6, Sunday-Saturday
  minimum_gap_hours: number;
}

export interface DecisionEngineConfig {
  intervention_threshold: number;
  intervention_weights: {
    churn: number;
    monetization: number;
    onboarding: number;
    support: number;
    celebration: number;
    event_trigger: number;
  };
  fatigue_limits: {
    daily_max: number;
    weekly_max: number;
    monthly_max: number;
    min_gap_hours: number;
  };
  channel_costs: Record<Channel, number>;
  ml_model_enabled: boolean;
  default_timezone: string;
}

export interface DecisionRequest {
  user_id: string;
  trigger_event?: TriggerEvent;
  force_analysis?: boolean;
  preferred_channel?: Channel;
  max_wait_hours?: number;
  user_context?: any;
  user_profile?: UserProfile | any;
}

export interface DecisionResponse {
  decision: InterventionDecision;
  alternatives?: InterventionDecision[];
  processing_time_ms: number;
  model_version: string;
}

export interface ScheduledDecision {
  decision_id: string;
  user_id: string;
  decision: InterventionDecision;
  scheduled_for: Date;
  status: 'pending' | 'sent' | 'cancelled' | 'failed';
  attempts: number;
  created_at: Date;
  updated_at: Date;
} 