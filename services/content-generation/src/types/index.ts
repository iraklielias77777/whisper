export interface ContentRequest {
  user_id: string;
  user_context: UserContext;
  intervention_type: string;
  strategy: ContentStrategy;
  channel: string;
  personalization_level: 'low' | 'medium' | 'high';
  urgency?: string;
  content_hints?: Record<string, any>;
}

export interface UserContext {
  user_id: string;
  name?: string;
  email?: string;
  lifecycle_stage: string;
  subscription_plan?: string;
  days_since_signup: number;
  engagement_score: number;
  feature_adoption_rate: number;
  churn_risk: number;
  upgrade_probability: number;
  timezone?: string;
  locale?: string;
  company_name?: string;
  industry?: string;
  recent_activity: RecentActivity[];
  behavioral_signals: BehavioralSignals;
  preferences: UserPreferences;
  user_details?: {
    first_name?: string;
    company?: string;
  };
  usage_insights?: {
    favorite_features?: string[];
    activity_stats?: Record<string, any>;
  };
  communication_preferences?: {
    preferred_length?: 'short' | 'medium' | 'long';
    preferred_tone?: 'professional' | 'friendly' | 'casual' | 'urgent';
  };
}

export interface RecentActivity {
  event_type: string;
  timestamp: string;
  properties: Record<string, any>;
}

export interface BehavioralSignals {
  most_used_feature?: string;
  last_login?: string;
  session_frequency: number;
  feature_usage: Record<string, number>;
  support_tickets: number;
  feedback_sentiment?: 'positive' | 'neutral' | 'negative';
  usage_trend: 'increasing' | 'stable' | 'decreasing';
  hitting_limits: boolean;
}

export interface UserPreferences {
  communication_frequency?: 'low' | 'medium' | 'high';
  content_type?: 'brief' | 'detailed' | 'visual';
  tone?: 'formal' | 'casual' | 'friendly' | 'professional';
  topics_of_interest?: string[];
  opt_out_types?: string[];
}

export interface ContentStrategy {
  template_id?: string;
  personalization_level: 'low' | 'medium' | 'high';
  tone: string;
  approach: string;
  key_messages: string[];
  cta_type: string;
  urgency?: 'low' | 'medium' | 'high' | 'critical';
  urgency_indicators: string[];
  social_proof?: boolean;
  special_offer?: SpecialOffer;
  content_hints: Record<string, any>;
}

export interface SpecialOffer {
  type: 'discount' | 'trial' | 'feature_unlock' | 'consultation';
  value: string;
  expiry_date?: Date;
  conditions?: string[];
}

export interface GeneratedContent {
  content_id: string;
  subject?: string;           // For email
  preview_text?: string;      // For email
  title?: string;             // For push/in-app
  body: string;
  cta_text: string;
  cta_link: string;
  metadata: ContentMetadata;
  personalization_data: PersonalizationData;
  generated_at: Date;
  expires_at?: Date;
}

export interface ContentMetadata {
  template_id?: string;
  generation_method: 'llm' | 'template' | 'hybrid' | 'fallback';
  model_used?: string;
  personalization_score: number;
  readability_score: number;
  sentiment_score: number;
  estimated_reading_time?: number;
  word_count: number;
  character_count: number;
  language: string;
  quality_score: number;
  variation_id?: string;
}

export interface PersonalizationData {
  elements_used: string[];
  user_specific_data: Record<string, any>;
  dynamic_content: Record<string, any>;
  fallback_content?: Record<string, any>;
  personalization_confidence: number;
}

export interface Template {
  id: string;
  name: string;
  description: string;
  intervention_type: string;
  channel: string;
  structure: TemplateStructure;
  variables: TemplateVariable[];
  success_rate: number;
  usage_count: number;
  created_at: Date;
  updated_at: Date;
  tags: string[];
  version: string;
}

export interface TemplateStructure {
  subject?: string;
  preview_text?: string;
  title?: string;
  body: string;
  cta: string;
  footer?: string;
  conditional_blocks?: ConditionalBlock[];
}

export interface ConditionalBlock {
  condition: string;
  content: string;
  priority: number;
}

export interface TemplateVariable {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'date' | 'array' | 'object';
  required: boolean;
  default_value?: any;
  description: string;
  validation?: VariableValidation;
}

export interface VariableValidation {
  min_length?: number;
  max_length?: number;
  pattern?: string;
  allowed_values?: any[];
}

export interface ContentVariation {
  id: string;
  content: GeneratedContent;
  score: number;
  generation_params: Record<string, any>;
  test_results?: TestResults;
}

export interface TestResults {
  open_rate?: number;
  click_rate?: number;
  conversion_rate?: number;
  engagement_score?: number;
  sample_size: number;
  confidence_interval?: number;
}

export interface PersonalizationContext {
  user_id: string;
  user_data: Record<string, any>;
  user_details?: Record<string, any>;
  usage_insights?: Record<string, any>;
  behavioral_insights: Record<string, any>;
  contextual_data: Record<string, any>;
  social_proof?: SocialProofData;
  social_proof_data?: SocialProofData;
  value_metrics?: ValueMetrics;
  urgency_factors?: UrgencyFactors;
  communication_preferences?: Record<string, any>;
  generated_at?: Date;
  expires_at?: Date;
}

export interface SocialProofData {
  similar_users_count?: number;
  success_stories?: string[];
  testimonials?: Testimonial[];
  usage_statistics?: Record<string, number>;
  peer_comparisons?: Record<string, any>;
  statistics?: Record<string, any>;
  industry_benchmarks?: Record<string, any>;
}

export interface Testimonial {
  id: string;
  text: string;
  author_name?: string;
  author_title?: string;
  author_company?: string;
  relevance_score: number;
  sentiment: 'positive' | 'neutral' | 'negative';
}

export interface ValueMetrics {
  time_saved?: number;
  time_saved_minutes?: number;
  money_saved?: number;
  cost_savings?: number;
  roi_percentage?: number;
  efficiency_gained?: number;
  efficiency_improvement?: number;
  goals_achieved?: number;
  milestones_reached?: string[] | number;
  tasks_completed?: number;
  streak_days?: number;
  comparative_metrics?: Record<string, number>;
}

export interface UrgencyFactors {
  time_sensitive_offer?: boolean;
  limited_availability?: boolean;
  deadline?: Date;
  consequence_of_inaction?: string;
  scarcity_indicators?: string[];
  trial_ending?: any;
  limit_approaching?: any;
  competitor_activity?: any;
  market_trends?: any;
  seasonal_relevance?: any;
}

export interface ContentGenConfig {
  llm_providers: {
    openai?: {
      api_key: string;
      model: string;
      max_tokens: number;
      temperature: number;
    };
    anthropic?: {
      api_key: string;
      model: string;
      max_tokens: number;
      temperature: number;
    };
  };
  template_engine: 'nunjucks' | 'handlebars';
  cache_ttl: number;
  max_variations: number;
  quality_threshold: number;
  personalization_threshold: number;
  fallback_strategy: 'template' | 'simple' | 'cache';
  content_moderation: boolean;
  a_b_testing_enabled: boolean;
}

export interface ContentValidationResult {
  is_valid: boolean;
  passes: boolean;
  quality_score: number;
  quality_metrics: QualityMetrics;
  issues: ContentIssue[];
  suggestions: string[];
  compliance_status: ComplianceStatus;
  validated_at: Date;
}

export interface ContentIssue {
  type: 'grammar' | 'spelling' | 'tone' | 'length' | 'formatting' | 'compliance' | 'validation_error' | 'quality' | 'spam' | 'personalization';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  suggestion?: string;
  field?: string;
  location?: {
    start: number;
    end: number;
  };
}

export interface ComplianceStatus {
  compliant: boolean;
  gdpr_compliant: boolean;
  can_spam_compliant: boolean;
  accessibility_score: number;
  content_warnings: string[];
  checks_passed: string[];
  checks_failed: string[];
}

export interface GenerationRequest {
  request_id: string;
  content_request: ContentRequest;
  timestamp: Date;
  priority: 'low' | 'medium' | 'high' | 'critical';
  retry_count?: number;
  parent_request_id?: string;
}

export interface GenerationResponse {
  request_id: string;
  status: 'success' | 'partial' | 'failed';
  content?: GeneratedContent;
  alternatives?: ContentVariation[];
  processing_time_ms: number;
  generation_method: 'llm' | 'template' | 'cache' | 'fallback' | 'hybrid';
  error?: string;
  warnings?: string[];
}

export interface ContentCache {
  cache_key: string;
  content: GeneratedContent;
  hit_count: number;
  created_at: Date;
  last_accessed: Date;
  expires_at: Date;
  tags: string[];
}

export interface ABTestConfig {
  test_id: string;
  name: string;
  description: string;
  variants: ABTestVariant[];
  traffic_allocation: Record<string, number>;
  success_metrics: string[];
  duration_days: number;
  min_sample_size: number;
  confidence_level: number;
  status: 'draft' | 'running' | 'paused' | 'completed';
}

export interface ABTestVariant {
  id: string;
  name: string;
  description: string;
  content_params: Record<string, any>;
  traffic_percentage: number;
}

export interface ContentAnalytics {
  content_id: string;
  template_id?: string;
  intervention_type: string;
  channel: string;
  generation_method: string;
  personalization_level: string;
  delivery_stats: DeliveryStats;
  engagement_stats: EngagementStats;
  conversion_stats: ConversionStats;
  quality_metrics: QualityMetrics;
  created_at: Date;
  last_updated: Date;
}

export interface DeliveryStats {
  sent_count: number;
  delivered_count: number;
  bounced_count: number;
  delivery_rate: number;
}

export interface EngagementStats {
  opened_count: number;
  clicked_count: number;
  shared_count: number;
  time_spent_reading?: number;
  open_rate: number;
  click_rate: number;
  engagement_rate: number;
}

export interface ConversionStats {
  conversion_count: number;
  conversion_rate: number;
  revenue_generated?: number;
  goal_completions: Record<string, number>;
}

export interface QualityMetrics {
  readability_score: number;
  sentiment_score: number;
  personalization_score: number;
  relevance_score: number;
  user_feedback_score?: number;
  spam_score: number;
  cta_strength: number;
  engagement_potential: number;
  length_appropriateness: number;
  sentiment_alignment: number;
  clarity_score: number;
  brand_alignment: number;
  urgency_effectiveness: number;
  trust_indicators: number;
}

export interface ContentOptimizationSuggestion {
  type: 'subject' | 'body' | 'cta' | 'timing' | 'personalization';
  current_value: string;
  suggested_value: string;
  expected_improvement: number;
  confidence: number;
  reasoning: string;
} 