// User Whisperer Shared - Main Export File
// Exports all shared utilities, configurations, and components

// Configuration and utilities
export { default as Config } from './utils/config';
export { default as Logger } from './utils/logger';
export { createServiceLogger } from './utils/logger';

// Backwards compatibility - create a default logger
import { createServiceLogger } from './utils/logger';
export const logger = createServiceLogger('shared');

// Database components
export * from './utils/database';

// Redis components  
export * from './utils/redis';

// Storage management (Python components - TypeScript declarations)
export interface StorageManager {
  initialize(): Promise<void>;
  store_data(data_type: string, data: any, user_id?: string): Promise<any>;
  retrieve_data(data_type: string, data_id?: string, user_id?: string): Promise<any[]>;
  cleanup_expired_data(): Promise<any>;
  get_storage_stats(): Promise<any>;
  health_check(): Promise<any>;
}

export interface CacheManager {
  initialize(): Promise<void>;
  get(key: string, fetch_fn?: Function): Promise<any>;
  set(key: string, value: any, ttl?: number): Promise<void>;
  delete(key: string): Promise<void>;
  invalidate(pattern: string): Promise<void>;
  get_stats(): Promise<any>;
  health_check(): Promise<any>;
}

export interface DataLifecycleManager {
  initialize(): Promise<void>;
  start_lifecycle_management(): Promise<void>;
  stop_lifecycle_management(): Promise<void>;
  handle_gdpr_request(request_type: string, user_id: string, app_id: string): Promise<any>;
  get_lifecycle_stats(): Promise<any>;
}

export interface PerformanceOptimizer {
  initialize(): Promise<void>;
  start_monitoring(): Promise<void>;
  stop_monitoring(): Promise<void>;
  record_request(response_time?: number, error?: boolean): void;
  get_performance_report(): Promise<any>;
  get_optimization_recommendations(): Promise<any[]>;
}

// Stream processing (Python components - TypeScript declarations)
export interface EventStreamProcessor {
  initialize(): Promise<void>;
  register_handler(event_type: string, handler: Function): void;
  publish_event(event_type: string, data: any, ordering_key?: string): Promise<string>;
  start_processing(): Promise<void>;
  stop_processing(): Promise<void>;
  get_stats(): Promise<any>;
}

export interface ComplexEventProcessor {
  register_pattern(name: string, condition: any, action: Function): void;
  process_event(event: any): Promise<void>;
  get_user_patterns(user_id: string): Promise<any[]>;
  get_stats(): Promise<any>;
}

// Event schemas and validation
// Note: JSON schema is imported by services when needed
// JSON schema will be loaded at runtime using require() or fs.readFileSync()

// Protocol buffer types (generated from .proto files)
export interface EventMessage {
  id: string;
  app_id: string;
  user_id: string;
  event_type: string;
  properties: Record<string, any>;
  context: Record<string, any>;
  timestamp: string;
}

export interface BehavioralAnalysisMessage {
  user_id: string;
  app_id: string;
  metrics: Record<string, number>;
  patterns: string[];
  lifecycle_stage: string;
  scores: Record<string, number>;
}

// Utility types
export interface ServiceConfig {
  // Database configuration
  database: {
    host: string;
    port: number;
    user: string;
    password: string;
    database: string;
    read_replicas?: string[];
    min_pool_size?: number;
    max_pool_size?: number;
  };
  
  // Redis configuration
  redis: {
    url: string;
    max_retries?: number;
    retry_delay_ms?: number;
  };
  
  // Storage configuration
  storage: {
    redis_url: string;
    postgres: any;
    gcp_project: string;
    bigquery_dataset: string;
    s3_bucket: string;
    encryption_key?: string;
  };
  
  // Cache configuration
  cache: {
    redis_url: string;
    l1_max_size?: number;
    l1_ttl?: number;
    l2_ttl?: number;
    cache_warming?: boolean;
    cdn?: {
      enabled: boolean;
      base_url?: string;
      api_key?: string;
    };
  };
  
  // Stream processing configuration
  stream: {
    project_id: string;
    topic_name: string;
    subscription_name: string;
    max_messages?: number;
    ack_deadline?: number;
  };
}

// Service health check interface
export interface HealthCheck {
  healthy: boolean;
  timestamp: string;
  components: Record<string, boolean>;
  details?: Record<string, any>;
}

// Common response interfaces
export interface ServiceResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

export interface PaginatedResponse<T = any> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  has_more: boolean;
}

// Event processing interfaces
export interface ProcessedEvent {
  id: string;
  original_event: EventMessage;
  enriched_data: Record<string, any>;
  processing_time_ms: number;
  processed_at: string;
}

export interface EventProcessingResult {
  processed_count: number;
  failed_count: number;
  processing_time_ms: number;
  errors?: string[];
}

// User profile interfaces
export interface UserProfile {
  id: string;
  app_id: string;
  external_user_id: string;
  email?: string;
  name?: string;
  lifecycle_stage: string;
  engagement_score?: number;
  churn_risk_score?: number;
  ltv_prediction?: number;
  upgrade_probability?: number;
  channel_preferences: Record<string, boolean>;
  created_at: string;
  updated_at: string;
  last_active_at?: string;
}

// Behavioral analysis interfaces
export interface BehavioralMetrics {
  user_id: string;
  engagement_score: number;
  session_frequency: number;
  feature_adoption_rate: number;
  content_consumption: number;
  social_engagement: number;
  support_tickets: number;
  computed_at: string;
}

export interface BehavioralPattern {
  pattern_name: string;
  user_id: string;
  confidence_score: number;
  matched_events: any[];
  detected_at: string;
  metadata: Record<string, any>;
}

// Decision engine interfaces
export interface DecisionContext {
  user_id: string;
  app_id: string;
  trigger_event?: EventMessage;
  user_profile: UserProfile;
  recent_behavior: BehavioralMetrics;
  channel_history: any[];
  current_campaigns: any[];
}

export interface InterventionDecision {
  should_intervene: boolean;
  intervention_type?: string;
  channel?: string;
  message_template?: string;
  scheduled_for?: string;
  urgency_level: string;
  confidence_score: number;
  reasoning: string[];
}

// Content generation interfaces
export interface ContentRequest {
  intervention_type: string;
  channel: string;
  user_context: UserProfile;
  personalization_data: Record<string, any>;
  template_id?: string;
  tone?: string;
  length?: string;
}

export interface GeneratedContent {
  subject?: string;
  body: string;
  cta?: string;
  personalization_score: number;
  estimated_effectiveness: number;
  generated_at: string;
  template_used?: string;
  variations?: string[];
}

// Message delivery interfaces
export interface MessageDelivery {
  id: string;
  user_id: string;
  channel: string;
  content: GeneratedContent;
  scheduled_at: string;
  sent_at?: string;
  delivered_at?: string;
  opened_at?: string;
  clicked_at?: string;
  status: string;
  provider_response?: any;
}

// Analytics and reporting interfaces
export interface AnalyticsEvent {
  event_type: string;
  user_id: string;
  app_id: string;
  properties: Record<string, any>;
  timestamp: string;
}

export interface PerformanceMetrics {
  response_time_ms: number;
  throughput_rps: number;
  error_rate: number;
  cache_hit_rate: number;
  database_connections: number;
  memory_usage: number;
  cpu_usage: number;
}

export interface ServiceMetrics {
  uptime_seconds: number;
  requests_total: number;
  errors_total: number;
  performance: PerformanceMetrics;
  health_checks: HealthCheck;
  last_updated: string;
}

// Error handling interfaces
export interface ServiceError {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
  request_id?: string;
  user_id?: string;
}

// Validation and schema interfaces
export interface ValidationResult {
  valid: boolean;
  errors?: string[];
  warnings?: string[];
}

export interface SchemaValidation {
  validate(data: any): ValidationResult;
  getSchema(): any;
}

// Export utility functions
export function createServiceResponse<T>(
  success: boolean,
  data?: T,
  error?: string
): ServiceResponse<T> {
  return {
    success,
    data,
    error,
    timestamp: new Date().toISOString()
  };
}

export function createPaginatedResponse<T>(
  data: T[],
  total: number,
  page: number,
  limit: number
): PaginatedResponse<T> {
  return {
    data,
    total,
    page,
    limit,
    has_more: (page * limit) < total
  };
}

export function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

export { HealthChecker } from './utils/health';
export { ServiceDiscovery } from './utils/service-discovery';
export { MessageQueue, TOPICS, SUBSCRIPTIONS } from './utils/message-queue';
export { AIOrchestrationClient, aiOrchestrationClient } from './utils/ai-orchestration-client';

export function sanitizeUserData(data: any): any {
  // Remove sensitive fields for logging/analytics
  const sanitized = { ...data };
  delete sanitized.email;
  delete sanitized.phone;
  delete sanitized.password;
  delete sanitized.api_key;
  delete sanitized.api_secret;
  return sanitized;
}

// Constants
export const SERVICE_NAMES = {
  EVENT_INGESTION: 'event-ingestion',
  BEHAVIORAL_ANALYSIS: 'behavioral-analysis',
  DECISION_ENGINE: 'decision-engine',
  CONTENT_GENERATION: 'content-generation',
  CHANNEL_ORCHESTRATION: 'channel-orchestration',
  AI_ORCHESTRATION: 'ai-orchestration'
} as const;

export const EVENT_TYPES = {
  // User events
  USER_SIGNUP: 'user_signup',
  USER_LOGIN: 'user_login',
  USER_LOGOUT: 'user_logout',
  PROFILE_UPDATE: 'profile_update',
  
  // Engagement events
  PAGE_VIEW: 'page_view',
  FEATURE_USED: 'feature_used',
  CONTENT_VIEWED: 'content_viewed',
  SEARCH_PERFORMED: 'search_performed',
  
  // Transaction events
  PURCHASE_STARTED: 'purchase_started',
  PURCHASE_COMPLETED: 'purchase_completed',
  SUBSCRIPTION_STARTED: 'subscription_started',
  SUBSCRIPTION_CANCELLED: 'subscription_cancelled',
  
  // System events
  ERROR_OCCURRED: 'error_occurred',
  SESSION_STARTED: 'session_started',
  SESSION_ENDED: 'session_ended'
} as const;

export const LIFECYCLE_STAGES = {
  NEW: 'new',
  ONBOARDING: 'onboarding',
  ACTIVATED: 'activated',
  ENGAGED: 'engaged',
  POWER_USER: 'power_user',
  AT_RISK: 'at_risk',
  DORMANT: 'dormant',
  CHURNED: 'churned',
  REACTIVATED: 'reactivated'
} as const;

export const CHANNELS = {
  EMAIL: 'email',
  SMS: 'sms',
  PUSH: 'push',
  WEBHOOK: 'webhook',
  IN_APP: 'in_app'
} as const;

export const INTERVENTION_TYPES = {
  ONBOARDING: 'onboarding',
  ENGAGEMENT: 'engagement',
  RETENTION: 'retention',
  MONETIZATION: 'monetization',
  REACTIVATION: 'reactivation',
  SUPPORT: 'support',
  EDUCATION: 'education',
  CELEBRATION: 'celebration'
} as const;

// Version information
export const VERSION = '1.0.0';
export const API_VERSION = 'v1';

// Default configurations
export const DEFAULT_CONFIG = {
  database: {
    min_pool_size: 10,
    max_pool_size: 50,
    command_timeout: 60
  },
  cache: {
    l1_max_size: 1000,
    l1_ttl: 300,
    l2_ttl: 3600,
    cache_warming: true
  },
  stream: {
    max_messages: 100,
    ack_deadline: 60
  }
} as const;