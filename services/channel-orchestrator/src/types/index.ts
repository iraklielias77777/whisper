export interface DeliveryRequest {
  message_id: string;
  user_id: string;
  channel: string;
  content: ContentPayload;
  send_time: Date;
  priority: number; // 1-5, 5 being highest
  retry_count?: number;
  max_retries?: number;
  metadata?: Record<string, any>;
}

export interface ContentPayload {
  // Email specific
  subject?: string;
  html_body?: string;
  text_body?: string;
  
  // SMS specific
  body?: string;
  
  // Push specific
  title?: string;
  image_url?: string;
  icon_url?: string;
  badge_url?: string;
  click_action?: string;
  deep_link?: string;
  custom_data?: Record<string, any>;
  badge?: number;
  
  // Common
  attachments?: AttachmentData[];
}

export interface AttachmentData {
  filename: string;
  content: Buffer;
  type: string;
}

export interface DeliveryResult {
  success: boolean;
  channel: string;
  provider_message_id?: string;
  delivered_at?: Date;
  error?: string;
  should_retry: boolean;
}

export interface ValidationResult {
  is_valid: boolean;
  errors: string[];
}

export interface RateLimitResult {
  allowed: boolean;
  limit: number;
  remaining: number;
  reset_time: Date;
}

export interface DeliveryMetrics {
  total_sent: number;
  total_delivered: number;
  total_failed: number;
  delivery_rate: number;
  average_delivery_time: number;
  by_channel: Record<string, ChannelMetrics>;
}

export interface ChannelMetrics {
  sent: number;
  delivered: number;
  failed: number;
  bounced: number;
  opened: number;
  clicked: number;
}

export interface UserContactInfo {
  user_id: string;
  email?: string;
  phone_number?: string;
  push_tokens?: string[];
  email_verified: boolean;
  phone_verified: boolean;
  unsubscribed_channels: string[];
  preferred_channels: string[];
  timezone?: string;
}

export interface ChannelOrchestratorConfig {
  redis_url: string;
  postgres_url: string;
  sendgrid: SendGridConfig;
  twilio: TwilioConfig;
  firebase: FirebaseConfig;
  rate_limits: RateLimitConfig;
  retry_settings: RetrySettings;
}

export interface SendGridConfig {
  api_key: string;
  from_email: string;
  from_name: string;
  tracking_domain?: string;
  webhook_url?: string;
}

export interface TwilioConfig {
  account_sid: string;
  auth_token: string;
  from_number: string;
  messaging_service_sid?: string;
  status_callback_url?: string;
}

export interface FirebaseConfig {
  service_account_path: string;
  dry_run?: boolean;
  firebase_service_account?: any;
  apns_key?: string;
  provider?: 'firebase' | 'apns' | 'web-push' | 'mock';
  web_push_vapid_keys?: {
    public_key: string;
    private_key: string;
    subject: string;
  };
}

export interface RateLimitConfig {
  email: {
    per_user_per_hour: number;
    per_user_per_day: number;
    global_per_minute: number;
  };
  sms: {
    per_user_per_hour: number;
    per_user_per_day: number;
    global_per_minute: number;
  };
  push: {
    per_user_per_hour: number;
    per_user_per_day: number;
    global_per_minute: number;
  };
}

export interface RetrySettings {
  max_retries: number;
  base_delay_seconds: number;
  max_delay_seconds: number;
  jitter_factor: number;
}

export interface QueuedMessage {
  request: DeliveryRequest;
  queued_at: Date;
  scheduled_for: Date;
}

export interface DeliveryStatus {
  message_id: string;
  status: 'queued' | 'sent' | 'delivered' | 'failed' | 'suppressed';
  provider_message_id?: string;
  sent_at?: Date;
  delivered_at?: Date;
  opened_at?: Date;
  clicked_at?: Date;
  error?: string;
  retry_count: number;
}

export interface WebhookEvent {
  type: string;
  message_id: string;
  provider_message_id: string;
  user_id: string;
  channel: string;
  timestamp: Date;
  data: Record<string, any>;
}

export interface SuppressionReason {
  type: 'unsubscribed' | 'invalid_contact' | 'duplicate' | 'business_hours' | 'rate_limit';
  message: string;
  timestamp: Date;
} 