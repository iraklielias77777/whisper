import { config as dotenvConfig } from 'dotenv';
import { z } from 'zod';
import path from 'path';

// Load environment variables from .env file
dotenvConfig({ path: path.resolve(process.cwd(), '.env') });

// Define comprehensive configuration schema
const configSchema = z.object({
  // Environment
  NODE_ENV: z.enum(['development', 'staging', 'production']).default('development'),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
  ENVIRONMENT: z.string().default('development'),
  
  // Database Configuration
  POSTGRES_HOST: z.string().default('localhost'),
  POSTGRES_PORT: z.coerce.number().default(5432),
  POSTGRES_DB: z.string().default('userwhisperer_dev'),
  POSTGRES_USER: z.string().default('uwdev'),
  POSTGRES_PASSWORD: z.string().default('localdev123'),
  POSTGRES_URL: z.string().url().default('postgresql://uwdev:localdev123@localhost:5432/userwhisperer_dev'),
  POSTGRES_MAX_CONNECTIONS: z.coerce.number().default(20),
  
  // Redis Configuration
  REDIS_HOST: z.string().default('localhost'),
  REDIS_PORT: z.coerce.number().default(6379),
  REDIS_PASSWORD: z.string().optional(),
  REDIS_DB: z.coerce.number().default(0),
  REDIS_URL: z.string().url().default('redis://localhost:6379/0'),
  
  // Event Ingestion Service
  EVENT_INGESTION_PORT: z.coerce.number().default(3001),
  EVENT_INGESTION_HOST: z.string().default('0.0.0.0'),
  
  // Behavioral Analysis Service
  BEHAVIORAL_ANALYSIS_PORT: z.coerce.number().default(3002),
  BEHAVIORAL_ANALYSIS_HOST: z.string().default('0.0.0.0'),
  
  // Decision Engine Service
  DECISION_ENGINE_PORT: z.coerce.number().default(3003),
  DECISION_ENGINE_HOST: z.string().default('0.0.0.0'),
  
  // Content Generation Service
  CONTENT_GENERATION_PORT: z.coerce.number().default(3004),
  CONTENT_GENERATION_HOST: z.string().default('0.0.0.0'),
  
  // Channel Orchestration Service
  CHANNEL_ORCHESTRATION_PORT: z.coerce.number().default(3005),
  CHANNEL_ORCHESTRATION_HOST: z.string().default('0.0.0.0'),
  CHANNEL_ORCHESTRATOR_PORT: z.coerce.number().default(3005), // Alternative name
  
  // AI Orchestration Service
  AI_ORCHESTRATION_PORT: z.coerce.number().default(8085),
  AI_ORCHESTRATION_HOST: z.string().default('0.0.0.0'),
  
  // API Gateway
  API_GATEWAY_PORT: z.coerce.number().default(3000),
  API_GATEWAY_HOST: z.string().default('0.0.0.0'),
  
  // Database
  DATABASE_URL: z.string().default('postgresql://uwdev:localdev123@localhost:5432/userwhisperer_dev'),
  
  // Rate Limiting
  RATE_LIMIT_ENABLED: z.coerce.boolean().default(true),
  RATE_LIMIT_WINDOW_MS: z.coerce.number().default(60000), // 1 minute
  RATE_LIMIT_MAX_REQUESTS: z.coerce.number().default(1000),
  
  // Email Rate Limiting
  EMAIL_RATE_LIMIT_PER_HOUR: z.string().default('10'),
  EMAIL_RATE_LIMIT_PER_DAY: z.string().default('50'),
  EMAIL_GLOBAL_RATE_LIMIT_PER_MINUTE: z.string().default('100'),
  
  // SMS Rate Limiting
  SMS_RATE_LIMIT_PER_HOUR: z.string().default('5'),
  SMS_RATE_LIMIT_PER_DAY: z.string().default('20'),
  SMS_GLOBAL_RATE_LIMIT_PER_MINUTE: z.string().default('50'),
  
  // Push Rate Limiting
  PUSH_RATE_LIMIT_PER_HOUR: z.string().default('20'),
  PUSH_RATE_LIMIT_PER_DAY: z.string().default('100'),
  PUSH_GLOBAL_RATE_LIMIT_PER_MINUTE: z.string().default('200'),
  
  // Retry Configuration
  MAX_DELIVERY_RETRIES: z.string().default('3'),
  RETRY_BASE_DELAY_SECONDS: z.string().default('60'),
  RETRY_MAX_DELAY_SECONDS: z.string().default('3600'),
  RETRY_JITTER_FACTOR: z.string().default('0.1'),
  
  // Queue Configuration
  QUEUE_BATCH_SIZE: z.coerce.number().default(100),
  QUEUE_MAX_SIZE: z.coerce.number().default(10000),
  QUEUE_HIGH_WATER_MARK: z.coerce.number().default(8000),
  QUEUE_LOW_WATER_MARK: z.coerce.number().default(2000),
  
  // Event Processing
  EVENT_DEDUPLICATION_TTL: z.coerce.number().default(3600), // 1 hour
  EVENT_QUARANTINE_TTL: z.coerce.number().default(604800), // 1 week
  EVENT_BATCH_TIMEOUT_MS: z.coerce.number().default(5000),
  
  // Monitoring
  METRICS_ENABLED: z.coerce.boolean().default(true),
  PROMETHEUS_PORT: z.coerce.number().default(9090),
  HEALTH_CHECK_INTERVAL: z.coerce.number().default(30000),
  
  // External Services - Email
  SENDGRID_API_KEY: z.string().default('mock-sendgrid-key'),
  SENDGRID_FROM_EMAIL: z.string().default('noreply@userwhisperer.com'),
  SENDGRID_FROM_NAME: z.string().default('User Whisperer'),
  SENDGRID_TRACKING_DOMAIN: z.string().default('track.userwhisperer.com'),
  SENDGRID_WEBHOOK_URL: z.string().default('https://api.userwhisperer.com/webhooks/sendgrid'),
  
  // External Services - SMS
  TWILIO_ACCOUNT_SID: z.string().default('mock-twilio-sid'),
  TWILIO_AUTH_TOKEN: z.string().default('mock-twilio-token'),
  TWILIO_FROM_NUMBER: z.string().default('+1234567890'),
  TWILIO_MESSAGING_SERVICE_SID: z.string().optional(),
  TWILIO_STATUS_CALLBACK_URL: z.string().default('https://api.userwhisperer.com/webhooks/twilio'),
  
  // External Services - Push
  FIREBASE_SERVICE_ACCOUNT_PATH: z.string().default('./firebase-service-account.json'),
  FIREBASE_DRY_RUN: z.string().default('false'),
  
  // External Services - AI
  OPENAI_API_KEY: z.string().optional(),
  ANTHROPIC_API_KEY: z.string().optional(),
  
  // Security
  JWT_SECRET: z.string().default('dev-secret-key-change-in-production'),
  ENCRYPTION_KEY: z.string().default('dev-encryption-key-32-chars-long'),
  
  // Features
  FEATURE_ML_MODELS_ENABLED: z.coerce.boolean().default(false),
  FEATURE_REAL_TIME_PROCESSING: z.coerce.boolean().default(true),
  FEATURE_BATCH_PROCESSING: z.coerce.boolean().default(true),
});

// Parse and validate configuration
const Config = configSchema.parse(process.env);

export default Config;
export { Config }; 