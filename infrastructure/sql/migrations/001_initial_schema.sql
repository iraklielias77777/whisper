-- Migration: 001_initial_schema.sql
-- Description: Initial database schema setup for User Whisperer Platform
-- Version: 1.0.0
-- Created: 2025-01-15

-- This migration sets up the complete database schema
-- Run this file after creating the database and extensions

-- =============================================================================
-- MIGRATION START
-- =============================================================================

BEGIN;

-- Create a migration tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT NOW(),
    description TEXT
);

-- Record this migration
INSERT INTO schema_migrations (version, description) 
VALUES ('001', 'Initial schema setup with core tables, indexes, and functions')
ON CONFLICT (version) DO NOTHING;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS ml;

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Applications table
CREATE TABLE IF NOT EXISTS core.applications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    api_key VARCHAR(500) NOT NULL UNIQUE,
    api_secret VARCHAR(500) NOT NULL,
    webhook_url VARCHAR(500),
    webhook_secret VARCHAR(255),
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    tier VARCHAR(50) NOT NULL DEFAULT 'free',
    rate_limit INTEGER DEFAULT 1000,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_status CHECK (status IN ('active', 'suspended', 'deleted')),
    CONSTRAINT valid_tier CHECK (tier IN ('free', 'starter', 'professional', 'enterprise'))
);

-- User profiles table
CREATE TABLE IF NOT EXISTS core.user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    app_id UUID NOT NULL REFERENCES core.applications(id) ON DELETE CASCADE,
    external_user_id VARCHAR(255) NOT NULL,
    
    -- Personal Information (encrypted)
    email VARCHAR(500),
    email_hash VARCHAR(64),
    phone VARCHAR(100),
    phone_hash VARCHAR(64),
    name VARCHAR(255),
    
    -- Account Information
    subscription_status VARCHAR(50) DEFAULT 'free',
    subscription_plan VARCHAR(100),
    subscription_started_at TIMESTAMP,
    subscription_ends_at TIMESTAMP,
    trial_ends_at TIMESTAMP,
    
    -- Behavioral Metrics
    lifecycle_stage VARCHAR(50) NOT NULL DEFAULT 'new',
    engagement_score DECIMAL(3,2) CHECK (engagement_score >= 0 AND engagement_score <= 1),
    churn_risk_score DECIMAL(3,2) CHECK (churn_risk_score >= 0 AND churn_risk_score <= 1),
    ltv_prediction DECIMAL(10,2),
    upgrade_probability DECIMAL(3,2) CHECK (upgrade_probability >= 0 AND upgrade_probability <= 1),
    
    -- Communication Preferences
    channel_preferences JSONB DEFAULT '{"email": true, "sms": false, "push": true}'::jsonb,
    optimal_send_hours INTEGER[] DEFAULT ARRAY[10, 14, 20],
    message_frequency_limit INTEGER DEFAULT 7,
    preferred_language VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'UTC',
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMP,
    deleted_at TIMESTAMP,
    
    -- Constraints
    CONSTRAINT unique_app_user UNIQUE(app_id, external_user_id),
    CONSTRAINT valid_lifecycle_stage CHECK (
        lifecycle_stage IN ('new', 'onboarding', 'activated', 'engaged', 'power_user', 'at_risk', 'dormant', 'churned', 'reactivated')
    ),
    CONSTRAINT valid_subscription_status CHECK (
        subscription_status IN ('free', 'trial', 'active', 'past_due', 'canceled', 'paused')
    )
);

-- Events table (partitioned)
CREATE TABLE IF NOT EXISTS core.events (
    id UUID DEFAULT uuid_generate_v4(),
    app_id UUID NOT NULL,
    user_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_category VARCHAR(50),
    
    -- Event Data
    properties JSONB,
    context JSONB,
    
    -- Enrichment Data
    user_context JSONB,
    geo_data JSONB,
    device_data JSONB,
    session_metrics JSONB,
    
    -- Processing Metadata
    created_at TIMESTAMP NOT NULL,
    processed_at TIMESTAMP,
    processing_time_ms INTEGER,
    
    -- Partitioning key
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Message history table
CREATE TABLE IF NOT EXISTS core.message_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    app_id UUID NOT NULL,
    user_id UUID NOT NULL,
    message_id VARCHAR(255) NOT NULL,
    
    -- Message Details
    channel VARCHAR(50) NOT NULL,
    message_type VARCHAR(100) NOT NULL,
    subject VARCHAR(500),
    content TEXT,
    
    -- Delivery Information
    scheduled_at TIMESTAMP NOT NULL,
    sent_at TIMESTAMP,
    delivered_at TIMESTAMP,
    opened_at TIMESTAMP,
    clicked_at TIMESTAMP,
    
    -- Status and Metadata
    status VARCHAR(50) NOT NULL DEFAULT 'scheduled',
    provider_id VARCHAR(255),
    provider_response JSONB,
    error_message TEXT,
    
    -- Performance Metrics
    delivery_time_ms INTEGER,
    open_time_seconds INTEGER,
    click_time_seconds INTEGER,
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_channel CHECK (channel IN ('email', 'sms', 'push', 'webhook')),
    CONSTRAINT valid_status CHECK (status IN ('scheduled', 'sending', 'sent', 'delivered', 'failed', 'bounced', 'opened', 'clicked'))
);

-- =============================================================================
-- ANALYTICS TABLES
-- =============================================================================

-- Daily user metrics
CREATE TABLE IF NOT EXISTS analytics.user_metrics_daily (
    user_id UUID NOT NULL,
    date DATE NOT NULL,
    app_id UUID NOT NULL,
    
    -- Activity Metrics
    event_count INTEGER DEFAULT 0,
    session_count INTEGER DEFAULT 0,
    total_session_duration_seconds INTEGER DEFAULT 0,
    unique_event_types INTEGER DEFAULT 0,
    
    -- Engagement Metrics
    features_used TEXT[],
    pages_viewed INTEGER DEFAULT 0,
    actions_completed INTEGER DEFAULT 0,
    errors_encountered INTEGER DEFAULT 0,
    
    -- Communication Metrics
    messages_sent INTEGER DEFAULT 0,
    messages_opened INTEGER DEFAULT 0,
    messages_clicked INTEGER DEFAULT 0,
    
    -- Calculated Scores
    daily_engagement_score DECIMAL(3,2),
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (user_id, date)
);

-- Weekly user metrics
CREATE TABLE IF NOT EXISTS analytics.user_metrics_weekly (
    user_id UUID NOT NULL,
    week_start DATE NOT NULL,
    app_id UUID NOT NULL,
    
    -- Aggregated weekly metrics
    total_events INTEGER DEFAULT 0,
    total_sessions INTEGER DEFAULT 0,
    avg_session_duration_seconds DECIMAL(10,2),
    unique_features_used INTEGER DEFAULT 0,
    
    -- Communication metrics
    total_messages_sent INTEGER DEFAULT 0,
    message_open_rate DECIMAL(3,2),
    message_click_rate DECIMAL(3,2),
    
    -- Engagement trends
    weekly_engagement_score DECIMAL(3,2),
    engagement_trend VARCHAR(20),
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (user_id, week_start),
    CONSTRAINT valid_engagement_trend CHECK (engagement_trend IN ('increasing', 'stable', 'decreasing'))
);

-- =============================================================================
-- ML TABLES
-- =============================================================================

-- ML Features
CREATE TABLE IF NOT EXISTS ml.features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    app_id UUID NOT NULL,
    feature_set VARCHAR(100) NOT NULL,
    features JSONB NOT NULL,
    computed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Metadata
    model_version VARCHAR(50),
    computation_time_ms INTEGER
);

-- ML Predictions
CREATE TABLE IF NOT EXISTS ml.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    app_id UUID NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    prediction_value DECIMAL(10,4),
    prediction_confidence DECIMAL(3,2),
    prediction_data JSONB,
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP,
    
    CONSTRAINT valid_confidence CHECK (prediction_confidence >= 0 AND prediction_confidence <= 1)
);

-- =============================================================================
-- AUDIT TABLES
-- =============================================================================

-- Audit log
CREATE TABLE IF NOT EXISTS audit.audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Actor Information
    actor_type VARCHAR(50) NOT NULL,
    actor_id VARCHAR(255),
    actor_ip INET,
    
    -- Action Information
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    
    -- Change Data
    old_values JSONB,
    new_values JSONB,
    
    -- Metadata
    request_id UUID,
    session_id VARCHAR(255),
    user_agent TEXT,
    
    -- Compliance
    gdpr_relevant BOOLEAN DEFAULT FALSE,
    data_classification VARCHAR(20),
    
    CONSTRAINT valid_actor_type CHECK (actor_type IN ('system', 'user', 'admin', 'api')),
    CONSTRAINT valid_data_classification CHECK (data_classification IN ('public', 'internal', 'confidential', 'restricted'))
);

-- Data retention policies
CREATE TABLE IF NOT EXISTS audit.data_retention_policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    data_type VARCHAR(100) NOT NULL UNIQUE,
    retention_days INTEGER NOT NULL,
    deletion_strategy VARCHAR(50) NOT NULL,
    last_cleanup_at TIMESTAMP,
    next_cleanup_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_deletion_strategy CHECK (deletion_strategy IN ('hard_delete', 'soft_delete', 'anonymize')),
    CONSTRAINT positive_retention CHECK (retention_days > 0)
);

-- GDPR requests
CREATE TABLE IF NOT EXISTS audit.gdpr_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    app_id UUID NOT NULL,
    request_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    requested_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    data_export_url TEXT,
    notes TEXT,
    
    -- Processing details
    processed_tables TEXT[],
    records_affected INTEGER,
    
    CONSTRAINT valid_request_type CHECK (request_type IN ('access', 'deletion', 'portability', 'rectification')),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled'))
);

-- Record successful migration
UPDATE schema_migrations 
SET applied_at = NOW() 
WHERE version = '001';

COMMIT;

-- =============================================================================
-- MIGRATION END
-- ============================================================================= 