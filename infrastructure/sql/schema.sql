-- User Whisperer Platform - Complete PostgreSQL Schema
-- Version: 1.0
-- Created: 2025-01-15

-- Database creation and configuration
-- Note: Run this as superuser to create database
-- CREATE DATABASE userwhisperer
--     WITH 
--     OWNER = uwadmin
--     ENCODING = 'UTF8'
--     LC_COLLATE = 'en_US.utf8'
--     LC_CTYPE = 'en_US.utf8'
--     TABLESPACE = pg_default
--     CONNECTION LIMIT = -1;

-- Connect to userwhisperer database
-- \c userwhisperer

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "postgres_fdw";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_cron";

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS ml;

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Application Registry
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
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for applications
CREATE INDEX IF NOT EXISTS idx_applications_api_key ON core.applications(api_key);
CREATE INDEX IF NOT EXISTS idx_applications_status ON core.applications(status);

-- User Profiles Table
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
    message_frequency_limit INTEGER DEFAULT 7, -- per week
    preferred_language VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'UTC',
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMP,
    deleted_at TIMESTAMP, -- Soft delete for GDPR
    
    -- Constraints
    CONSTRAINT unique_app_user UNIQUE(app_id, external_user_id),
    CONSTRAINT valid_lifecycle_stage CHECK (
        lifecycle_stage IN ('new', 'onboarding', 'activated', 'engaged', 'power_user', 'at_risk', 'dormant', 'churned', 'reactivated')
    ),
    CONSTRAINT valid_subscription_status CHECK (
        subscription_status IN ('free', 'trial', 'active', 'past_due', 'canceled', 'paused')
    )
);

-- Indexes for user_profiles
CREATE INDEX IF NOT EXISTS idx_user_profiles_app_id ON core.user_profiles(app_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_external_user_id ON core.user_profiles(external_user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_email_hash ON core.user_profiles(email_hash) WHERE email_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_profiles_phone_hash ON core.user_profiles(phone_hash) WHERE phone_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_profiles_lifecycle_stage ON core.user_profiles(app_id, lifecycle_stage);
CREATE INDEX IF NOT EXISTS idx_user_profiles_churn_risk ON core.user_profiles(app_id, churn_risk_score) WHERE churn_risk_score > 0.5;
CREATE INDEX IF NOT EXISTS idx_user_profiles_active ON core.user_profiles(app_id, last_active_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_user_profiles_subscription ON core.user_profiles(app_id, subscription_status);
CREATE INDEX IF NOT EXISTS idx_user_profiles_engagement ON core.user_profiles(app_id, engagement_score DESC) WHERE engagement_score IS NOT NULL;

-- Events Table (Partitioned by date)
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

-- Create monthly partitions for events (only if they don't exist)
DO $$
DECLARE
    start_date date := '2025-01-01';
    end_date date;
    partition_name text;
    partition_exists boolean;
BEGIN
    FOR i IN 0..23 LOOP -- 2 years of partitions
        end_date := start_date + interval '1 month';
        partition_name := 'events_' || to_char(start_date, 'YYYY_MM');
        
        -- Check if partition exists
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_schema = 'core' AND table_name = partition_name
        ) INTO partition_exists;
        
        IF NOT partition_exists THEN
            EXECUTE format('
                CREATE TABLE core.%I PARTITION OF core.events
                FOR VALUES FROM (%L) TO (%L)',
                partition_name,
                start_date,
                end_date
            );
            
            -- Create indexes on partition
            EXECUTE format('
                CREATE INDEX %I ON core.%I (app_id, user_id, created_at DESC)',
                'idx_' || partition_name || '_app_user',
                partition_name
            );
            
            EXECUTE format('
                CREATE INDEX %I ON core.%I (event_type, created_at DESC)',
                'idx_' || partition_name || '_type',
                partition_name
            );
            
            EXECUTE format('
                CREATE INDEX %I ON core.%I USING GIN (properties)',
                'idx_' || partition_name || '_properties',
                partition_name
            );
            
            EXECUTE format('
                CREATE INDEX %I ON core.%I (user_id, event_type, created_at)',
                'idx_' || partition_name || '_user_type',
                partition_name
            );
        END IF;
        
        start_date := end_date;
    END LOOP;
END $$;

-- Message History Table
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

-- Indexes for message_history
CREATE INDEX IF NOT EXISTS idx_message_history_app_user ON core.message_history(app_id, user_id, scheduled_at DESC);
CREATE INDEX IF NOT EXISTS idx_message_history_message_id ON core.message_history(message_id);
CREATE INDEX IF NOT EXISTS idx_message_history_status ON core.message_history(status, scheduled_at);
CREATE INDEX IF NOT EXISTS idx_message_history_channel ON core.message_history(channel, scheduled_at DESC);

-- =============================================================================
-- PERFORMANCE OPTIMIZATION TABLES
-- =============================================================================

-- Aggregated Metrics Table (for fast queries)
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

-- Create index for fast lookups
CREATE INDEX IF NOT EXISTS idx_user_metrics_daily_date ON analytics.user_metrics_daily(date, app_id);
CREATE INDEX IF NOT EXISTS idx_user_metrics_daily_app ON analytics.user_metrics_daily(app_id, date DESC);

-- Weekly aggregations
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
    engagement_trend VARCHAR(20), -- 'increasing', 'stable', 'decreasing'
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (user_id, week_start),
    CONSTRAINT valid_engagement_trend CHECK (engagement_trend IN ('increasing', 'stable', 'decreasing'))
);

-- Materialized View for Active Users Summary
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.active_users_summary AS
SELECT 
    app_id,
    lifecycle_stage,
    COUNT(*) as user_count,
    AVG(engagement_score) as avg_engagement,
    AVG(churn_risk_score) as avg_churn_risk,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ltv_prediction) as median_ltv,
    COUNT(*) FILTER (WHERE last_active_at > NOW() - INTERVAL '1 day') as daily_active,
    COUNT(*) FILTER (WHERE last_active_at > NOW() - INTERVAL '7 days') as weekly_active,
    COUNT(*) FILTER (WHERE last_active_at > NOW() - INTERVAL '30 days') as monthly_active
FROM core.user_profiles
WHERE deleted_at IS NULL
GROUP BY app_id, lifecycle_stage;

-- Create unique index on materialized view (only if it doesn't exist)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE schemaname = 'analytics' AND tablename = 'active_users_summary' AND indexname = 'idx_active_users_summary') THEN
        CREATE UNIQUE INDEX idx_active_users_summary ON analytics.active_users_summary(app_id, lifecycle_stage);
    END IF;
END $$;

-- =============================================================================
-- MACHINE LEARNING TABLES
-- =============================================================================

-- ML Features Storage
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

CREATE INDEX IF NOT EXISTS idx_ml_features_user ON ml.features(user_id, feature_set, computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_ml_features_app ON ml.features(app_id, feature_set, computed_at DESC);

-- Model Predictions
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

CREATE INDEX IF NOT EXISTS idx_ml_predictions_user ON ml.predictions(user_id, model_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_type ON ml.predictions(prediction_type, created_at DESC);

-- Model Performance Tracking
CREATE TABLE IF NOT EXISTS ml.model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    evaluation_date DATE NOT NULL,
    
    -- Additional context
    dataset_size INTEGER,
    evaluation_notes TEXT,
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- AUDIT AND COMPLIANCE TABLES
-- =============================================================================

-- Audit Log Table
CREATE TABLE IF NOT EXISTS audit.audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Actor Information
    actor_type VARCHAR(50) NOT NULL, -- 'system', 'user', 'admin', 'api'
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
    data_classification VARCHAR(20), -- 'public', 'internal', 'confidential', 'restricted'
    
    CONSTRAINT valid_actor_type CHECK (actor_type IN ('system', 'user', 'admin', 'api')),
    CONSTRAINT valid_data_classification CHECK (data_classification IN ('public', 'internal', 'confidential', 'restricted'))
);

-- Partition audit log by month for performance
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit.audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_actor ON audit.audit_log(actor_id) WHERE actor_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_audit_log_resource ON audit.audit_log(resource_type, resource_id) WHERE resource_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_audit_log_gdpr ON audit.audit_log(gdpr_relevant, timestamp) WHERE gdpr_relevant = TRUE;
CREATE INDEX IF NOT EXISTS idx_audit_log_request ON audit.audit_log(request_id) WHERE request_id IS NOT NULL;

-- Data Retention Policy Table
CREATE TABLE IF NOT EXISTS audit.data_retention_policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    data_type VARCHAR(100) NOT NULL UNIQUE,
    retention_days INTEGER NOT NULL,
    deletion_strategy VARCHAR(50) NOT NULL, -- 'hard_delete', 'soft_delete', 'anonymize'
    last_cleanup_at TIMESTAMP,
    next_cleanup_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_deletion_strategy CHECK (deletion_strategy IN ('hard_delete', 'soft_delete', 'anonymize')),
    CONSTRAINT positive_retention CHECK (retention_days > 0)
);

-- Insert default retention policies (only if table is empty)
INSERT INTO audit.data_retention_policies (data_type, retention_days, deletion_strategy) 
SELECT * FROM (VALUES
    ('events', 365, 'hard_delete'),
    ('user_profiles', 1095, 'anonymize'), -- 3 years
    ('message_history', 180, 'hard_delete'),
    ('audit_log', 2555, 'hard_delete'), -- 7 years for compliance
    ('ml_predictions', 30, 'hard_delete'),
    ('ml_features', 7, 'hard_delete'),
    ('user_metrics_daily', 395, 'hard_delete'), -- 13 months
    ('user_metrics_weekly', 730, 'hard_delete') -- 2 years
) AS t(data_type, retention_days, deletion_strategy)
WHERE NOT EXISTS (SELECT 1 FROM audit.data_retention_policies);

-- GDPR Request Tracking
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

CREATE INDEX IF NOT EXISTS idx_gdpr_requests_user ON audit.gdpr_requests(user_id, requested_at DESC);
CREATE INDEX IF NOT EXISTS idx_gdpr_requests_status ON audit.gdpr_requests(status, requested_at);

-- Lifecycle Events Tracking
CREATE TABLE IF NOT EXISTS audit.lifecycle_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    data_type VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,
    affected_records INTEGER DEFAULT 0,
    details JSONB,
    
    CONSTRAINT valid_action CHECK (action IN ('retain', 'archive', 'anonymize', 'delete', 'migrate'))
);

CREATE INDEX IF NOT EXISTS idx_lifecycle_events_timestamp ON audit.lifecycle_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_lifecycle_events_type ON audit.lifecycle_events(data_type, timestamp DESC);

-- Lifecycle Reports
CREATE TABLE IF NOT EXISTS audit.lifecycle_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_date TIMESTAMP NOT NULL DEFAULT NOW(),
    period VARCHAR(20) NOT NULL,
    report_data JSONB NOT NULL,
    
    CONSTRAINT valid_period CHECK (period IN ('daily', 'weekly', 'monthly', 'quarterly'))
);

CREATE INDEX IF NOT EXISTS idx_lifecycle_reports_date ON audit.lifecycle_reports(report_date DESC);

-- =============================================================================
-- FUNCTIONS AND TRIGGERS
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to relevant tables (only if they don't exist)
DO $$
BEGIN
    -- user_profiles trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_user_profiles_updated_at') THEN
        CREATE TRIGGER update_user_profiles_updated_at 
        BEFORE UPDATE ON core.user_profiles 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
    
    -- applications trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_applications_updated_at') THEN
        CREATE TRIGGER update_applications_updated_at 
        BEFORE UPDATE ON core.applications 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
    
    -- message_history trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_message_history_updated_at') THEN
        CREATE TRIGGER update_message_history_updated_at 
        BEFORE UPDATE ON core.message_history 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
    
    -- user_metrics_daily trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_user_metrics_daily_updated_at') THEN
        CREATE TRIGGER update_user_metrics_daily_updated_at 
        BEFORE UPDATE ON analytics.user_metrics_daily 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
    
    -- retention_policies trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_retention_policies_updated_at') THEN
        CREATE TRIGGER update_retention_policies_updated_at 
        BEFORE UPDATE ON audit.data_retention_policies 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_active_users_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.active_users_summary;
    
    -- Log the refresh
    INSERT INTO audit.audit_log (
        actor_type, action, resource_type,
        old_values, new_values
    ) VALUES (
        'system', 'refresh_materialized_view', 'active_users_summary',
        '{"action": "scheduled_refresh"}'::jsonb,
        '{"completed_at": "' || NOW() || '"}'::jsonb
    );
END;
$$ LANGUAGE plpgsql;

-- Function for cleanup old data based on retention policies
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS INTEGER AS $$
DECLARE
    policy RECORD;
    cleanup_count INTEGER := 0;
    total_deleted INTEGER := 0;
BEGIN
    FOR policy IN 
        SELECT * FROM audit.data_retention_policies 
        WHERE is_active = TRUE 
        AND (next_cleanup_at IS NULL OR next_cleanup_at <= NOW())
    LOOP
        CASE policy.data_type
            WHEN 'events' THEN
                -- Delete old events
                DELETE FROM core.events 
                WHERE created_at < NOW() - (policy.retention_days || ' days')::interval;
                GET DIAGNOSTICS cleanup_count = ROW_COUNT;
                
            WHEN 'message_history' THEN
                -- Delete old message history
                DELETE FROM core.message_history 
                WHERE created_at < NOW() - (policy.retention_days || ' days')::interval;
                GET DIAGNOSTICS cleanup_count = ROW_COUNT;
                
            WHEN 'audit_log' THEN
                -- Delete old audit logs (non-GDPR)
                DELETE FROM audit.audit_log 
                WHERE timestamp < NOW() - (policy.retention_days || ' days')::interval
                AND gdpr_relevant = FALSE;
                GET DIAGNOSTICS cleanup_count = ROW_COUNT;
                
        END CASE;
        
        total_deleted := total_deleted + cleanup_count;
        
        -- Update policy
        UPDATE audit.data_retention_policies 
        SET 
            last_cleanup_at = NOW(),
            next_cleanup_at = NOW() + INTERVAL '1 day'
        WHERE id = policy.id;
        
    END LOOP;
    
    RETURN total_deleted;
END;
$$ LANGUAGE plpgsql;

-- Schedule periodic jobs (only if pg_cron extension is available)
DO $$
BEGIN
    -- Check if pg_cron extension is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        -- Refresh materialized views every 15 minutes
        PERFORM cron.schedule(
            'refresh-active-users',
            '*/15 * * * *',
            'SELECT refresh_active_users_summary();'
        );
        
        -- Cleanup expired data daily at 2 AM
        PERFORM cron.schedule(
            'cleanup-expired-data',
            '0 2 * * *',
            'SELECT cleanup_expired_data();'
        );
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        -- pg_cron not available, continue without scheduling
        NULL;
END $$;

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- User engagement summary view
CREATE OR REPLACE VIEW analytics.user_engagement_summary AS
SELECT 
    up.id,
    up.app_id,
    up.external_user_id,
    up.lifecycle_stage,
    up.engagement_score,
    up.churn_risk_score,
    up.last_active_at,
    
    -- Recent activity (last 7 days)
    COALESCE(recent.event_count, 0) as recent_events,
    COALESCE(recent.session_count, 0) as recent_sessions,
    
    -- Communication metrics (last 30 days)
    COALESCE(comm.messages_sent, 0) as recent_messages,
    COALESCE(comm.messages_opened, 0) as recent_opens,
    CASE 
        WHEN comm.messages_sent > 0 
        THEN ROUND((comm.messages_opened::decimal / comm.messages_sent) * 100, 2)
        ELSE 0 
    END as open_rate_pct
    
FROM core.user_profiles up
LEFT JOIN (
    SELECT 
        user_id,
        SUM(event_count) as event_count,
        SUM(session_count) as session_count
    FROM analytics.user_metrics_daily 
    WHERE date >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY user_id
) recent ON up.id = recent.user_id
LEFT JOIN (
    SELECT 
        user_id,
        SUM(messages_sent) as messages_sent,
        SUM(messages_opened) as messages_opened
    FROM analytics.user_metrics_daily 
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY user_id
) comm ON up.id = comm.user_id
WHERE up.deleted_at IS NULL;

-- Message performance view
CREATE OR REPLACE VIEW analytics.message_performance AS
SELECT 
    mh.channel,
    mh.message_type,
    DATE(mh.scheduled_at) as send_date,
    COUNT(*) as total_sent,
    COUNT(*) FILTER (WHERE status = 'delivered') as delivered,
    COUNT(*) FILTER (WHERE opened_at IS NOT NULL) as opened,
    COUNT(*) FILTER (WHERE clicked_at IS NOT NULL) as clicked,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    
    -- Rates
    ROUND(
        (COUNT(*) FILTER (WHERE status = 'delivered')::decimal / COUNT(*)) * 100, 2
    ) as delivery_rate_pct,
    ROUND(
        (COUNT(*) FILTER (WHERE opened_at IS NOT NULL)::decimal / 
         COUNT(*) FILTER (WHERE status = 'delivered')) * 100, 2
    ) as open_rate_pct,
    ROUND(
        (COUNT(*) FILTER (WHERE clicked_at IS NOT NULL)::decimal / 
         COUNT(*) FILTER (WHERE opened_at IS NOT NULL)) * 100, 2
    ) as click_rate_pct
    
FROM core.message_history mh
WHERE mh.scheduled_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY mh.channel, mh.message_type, DATE(mh.scheduled_at)
ORDER BY send_date DESC;

-- Grant permissions
GRANT USAGE ON SCHEMA core TO uwapp;
GRANT USAGE ON SCHEMA analytics TO uwapp;
GRANT USAGE ON SCHEMA audit TO uwapp;
GRANT USAGE ON SCHEMA ml TO uwapp;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA core TO uwapp;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA analytics TO uwapp;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO uwapp;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ml TO uwapp;

GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO uwreadonly;
GRANT SELECT ON ALL TABLES IN SCHEMA audit TO uwreadonly;

-- Grant sequence permissions
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA core TO uwapp;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analytics TO uwapp;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO uwapp;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ml TO uwapp;