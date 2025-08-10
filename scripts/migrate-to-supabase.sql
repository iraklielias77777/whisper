-- User Whisperer Platform - Supabase Migration Script
-- Run this in your Supabase SQL editor or via CLI

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS events;

-- =====================================================
-- CORE USER & EVENT TABLES
-- =====================================================

-- User profiles table
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_user_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255),
    name VARCHAR(255),
    phone_number VARCHAR(50),
    timezone VARCHAR(50) DEFAULT 'UTC',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    email_verified BOOLEAN DEFAULT FALSE,
    phone_verified BOOLEAN DEFAULT FALSE,
    subscription_plan VARCHAR(50) DEFAULT 'free',
    last_seen_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- User events table
CREATE TABLE IF NOT EXISTS user_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    session_id VARCHAR(255),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    source VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User behavioral metrics
CREATE TABLE IF NOT EXISTS user_behavioral_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    engagement_score DECIMAL(5,4) DEFAULT 0,
    churn_probability DECIMAL(5,4) DEFAULT 0,
    feature_usage JSONB DEFAULT '{}'::jsonb,
    communication_preferences JSONB DEFAULT '{}'::jsonb,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- MESSAGING & COMMUNICATION TABLES
-- =====================================================

-- Message history
CREATE TABLE IF NOT EXISTS message_history (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    channel VARCHAR(50) NOT NULL, -- email, sms, push, webhook
    message_type VARCHAR(100) NOT NULL,
    content JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending', -- pending, sent, delivered, failed
    sent_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    clicked_at TIMESTAMPTZ,
    provider_message_id VARCHAR(255),
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User unsubscriptions
CREATE TABLE IF NOT EXISTS user_unsubscriptions (
    unsubscription_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    channel VARCHAR(50) NOT NULL, -- email, sms, push, all
    reason VARCHAR(255),
    unsubscribed_at TIMESTAMPTZ DEFAULT NOW()
);

-- User push tokens
CREATE TABLE IF NOT EXISTS user_push_tokens (
    token_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    token VARCHAR(500) NOT NULL,
    platform VARCHAR(50) NOT NULL, -- ios, android, web
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

-- =====================================================
-- AI & ML TABLES
-- =====================================================

-- AI orchestration results
CREATE TABLE IF NOT EXISTS ai_orchestration_results (
    orchestration_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    trigger_event VARCHAR(100) NOT NULL,
    strategy_decisions JSONB NOT NULL,
    ai_insights JSONB NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    models_used TEXT[] DEFAULT '{}',
    processing_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ML model predictions
CREATE TABLE IF NOT EXISTS ml_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    prediction_type VARCHAR(100) NOT NULL,
    input_features JSONB NOT NULL,
    prediction_value DECIMAL(10,6),
    prediction_class VARCHAR(100),
    confidence DECIMAL(5,4),
    model_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Content experiments (A/B testing)
CREATE TABLE IF NOT EXISTS content_experiments (
    experiment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_name VARCHAR(255) NOT NULL,
    variant_name VARCHAR(100) NOT NULL,
    content_data JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'active', -- active, paused, completed
    success_metric VARCHAR(100),
    target_audience JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

-- Experiment assignments
CREATE TABLE IF NOT EXISTS experiment_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    experiment_id UUID REFERENCES content_experiments(experiment_id),
    variant_name VARCHAR(100) NOT NULL,
    assigned_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- PERFORMANCE & ANALYTICS TABLES
-- =====================================================

-- Delivery status tracking
CREATE TABLE IF NOT EXISTS delivery_status (
    status_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) NOT NULL,
    provider_message_id VARCHAR(255),
    delivered_at TIMESTAMPTZ,
    error TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- System performance metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    labels JSONB DEFAULT '{}'::jsonb,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- User profiles indexes
CREATE INDEX IF NOT EXISTS idx_user_profiles_external_id ON user_profiles(external_user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON user_profiles(email);
CREATE INDEX IF NOT EXISTS idx_user_profiles_created_at ON user_profiles(created_at);

-- User events indexes
CREATE INDEX IF NOT EXISTS idx_user_events_user_id ON user_events(user_id);
CREATE INDEX IF NOT EXISTS idx_user_events_type ON user_events(event_type);
CREATE INDEX IF NOT EXISTS idx_user_events_timestamp ON user_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_events_session ON user_events(session_id);
CREATE INDEX IF NOT EXISTS idx_user_events_user_type ON user_events(user_id, event_type);

-- Message history indexes
CREATE INDEX IF NOT EXISTS idx_message_history_user_id ON message_history(user_id);
CREATE INDEX IF NOT EXISTS idx_message_history_channel ON message_history(channel);
CREATE INDEX IF NOT EXISTS idx_message_history_status ON message_history(status);
CREATE INDEX IF NOT EXISTS idx_message_history_sent_at ON message_history(sent_at);

-- AI orchestration indexes
CREATE INDEX IF NOT EXISTS idx_ai_orchestration_user_id ON ai_orchestration_results(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_orchestration_trigger ON ai_orchestration_results(trigger_event);
CREATE INDEX IF NOT EXISTS idx_ai_orchestration_created_at ON ai_orchestration_results(created_at);

-- ML predictions indexes
CREATE INDEX IF NOT EXISTS idx_ml_predictions_user_id ON ml_predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON ml_predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_created_at ON ml_predictions(created_at);

-- =====================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================

-- Enable RLS on sensitive tables
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE message_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_behavioral_metrics ENABLE ROW LEVEL SECURITY;

-- Service role can access everything
CREATE POLICY "Service role has full access" ON user_profiles
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role has full access" ON user_events
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role has full access" ON message_history
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role has full access" ON user_behavioral_metrics
    FOR ALL USING (auth.role() = 'service_role');

-- Anon role can only insert events (for SDK)
CREATE POLICY "Anonymous can insert events" ON user_events
    FOR INSERT WITH CHECK (true);

-- =====================================================
-- FUNCTIONS & TRIGGERS
-- =====================================================

-- Function to update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_user_profiles_updated_at 
    BEFORE UPDATE ON user_profiles 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_behavioral_metrics_updated_at 
    BEFORE UPDATE ON user_behavioral_metrics 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate engagement score
CREATE OR REPLACE FUNCTION calculate_engagement_score(p_user_id VARCHAR(255))
RETURNS DECIMAL(5,4) AS $$
DECLARE
    score DECIMAL(5,4) := 0;
    event_count INTEGER;
    recent_events INTEGER;
BEGIN
    -- Count total events
    SELECT COUNT(*) INTO event_count 
    FROM user_events 
    WHERE user_id = p_user_id;
    
    -- Count recent events (last 30 days)
    SELECT COUNT(*) INTO recent_events 
    FROM user_events 
    WHERE user_id = p_user_id 
    AND timestamp > NOW() - INTERVAL '30 days';
    
    -- Simple engagement calculation
    score := LEAST(1.0, (recent_events::DECIMAL / GREATEST(1, event_count)) * 2);
    
    RETURN score;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- INITIAL DATA & CONFIGURATION
-- =====================================================

-- Insert default system configuration
CREATE TABLE IF NOT EXISTS system_config (
    config_key VARCHAR(255) PRIMARY KEY,
    config_value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO system_config (config_key, config_value, description) VALUES
('ai_orchestration_enabled', 'true', 'Enable AI orchestration features'),
('default_rate_limits', '{"email": {"per_hour": 10, "per_day": 50}, "sms": {"per_hour": 5, "per_day": 20}, "push": {"per_hour": 20, "per_day": 100}}', 'Default rate limits per channel'),
('ml_models_enabled', 'true', 'Enable machine learning models'),
('content_experimentation_enabled', 'true', 'Enable A/B testing features')
ON CONFLICT (config_key) DO NOTHING;

-- Create API keys table for SDK authentication
CREATE TABLE IF NOT EXISTS api_keys (
    api_key_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_name VARCHAR(255) NOT NULL,
    api_key VARCHAR(255) UNIQUE NOT NULL,
    permissions TEXT[] DEFAULT '{}',
    rate_limit_tier VARCHAR(50) DEFAULT 'standard',
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

-- Insert a test API key
INSERT INTO api_keys (key_name, api_key, permissions) VALUES
('Test API Key', 'test-api-key-12345', ARRAY['events:write', 'analytics:read'])
ON CONFLICT (api_key) DO NOTHING;

-- =====================================================
-- VIEWS FOR ANALYTICS
-- =====================================================

-- User engagement summary view
CREATE OR REPLACE VIEW user_engagement_summary AS
SELECT 
    up.user_id,
    up.external_user_id,
    up.email,
    up.subscription_plan,
    ubm.engagement_score,
    ubm.churn_probability,
    COUNT(ue.event_id) as total_events,
    COUNT(CASE WHEN ue.timestamp > NOW() - INTERVAL '7 days' THEN 1 END) as events_last_7_days,
    COUNT(CASE WHEN ue.timestamp > NOW() - INTERVAL '30 days' THEN 1 END) as events_last_30_days,
    MAX(ue.timestamp) as last_event_at,
    up.created_at,
    up.last_seen_at
FROM user_profiles up
LEFT JOIN user_behavioral_metrics ubm ON up.external_user_id = ubm.user_id
LEFT JOIN user_events ue ON up.external_user_id = ue.user_id
GROUP BY up.user_id, up.external_user_id, up.email, up.subscription_plan, 
         ubm.engagement_score, ubm.churn_probability, up.created_at, up.last_seen_at;

-- Message performance view
CREATE OR REPLACE VIEW message_performance AS
SELECT 
    channel,
    DATE_TRUNC('day', sent_at) as date,
    COUNT(*) as messages_sent,
    COUNT(CASE WHEN status = 'delivered' THEN 1 END) as messages_delivered,
    COUNT(CASE WHEN opened_at IS NOT NULL THEN 1 END) as messages_opened,
    COUNT(CASE WHEN clicked_at IS NOT NULL THEN 1 END) as messages_clicked,
    ROUND(COUNT(CASE WHEN status = 'delivered' THEN 1 END)::DECIMAL / COUNT(*) * 100, 2) as delivery_rate,
    ROUND(COUNT(CASE WHEN opened_at IS NOT NULL THEN 1 END)::DECIMAL / COUNT(CASE WHEN status = 'delivered' THEN 1 END) * 100, 2) as open_rate,
    ROUND(COUNT(CASE WHEN clicked_at IS NOT NULL THEN 1 END)::DECIMAL / COUNT(CASE WHEN opened_at IS NOT NULL THEN 1 END) * 100, 2) as click_through_rate
FROM message_history
WHERE sent_at IS NOT NULL
GROUP BY channel, DATE_TRUNC('day', sent_at)
ORDER BY date DESC, channel;

COMMIT;

-- Grant permissions to service role
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO service_role;

-- Grant limited permissions to anon role (for SDK)
GRANT INSERT ON user_events TO anon;
GRANT SELECT ON api_keys TO anon;
