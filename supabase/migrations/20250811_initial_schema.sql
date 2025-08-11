-- User Whisperer Platform Database Schema
-- Compatible with Supabase Edge Functions

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- User Events Table
CREATE TABLE IF NOT EXISTS user_events (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    properties JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    session_id VARCHAR(255),
    page_url TEXT,
    user_agent TEXT,
    ip_address INET,
    referrer TEXT,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User Profiles Table
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    properties JSONB DEFAULT '{}',
    first_seen_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Behavioral Metrics Table
CREATE TABLE IF NOT EXISTS behavioral_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_count INTEGER DEFAULT 0,
    page_views INTEGER DEFAULT 0,
    session_duration_avg NUMERIC DEFAULT 0,
    bounce_rate NUMERIC DEFAULT 0,
    pages_per_session NUMERIC DEFAULT 0,
    feature_usage_depth NUMERIC DEFAULT 0,
    new_feature_adoption NUMERIC DEFAULT 0,
    feature_stickiness NUMERIC DEFAULT 0,
    most_used_features JSONB DEFAULT '[]',
    churn_risk_score NUMERIC DEFAULT 0,
    engagement_score NUMERIC DEFAULT 0,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Content Generation Table
CREATE TABLE IF NOT EXISTS generated_content (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    template_id VARCHAR(100),
    content JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    quality_score NUMERIC,
    status VARCHAR(20) DEFAULT 'generated',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Message Delivery Table
CREATE TABLE IF NOT EXISTS message_deliveries (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    content_id UUID REFERENCES generated_content(id),
    channel VARCHAR(20) NOT NULL, -- email, sms, push
    delivery_status VARCHAR(20) DEFAULT 'pending',
    delivery_attempts INTEGER DEFAULT 0,
    last_attempt_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Decision Log Table
CREATE TABLE IF NOT EXISTS decision_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    decision_type VARCHAR(50) NOT NULL,
    strategy JSONB NOT NULL,
    context JSONB DEFAULT '{}',
    result JSONB DEFAULT '{}',
    confidence_score NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_events_user_id ON user_events(user_id);
CREATE INDEX IF NOT EXISTS idx_user_events_timestamp ON user_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_events_event_type ON user_events(event_type);
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_behavioral_metrics_user_id ON behavioral_metrics(user_id);
CREATE INDEX IF NOT EXISTS idx_generated_content_user_id ON generated_content(user_id);
CREATE INDEX IF NOT EXISTS idx_message_deliveries_user_id ON message_deliveries(user_id);
CREATE INDEX IF NOT EXISTS idx_message_deliveries_status ON message_deliveries(delivery_status);
CREATE INDEX IF NOT EXISTS idx_decision_logs_user_id ON decision_logs(user_id);

-- Row Level Security (RLS) Policies
ALTER TABLE user_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE behavioral_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE generated_content ENABLE ROW LEVEL SECURITY;
ALTER TABLE message_deliveries ENABLE ROW LEVEL SECURITY;
ALTER TABLE decision_logs ENABLE ROW LEVEL SECURITY;

-- Service role policies (full access for Edge Functions)
CREATE POLICY "Enable all operations for service role" ON user_events
FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON user_profiles
FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON behavioral_metrics
FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON generated_content
FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON message_deliveries
FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON decision_logs
FOR ALL USING (auth.role() = 'service_role');
