-- User Whisperer Database Initialization
-- This script sets up the basic database structure

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS events;
CREATE SCHEMA IF NOT EXISTS users;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Grant permissions
GRANT ALL ON SCHEMA events TO uwdev;
GRANT ALL ON SCHEMA users TO uwdev;
GRANT ALL ON SCHEMA analytics TO uwdev;

-- Create basic tables
CREATE TABLE IF NOT EXISTS events.raw_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(32) UNIQUE NOT NULL,
    app_id VARCHAR(32) NOT NULL,
    user_id VARCHAR(32) NOT NULL,
    session_id VARCHAR(32),
    event_type VARCHAR(100) NOT NULL,
    properties JSONB DEFAULT '{}',
    context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_user_id ON events.raw_events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_app_id ON events.raw_events(app_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events.raw_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events.raw_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON events.raw_events(created_at);

-- User profiles table
CREATE TABLE IF NOT EXISTS users.profiles (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(32) UNIQUE NOT NULL,
    app_id VARCHAR(32) NOT NULL,
    profile_data JSONB DEFAULT '{}',
    behavioral_metrics JSONB DEFAULT '{}',
    ml_features JSONB DEFAULT '{}',
    lifecycle_stage VARCHAR(50) DEFAULT 'onboarding',
    segment VARCHAR(50) DEFAULT 'new_user',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_profiles_user_id ON users.profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_profiles_app_id ON users.profiles(app_id);
CREATE INDEX IF NOT EXISTS idx_profiles_lifecycle ON users.profiles(lifecycle_stage);
CREATE INDEX IF NOT EXISTS idx_profiles_segment ON users.profiles(segment);

-- Quarantined events table
CREATE TABLE IF NOT EXISTS events.quarantined_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(32) NOT NULL,
    data JSONB NOT NULL,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quarantined_created_at ON events.quarantined_events(created_at);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for user profiles
DROP TRIGGER IF EXISTS update_profiles_updated_at ON users.profiles;
CREATE TRIGGER update_profiles_updated_at
    BEFORE UPDATE ON users.profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create initial admin user for development
INSERT INTO users.profiles (user_id, app_id, profile_data, lifecycle_stage)
VALUES ('usr_admin_dev_123456', 'app_dev_123456', '{"name": "Dev Admin", "email": "admin@dev.local"}', 'activated')
ON CONFLICT (user_id) DO NOTHING;

-- Create analytics views
CREATE OR REPLACE VIEW analytics.daily_events AS
SELECT 
    DATE_TRUNC('day', timestamp) as day,
    app_id,
    event_type,
    COUNT(*) as event_count,
    COUNT(DISTINCT user_id) as unique_users
FROM events.raw_events
GROUP BY DATE_TRUNC('day', timestamp), app_id, event_type;

CREATE OR REPLACE VIEW analytics.user_activity_summary AS
SELECT 
    user_id,
    app_id,
    COUNT(*) as total_events,
    COUNT(DISTINCT event_type) as unique_event_types,
    COUNT(DISTINCT DATE_TRUNC('day', timestamp)) as active_days,
    MIN(timestamp) as first_event,
    MAX(timestamp) as last_event
FROM events.raw_events
GROUP BY user_id, app_id;

COMMIT; 