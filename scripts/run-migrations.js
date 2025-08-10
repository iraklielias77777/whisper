#!/usr/bin/env node

/**
 * Database Migration Runner
 * Executes all database migrations and seeds initial data
 */

const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

class MigrationRunner {
  constructor() {
    this.connectionString = process.env.POSTGRES_URL || 'postgresql://uwdev:localdev123@localhost:5432/userwhisperer_dev';
    this.migrationsDir = path.join(process.cwd(), 'infrastructure', 'sql', 'migrations');
    this.seedsDir = path.join(process.cwd(), 'infrastructure', 'sql', 'seeds');
    this.client = null;
  }

  async run() {
    console.log('🗄️  Starting Database Migration Process...\n');

    try {
      await this.connect();
      await this.createMigrationsTable();
      await this.runMigrations();
      await this.seedInitialData();
      await this.validateSchema();
      
      console.log('\n✅ Database migration completed successfully!');
      return true;
    } catch (error) {
      console.error('\n❌ Migration failed:', error.message);
      return false;
    } finally {
      await this.disconnect();
    }
  }

  async connect() {
    console.log('🔌 Connecting to database...');
    this.client = new Client({ connectionString: this.connectionString });
    await this.client.connect();
    console.log('✅ Database connected\n');
  }

  async disconnect() {
    if (this.client) {
      await this.client.end();
      console.log('🔌 Database disconnected');
    }
  }

  async createMigrationsTable() {
    console.log('📋 Creating migrations tracking table...');
    
    await this.client.query(`
      CREATE TABLE IF NOT EXISTS schema_migrations (
        version VARCHAR(255) PRIMARY KEY,
        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        execution_time_ms INTEGER,
        checksum VARCHAR(64)
      );
    `);
    
    console.log('✅ Migrations table ready\n');
  }

  async runMigrations() {
    console.log('🚀 Running database migrations...');

    // Check if migrations directory exists
    if (!fs.existsSync(this.migrationsDir)) {
      console.log('📁 Creating migrations directory...');
      fs.mkdirSync(this.migrationsDir, { recursive: true });
    }

    // Run initial schema if no migrations exist
    const schemaPath = path.join(process.cwd(), 'infrastructure', 'sql', 'schema.sql');
    if (fs.existsSync(schemaPath)) {
      const { rows } = await this.client.query('SELECT COUNT(*) as count FROM schema_migrations');
      
      if (parseInt(rows[0].count) === 0) {
        console.log('📊 Running initial schema...');
        const startTime = Date.now();
        
        const schemaSQL = fs.readFileSync(schemaPath, 'utf8');
        await this.client.query(schemaSQL);
        
        const executionTime = Date.now() - startTime;
        
        // Record schema as migration
        await this.client.query(`
          INSERT INTO schema_migrations (version, execution_time_ms, checksum) 
          VALUES ($1, $2, $3)
        `, ['001_initial_schema', executionTime, this.calculateChecksum(schemaSQL)]);
        
        console.log(`✅ Initial schema applied (${executionTime}ms)`);
      }
    }

    // Run additional migrations
    const migrationFiles = fs.existsSync(this.migrationsDir) 
      ? fs.readdirSync(this.migrationsDir)
          .filter(file => file.endsWith('.sql'))
          .sort()
      : [];

    if (migrationFiles.length === 0) {
      console.log('📝 No additional migrations found');
    } else {
      for (const file of migrationFiles) {
        await this.runMigration(file);
      }
    }

    console.log('✅ All migrations completed\n');
  }

  async runMigration(filename) {
    const version = path.basename(filename, '.sql');
    
    // Check if already applied
    const { rows } = await this.client.query(
      'SELECT version FROM schema_migrations WHERE version = $1',
      [version]
    );

    if (rows.length > 0) {
      console.log(`⚠️  Migration ${version} already applied, skipping`);
      return;
    }

    console.log(`🔄 Running migration: ${version}`);
    const startTime = Date.now();

    const migrationPath = path.join(this.migrationsDir, filename);
    const migrationSQL = fs.readFileSync(migrationPath, 'utf8');

    try {
      await this.client.query('BEGIN');
      await this.client.query(migrationSQL);
      
      const executionTime = Date.now() - startTime;
      
      await this.client.query(`
        INSERT INTO schema_migrations (version, execution_time_ms, checksum) 
        VALUES ($1, $2, $3)
      `, [version, executionTime, this.calculateChecksum(migrationSQL)]);
      
      await this.client.query('COMMIT');
      console.log(`✅ Migration ${version} completed (${executionTime}ms)`);
    } catch (error) {
      await this.client.query('ROLLBACK');
      throw new Error(`Migration ${version} failed: ${error.message}`);
    }
  }

  async seedInitialData() {
    console.log('🌱 Seeding initial data...');

    // Create seeds directory if it doesn't exist
    if (!fs.existsSync(this.seedsDir)) {
      console.log('📁 Creating seeds directory...');
      fs.mkdirSync(this.seedsDir, { recursive: true });
    }

    // Create seed data files
    await this.createSeedFiles();

    const seedFiles = fs.readdirSync(this.seedsDir)
      .filter(file => file.endsWith('.sql'))
      .sort();

    if (seedFiles.length === 0) {
      console.log('📝 No seed files found, creating sample data...');
      await this.createSampleData();
    } else {
      for (const file of seedFiles) {
        await this.runSeed(file);
      }
    }

    console.log('✅ Data seeding completed\n');
  }

  async createSeedFiles() {
    // Create sample user profiles seed
    const userProfilesSeed = `
-- Sample User Profiles for Testing
INSERT INTO user_profiles (
  external_user_id, email, name, lifecycle_stage, engagement_score, 
  churn_risk_score, ltv_prediction, subscription_status, subscription_plan,
  created_at, last_active_at
) VALUES 
  ('demo_user_1', 'demo1@example.com', 'Demo User 1', 'active', 85.5, 15.2, 1250.00, 'active', 'premium', NOW() - INTERVAL '30 days', NOW() - INTERVAL '1 day'),
  ('demo_user_2', 'demo2@example.com', 'Demo User 2', 'trial', 65.0, 35.8, 850.00, 'trial', 'free', NOW() - INTERVAL '7 days', NOW() - INTERVAL '2 hours'),
  ('demo_user_3', 'demo3@example.com', 'Demo User 3', 'churned', 25.0, 85.5, 450.00, 'cancelled', 'basic', NOW() - INTERVAL '90 days', NOW() - INTERVAL '30 days')
ON CONFLICT (external_user_id) DO NOTHING;
`;

    const userProfilesPath = path.join(this.seedsDir, '001_user_profiles.sql');
    if (!fs.existsSync(userProfilesPath)) {
      fs.writeFileSync(userProfilesPath, userProfilesSeed);
    }

    // Create behavioral patterns seed
    const behavioralPatternsSeed = `
-- Sample Behavioral Patterns for Testing
INSERT INTO behavioral_patterns (
  user_id, pattern_type, pattern_data, confidence_score, 
  first_detected, last_detected, is_active
) VALUES 
  ('demo_user_1', 'high_engagement', '{"daily_sessions": 5, "avg_session_duration": 1800}', 0.92, NOW() - INTERVAL '20 days', NOW() - INTERVAL '1 day', true),
  ('demo_user_2', 'trial_exploration', '{"features_explored": 8, "documentation_views": 12}', 0.78, NOW() - INTERVAL '6 days', NOW() - INTERVAL '3 hours', true),
  ('demo_user_3', 'churn_indicators', '{"days_inactive": 30, "support_tickets": 3}', 0.85, NOW() - INTERVAL '35 days', NOW() - INTERVAL '30 days', false)
ON CONFLICT (user_id, pattern_type) DO NOTHING;
`;

    const behavioralPatternsPath = path.join(this.seedsDir, '002_behavioral_patterns.sql');
    if (!fs.existsSync(behavioralPatternsPath)) {
      fs.writeFileSync(behavioralPatternsPath, behavioralPatternsSeed);
    }
  }

  async runSeed(filename) {
    const seedPath = path.join(this.seedsDir, filename);
    console.log(`🌱 Running seed: ${filename}`);

    try {
      const seedSQL = fs.readFileSync(seedPath, 'utf8');
      await this.client.query(seedSQL);
      console.log(`✅ Seed ${filename} completed`);
    } catch (error) {
      console.log(`⚠️  Seed ${filename} failed (may be expected): ${error.message}`);
    }
  }

  async createSampleData() {
    console.log('📝 Creating sample data directly...');

    try {
      // Insert sample user profiles
      await this.client.query(`
        INSERT INTO user_profiles (
          external_user_id, email, name, lifecycle_stage, engagement_score,
          churn_risk_score, ltv_prediction, subscription_status, subscription_plan,
          created_at, last_active_at
        ) VALUES 
          ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11),
          ($12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
        ON CONFLICT (external_user_id) DO NOTHING
      `, [
        'sample_user_1', 'sample1@example.com', 'Sample User 1', 'active', 75.5,
        25.2, 1100.00, 'active', 'premium', new Date(Date.now() - 30*24*60*60*1000), new Date(Date.now() - 24*60*60*1000),
        'sample_user_2', 'sample2@example.com', 'Sample User 2', 'trial', 55.0,
        45.8, 650.00, 'trial', 'free', new Date(Date.now() - 7*24*60*60*1000), new Date(Date.now() - 2*60*60*1000)
      ]);

      console.log('✅ Sample user profiles created');
    } catch (error) {
      console.log(`⚠️  Sample data creation failed: ${error.message}`);
    }
  }

  async validateSchema() {
    console.log('🔍 Validating database schema...');

    const requiredTables = [
      'user_profiles',
      'behavioral_patterns', 
      'ml_models',
      'intervention_decisions',
      'message_deliveries',
      'performance_metrics',
      'schema_migrations'
    ];

    let validTables = 0;

    for (const table of requiredTables) {
      try {
        const { rows } = await this.client.query(`
          SELECT table_name 
          FROM information_schema.tables 
          WHERE table_schema = 'public' AND table_name = $1
        `, [table]);

        if (rows.length > 0) {
          console.log(`  ✅ Table '${table}' exists`);
          validTables++;
        } else {
          console.log(`  ❌ Table '${table}' missing`);
        }
      } catch (error) {
        console.log(`  ❌ Error checking table '${table}': ${error.message}`);
      }
    }

    // Check table row counts
    const { rows: migrationCount } = await this.client.query('SELECT COUNT(*) as count FROM schema_migrations');
    const { rows: userCount } = await this.client.query('SELECT COUNT(*) as count FROM user_profiles');

    console.log(`\n📊 Schema Validation Summary:`);
    console.log(`  • Tables: ${validTables}/${requiredTables.length} present`);
    console.log(`  • Migrations: ${migrationCount[0].count} applied`);
    console.log(`  • Sample Users: ${userCount[0].count} created`);

    if (validTables === requiredTables.length) {
      console.log('✅ Schema validation passed');
    } else {
      console.log('⚠️  Schema validation has issues');
    }
  }

  calculateChecksum(content) {
    const crypto = require('crypto');
    return crypto.createHash('sha256').update(content).digest('hex').substring(0, 16);
  }
}

// Run migrations if called directly
if (require.main === module) {
  const runner = new MigrationRunner();
  runner.run()
    .then((success) => {
      process.exit(success ? 0 : 1);
    })
    .catch((error) => {
      console.error('Migration runner error:', error);
      process.exit(1);
    });
}

module.exports = MigrationRunner;
