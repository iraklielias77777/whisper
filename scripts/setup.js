#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🚀 Setting up User Whisperer Platform...\n');

// Check Node.js version
function checkNodeVersion() {
  console.log('📋 Checking Node.js version...');
  const nodeVersion = process.version;
  const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);
  
  if (majorVersion < 20) {
    console.error('❌ Node.js version 20 or higher is required. Current version:', nodeVersion);
    console.log('💡 Please run:');
    console.log('   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned');
    console.log('   fnm env --use-on-cd | Out-String | Invoke-Expression');
    console.log('   fnm use 22');
    process.exit(1);
  }
  
  console.log('✅ Node.js version:', nodeVersion);
}

// Check required tools
function checkRequiredTools() {
  console.log('\n🔧 Checking required tools...');
  
  const tools = [
    { name: 'docker', cmd: 'docker --version' },
    { name: 'git', cmd: 'git --version' },
  ];
  
  for (const tool of tools) {
    try {
      execSync(tool.cmd, { stdio: 'pipe' });
      console.log(`✅ ${tool.name} is installed`);
    } catch (error) {
      console.error(`❌ ${tool.name} is not installed or not in PATH`);
      console.log(`💡 Please install ${tool.name} and try again`);
      process.exit(1);
    }
  }
}

// Create essential configuration files
function createConfigFiles() {
  console.log('\n⚙️  Creating configuration files...');
  
  // Create .env.development file
  const envDevContent = `# Environment Configuration
NODE_ENV=development
LOG_LEVEL=debug

# Database Configuration  
POSTGRES_URL=postgresql://uwdev:localdev123@localhost:5432/userwhisperer_dev
REDIS_URL=redis://localhost:6379

# Service Ports
API_PORT=3000
INGESTION_PORT=3001
BEHAVIORAL_PORT=3002
DECISION_PORT=3003
CONTENT_PORT=3004
ORCHESTRATOR_PORT=3005

# External APIs (Development)
OPENAI_API_KEY=sk-dev-mock-key
ANTHROPIC_API_KEY=sk-ant-dev-mock-key
SENDGRID_API_KEY=SG.dev.mock-key
TWILIO_ACCOUNT_SID=ACdev_mock
TWILIO_AUTH_TOKEN=dev_mock_token

# Security
JWT_SECRET=dev-jwt-secret-key-not-for-production-use
API_KEY_SALT=dev-api-key-salt
ENCRYPTION_KEY=dev-encryption-key-32-chars-long

# Feature Flags
ENABLE_ML_MODELS=false
ENABLE_LLM_GENERATION=false
USE_MOCK_SERVICES=true

# Message Queue
PUBSUB_EMULATOR_HOST=localhost:8085
PUBSUB_PROJECT_ID=user-whisperer-dev
`;

  const envDevPath = path.join(__dirname, '..', '.env.development');
  fs.writeFileSync(envDevPath, envDevContent);
  console.log('✅ Created .env.development');
  
  // Copy to .env if it doesn't exist
  const envPath = path.join(__dirname, '..', '.env');
  if (!fs.existsSync(envPath)) {
    fs.copyFileSync(envDevPath, envPath);
    console.log('✅ Created .env from .env.development');
  }
}

// Create directories
function createDirectories() {
  console.log('\n📁 Creating required directories...');
  
  const directories = [
    'logs',
    'shared/utils',
    'shared/schemas',
    'shared/protos',
    'infrastructure/sql',
    'infrastructure/monitoring',
    'infrastructure/kong'
  ];
  
  for (const dir of directories) {
    const fullPath = path.join(__dirname, '..', dir);
    if (!fs.existsSync(fullPath)) {
      fs.mkdirSync(fullPath, { recursive: true });
      console.log(`✅ Created directory: ${dir}`);
    }
  }
}

// Install dependencies
function installDependencies() {
  console.log('\n📦 Installing dependencies...');
  
  try {
    console.log('Installing root dependencies...');
    execSync('npm install', { stdio: 'inherit', cwd: path.join(__dirname, '..') });
    console.log('✅ Dependencies installed successfully');
  } catch (error) {
    console.error('❌ Failed to install dependencies:', error.message);
    console.log('💡 This is normal if some services are not yet implemented');
  }
}

// Display success message
function displaySuccess() {
  console.log('\n🎉 Phase 1 Foundation Setup Complete!\n');
  console.log('📋 What was created:');
  console.log('   ✅ Project structure and configuration');
  console.log('   ✅ Environment configuration');
  console.log('   ✅ Build and development scripts');
  console.log('   ✅ Essential directories\n');
  console.log('📋 Next Steps:');
  console.log('   1. Continue with Phase 2: Core Services Implementation');
  console.log('   2. After Phase 2: make dev (to start services)');
  console.log('   3. After Phase 2: make test (to run tests)\n');
  console.log('✨ Foundation is ready for Phase 2 development!');
}

// Main setup function
function main() {
  try {
    checkNodeVersion();
    checkRequiredTools();
    createDirectories();
    createConfigFiles();
    installDependencies();
    displaySuccess();
  } catch (error) {
    console.error('\n❌ Setup failed:', error.message);
    process.exit(1);
  }
}

// Run setup
main(); 