# User Whisperer Platform Startup Script
# PowerShell version for Windows development

Write-Host "🚀 Starting User Whisperer Platform..." -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Gray

# Check if Docker is running
Write-Host "🐳 Checking Docker..." -ForegroundColor Yellow
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if docker-compose is available
Write-Host "🔧 Checking Docker Compose..." -ForegroundColor Yellow
try {
    docker-compose version | Out-Null
    Write-Host "✅ Docker Compose is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose is not available." -ForegroundColor Red
    exit 1
}

# Create environment file if it doesn't exist
if (!(Test-Path ".env")) {
    Write-Host "📝 Creating .env file..." -ForegroundColor Yellow
    @"
# User Whisperer Platform Environment Variables
NODE_ENV=development
LOG_LEVEL=debug

# Database
POSTGRES_URL=postgresql://uwdev:localdev123@localhost:5432/userwhisperer_dev

# Redis
REDIS_URL=redis://localhost:6379

# API Keys (Development - Replace in production)
OPENAI_API_KEY=sk-dev-mock
ANTHROPIC_API_KEY=sk-ant-dev-mock
SENDGRID_API_KEY=SG.dev.mock
TWILIO_ACCOUNT_SID=ACdev
TWILIO_AUTH_TOKEN=devtoken

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_WINDOW_MS=60000
RATE_LIMIT_MAX_REQUESTS=100

# Metrics
METRICS_ENABLED=true

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
"@ | Out-File -FilePath ".env" -Encoding UTF8

    Write-Host "✅ .env file created" -ForegroundColor Green
}

# Build and start services
Write-Host "🏗️  Building and starting services..." -ForegroundColor Yellow
Write-Host "This may take a few minutes on first run..." -ForegroundColor Gray

try {
    docker-compose up -d --build
    Write-Host "✅ Services started successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to start services" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

# Wait for services to be ready
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Run validation
Write-Host "🔍 Running service validation..." -ForegroundColor Yellow
try {
    node scripts/validate-services.js
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ All validations passed!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Some validations failed. Check the output above." -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Validation script failed" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎉 User Whisperer Platform is now running!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Service URLs:" -ForegroundColor Cyan
Write-Host "  • Kong API Gateway: http://localhost:8000" -ForegroundColor White
Write-Host "  • Kong Admin API: http://localhost:8001" -ForegroundColor White
Write-Host "  • Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host "  • Grafana: http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host "  • Event Ingestion: http://localhost:3001/health" -ForegroundColor White
Write-Host "  • Behavioral Analysis: http://localhost:3002/health" -ForegroundColor White
Write-Host "  • Decision Engine: http://localhost:3003/health" -ForegroundColor White
Write-Host "  • Content Generation: http://localhost:3004/health" -ForegroundColor White
Write-Host "  • Channel Orchestrator: http://localhost:3005/health" -ForegroundColor White
Write-Host "  • AI Orchestration: http://localhost:3006/health" -ForegroundColor White
Write-Host ""
Write-Host "📋 Useful Commands:" -ForegroundColor Cyan
Write-Host "  • View logs: docker-compose logs -f [service-name]" -ForegroundColor White
Write-Host "  • Restart service: docker-compose restart [service-name]" -ForegroundColor White
Write-Host "  • Stop platform: docker-compose down" -ForegroundColor White
Write-Host "  • View status: docker-compose ps" -ForegroundColor White
Write-Host "  • Run validation: node scripts/validate-services.js" -ForegroundColor White
Write-Host ""
Write-Host "🔧 To stop the platform: docker-compose down" -ForegroundColor Yellow
