# User Whisperer Platform - Production Deployment Script
# Run this script to start all services locally with production configuration

Write-Host "🚀 User Whisperer Platform - Production Deployment" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Check if production.env exists
if (-not (Test-Path "production.env")) {
    Write-Host "❌ Error: production.env file not found!" -ForegroundColor Red
    Write-Host "Please create production.env with your configuration." -ForegroundColor Yellow
    exit 1
}

# Load environment variables from production.env
Write-Host "🔧 Loading production environment variables..." -ForegroundColor Cyan
Get-Content "production.env" | ForEach-Object {
    if ($_ -match "^([^#][^=]+)=(.*)$") {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        Write-Host "   Set: $($matches[1])" -ForegroundColor Gray
    }
}

# Set additional environment variables
$env:NODE_ENV = "production"
$env:PORT = "3001"

Write-Host "`n🏗️ Building shared library..." -ForegroundColor Yellow
Set-Location "shared"
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to build shared library" -ForegroundColor Red
    exit 1
}
Set-Location ".."

Write-Host "`n📡 Starting Event Ingestion Service (Port 3001)..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-Command", "cd '$PWD\services\event-ingestion'; npm start" -WindowStyle Minimized

Start-Sleep -Seconds 3

Write-Host "📊 Starting Behavioral Analysis Service (Port 3002)..." -ForegroundColor Blue
$env:PORT = "3002"
Start-Process powershell -ArgumentList "-Command", "cd '$PWD\services\behavioral-analysis'; npm start" -WindowStyle Minimized

Start-Sleep -Seconds 3

Write-Host "🧠 Starting Decision Engine Service (Port 3003)..." -ForegroundColor Blue
$env:PORT = "3003"
Start-Process powershell -ArgumentList "-Command", "cd '$PWD\services\decision-engine'; npm start" -WindowStyle Minimized

Start-Sleep -Seconds 3

Write-Host "✍️ Starting Content Generation Service (Port 3004)..." -ForegroundColor Blue
$env:PORT = "3004"
Start-Process powershell -ArgumentList "-Command", "cd '$PWD\services\content-generation'; npm start" -WindowStyle Minimized

Start-Sleep -Seconds 3

Write-Host "📮 Starting Channel Orchestrator Service (Port 3005)..." -ForegroundColor Blue
$env:PORT = "3005"
Start-Process powershell -ArgumentList "-Command", "cd '$PWD\services\channel-orchestrator'; npm start" -WindowStyle Minimized

Start-Sleep -Seconds 3

Write-Host "🤖 Starting AI Orchestration Service (Port 8085)..." -ForegroundColor Blue
$env:PORT = "8085"
Start-Process powershell -ArgumentList "-Command", "cd '$PWD\services\ai-orchestration'; npm start" -WindowStyle Minimized

Write-Host "`n⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "`n🔍 Testing service health..." -ForegroundColor Green

$services = @(
    @{Name="Event Ingestion"; Port=3001; Path="/health"},
    @{Name="Behavioral Analysis"; Port=3002; Path="/health"},
    @{Name="Decision Engine"; Port=3003; Path="/health"},
    @{Name="Content Generation"; Port=3004; Path="/health"},
    @{Name="Channel Orchestrator"; Port=3005; Path="/health"},
    @{Name="AI Orchestration"; Port=8085; Path="/health"}
)

$allHealthy = $true

foreach ($service in $services) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$($service.Port)$($service.Path)" -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ $($service.Name) - Healthy" -ForegroundColor Green
        } else {
            Write-Host "⚠️ $($service.Name) - Unhealthy (Status: $($response.StatusCode))" -ForegroundColor Yellow
            $allHealthy = $false
        }
    } catch {
        Write-Host "❌ $($service.Name) - Not responding" -ForegroundColor Red
        $allHealthy = $false
    }
}

if ($allHealthy) {
    Write-Host "`n🎉 All services are healthy and running!" -ForegroundColor Green
    Write-Host "`n📱 You can now test the SDK integration:" -ForegroundColor Cyan
    Write-Host "   Open: test-sdk-integration.html" -ForegroundColor White
    Write-Host "`n🌐 API Endpoints available at:" -ForegroundColor Cyan
    Write-Host "   http://localhost:3001/v1/events" -ForegroundColor White
    Write-Host "   http://localhost:3002/v1/analytics" -ForegroundColor White
    Write-Host "   http://localhost:3003/v1/decisions" -ForegroundColor White
    Write-Host "   http://localhost:3004/v1/content" -ForegroundColor White
    Write-Host "   http://localhost:3005/v1/delivery" -ForegroundColor White
    Write-Host "   http://localhost:8085/v1/orchestrate" -ForegroundColor White
} else {
    Write-Host "`n⚠️ Some services are not healthy. Check the logs for details." -ForegroundColor Yellow
}

Write-Host "`n📊 Service Management:" -ForegroundColor Cyan
Write-Host "   To stop all services: run stop-services.ps1" -ForegroundColor White
Write-Host "   To view logs: check individual service windows" -ForegroundColor White
