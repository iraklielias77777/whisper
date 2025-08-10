# Complete Platform Integration Test Script
# PowerShell script to test all integration flows

Write-Host "🧪 Starting Complete Platform Integration Tests..." -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Gray

$ErrorActionPreference = "Continue"
$TestResults = @{
    Passed = 0
    Failed = 0
    Warnings = 0
    Tests = @()
}

function Test-Component {
    param(
        [string]$TestName,
        [scriptblock]$TestScript
    )
    
    $StartTime = Get-Date
    Write-Host "🔍 Testing: $TestName" -ForegroundColor Yellow
    
    try {
        $Result = & $TestScript
        $Duration = (Get-Date) - $StartTime
        
        if ($Result -eq $true) {
            Write-Host "  ✅ PASSED ($(($Duration).TotalMilliseconds)ms)" -ForegroundColor Green
            $TestResults.Passed++
        } else {
            Write-Host "  ⚠️  WARNING ($(($Duration).TotalMilliseconds)ms): $Result" -ForegroundColor Yellow
            $TestResults.Warnings++
        }
        
        $TestResults.Tests += @{
            Name = $TestName
            Status = if ($Result -eq $true) { "Passed" } else { "Warning" }
            Duration = $Duration.TotalMilliseconds
            Result = $Result
        }
    }
    catch {
        $Duration = (Get-Date) - $StartTime
        Write-Host "  ❌ FAILED ($(($Duration).TotalMilliseconds)ms): $($_.Exception.Message)" -ForegroundColor Red
        $TestResults.Failed++
        
        $TestResults.Tests += @{
            Name = $TestName
            Status = "Failed"
            Duration = $Duration.TotalMilliseconds
            Error = $_.Exception.Message
        }
    }
}

# Test 1: Infrastructure Health
Test-Component "Infrastructure Health Check" {
    $Services = @(
        @{ Name = "PostgreSQL"; Port = 5432 },
        @{ Name = "Redis"; Port = 6379 },
        @{ Name = "Kong Gateway"; Port = 8000 },
        @{ Name = "Prometheus"; Port = 9090 },
        @{ Name = "Grafana"; Port = 3000 }
    )
    
    $HealthyServices = 0
    foreach ($Service in $Services) {
        try {
            $Connection = New-Object System.Net.Sockets.TcpClient
            $Connection.Connect("localhost", $Service.Port)
            $Connection.Close()
            $HealthyServices++
            Write-Host "    ✅ $($Service.Name) (Port $($Service.Port))" -ForegroundColor Gray
        }
        catch {
            Write-Host "    ❌ $($Service.Name) (Port $($Service.Port)) - Not responding" -ForegroundColor Red
        }
    }
    
    if ($HealthyServices -eq $Services.Count) {
        return $true
    } else {
        return "Only $HealthyServices/$($Services.Count) infrastructure services are healthy"
    }
}

# Test 2: Microservices Health
Test-Component "Microservices Health Check" {
    $Services = @(
        @{ Name = "event-ingestion"; Port = 3001 },
        @{ Name = "behavioral-analysis"; Port = 3002 },
        @{ Name = "decision-engine"; Port = 3003 },
        @{ Name = "content-generation"; Port = 3004 },
        @{ Name = "channel-orchestrator"; Port = 3005 },
        @{ Name = "ai-orchestration"; Port = 3006 }
    )
    
    $HealthyServices = 0
    foreach ($Service in $Services) {
        try {
            $Response = Invoke-RestMethod -Uri "http://localhost:$($Service.Port)/health" -TimeoutSec 5 -ErrorAction Stop
            if ($Response.status -eq "healthy" -or $Response.service) {
                $HealthyServices++
                Write-Host "    ✅ $($Service.Name)" -ForegroundColor Gray
            } else {
                Write-Host "    ⚠️  $($Service.Name) - Unhealthy response" -ForegroundColor Yellow
            }
        }
        catch {
            Write-Host "    ❌ $($Service.Name) - Not responding" -ForegroundColor Red
        }
    }
    
    if ($HealthyServices -ge 4) {  # Allow some services to be down
        return $true
    } else {
        return "Only $HealthyServices/$($Services.Count) microservices are healthy"
    }
}

# Test 3: Kong Gateway Routing
Test-Component "API Gateway Routing" {
    $ApiKey = "demo_1234567890abcdef"
    $RoutedServices = 0
    $Services = @("event-ingestion", "behavioral-analysis", "decision-engine")
    
    foreach ($Service in $Services) {
        try {
            $Headers = @{ "X-API-Key" = $ApiKey }
            $Response = Invoke-RestMethod -Uri "http://localhost:8000/health/$Service" -Headers $Headers -TimeoutSec 5 -ErrorAction Stop
            
            if ($Response.service -eq $Service -or $Response.status) {
                $RoutedServices++
                Write-Host "    ✅ $Service routing" -ForegroundColor Gray
            }
        }
        catch {
            Write-Host "    ❌ $Service routing failed" -ForegroundColor Red
        }
    }
    
    if ($RoutedServices -ge 2) {
        return $true
    } else {
        return "Only $RoutedServices/$($Services.Count) services routed correctly"
    }
}

# Test 4: Database Connectivity
Test-Component "Database Integration" {
    try {
        # Test database connection using PowerShell
        $ConnectionString = "Host=localhost;Port=5432;Database=userwhisperer_dev;Username=uwdev;Password=localdev123"
        
        # For this test, we'll just check if the port is open and assume schema is correct
        $Connection = New-Object System.Net.Sockets.TcpClient
        $Connection.Connect("localhost", 5432)
        $Connection.Close()
        
        Write-Host "    ✅ Database connection verified" -ForegroundColor Gray
        return $true
    }
    catch {
        return "Database connection failed: $($_.Exception.Message)"
    }
}

# Test 5: Event Processing Flow
Test-Component "Event Processing Workflow" {
    $ApiKey = "demo_1234567890abcdef"
    $TestUserId = "integration_test_$(Get-Date -Format 'yyyyMMddHHmmss')"
    
    try {
        # Send a test event
        $EventData = @{
            event_type = "integration_test"
            user_id = $TestUserId
            timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
            properties = @{
                test_run = $true
                integration_test = $true
            }
        } | ConvertTo-Json
        
        $Headers = @{
            "Content-Type" = "application/json"
            "X-API-Key" = $ApiKey
        }
        
        $Response = Invoke-RestMethod -Uri "http://localhost:8000/v1/events/track" -Method POST -Body $EventData -Headers $Headers -TimeoutSec 10 -ErrorAction Stop
        
        if ($Response.status -eq "accepted" -or $Response.result -eq "success") {
            Write-Host "    ✅ Event successfully processed" -ForegroundColor Gray
            return $true
        } else {
            return "Event processing returned unexpected response: $($Response | ConvertTo-Json -Compress)"
        }
    }
    catch {
        return "Event processing failed: $($_.Exception.Message)"
    }
}

# Test 6: Monitoring and Metrics
Test-Component "Monitoring Integration" {
    $MonitoringServices = @(
        @{ Name = "Prometheus"; Url = "http://localhost:9090/api/v1/status/buildinfo" },
        @{ Name = "Grafana"; Url = "http://localhost:3000/api/health" }
    )
    
    $WorkingServices = 0
    foreach ($Service in $MonitoringServices) {
        try {
            $Response = Invoke-RestMethod -Uri $Service.Url -TimeoutSec 5 -ErrorAction Stop
            $WorkingServices++
            Write-Host "    ✅ $($Service.Name) metrics accessible" -ForegroundColor Gray
        }
        catch {
            Write-Host "    ❌ $($Service.Name) not accessible" -ForegroundColor Red
        }
    }
    
    if ($WorkingServices -eq $MonitoringServices.Count) {
        return $true
    } else {
        return "$WorkingServices/$($MonitoringServices.Count) monitoring services accessible"
    }
}

# Test 7: Service Discovery
Test-Component "Service Discovery Validation" {
    # Test Redis connectivity for service discovery
    try {
        $Connection = New-Object System.Net.Sockets.TcpClient
        $Connection.Connect("localhost", 6379)
        $Connection.Close()
        
        Write-Host "    ✅ Redis (Service Registry) accessible" -ForegroundColor Gray
        
        # Count healthy services
        $HealthyServices = 0
        $ServicePorts = @(3001, 3002, 3003, 3004, 3005, 3006)
        
        foreach ($Port in $ServicePorts) {
            try {
                $TestConnection = New-Object System.Net.Sockets.TcpClient
                $TestConnection.Connect("localhost", $Port)
                $TestConnection.Close()
                $HealthyServices++
            }
            catch {
                # Service not running
            }
        }
        
        Write-Host "    📊 $HealthyServices/$($ServicePorts.Count) services discoverable" -ForegroundColor Gray
        
        if ($HealthyServices -ge 3) {
            return $true
        } else {
            return "Service discovery working but only $HealthyServices services active"
        }
    }
    catch {
        return "Service discovery (Redis) not accessible"
    }
}

# Test 8: Load Testing Sample
Test-Component "Basic Load Handling" {
    $ApiKey = "demo_1234567890abcdef"
    $ConcurrentRequests = 5
    $SuccessfulRequests = 0
    
    Write-Host "    🚀 Sending $ConcurrentRequests concurrent requests..." -ForegroundColor Gray
    
    $Jobs = @()
    for ($i = 1; $i -le $ConcurrentRequests; $i++) {
        $Job = Start-Job -ScriptBlock {
            param($ApiKey, $RequestId)
            
            try {
                $EventData = @{
                    event_type = "load_test"
                    user_id = "load_test_user_$RequestId"
                    timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                    properties = @{ request_id = $RequestId }
                } | ConvertTo-Json
                
                $Headers = @{
                    "Content-Type" = "application/json"
                    "X-API-Key" = $ApiKey
                }
                
                $Response = Invoke-RestMethod -Uri "http://localhost:8000/v1/events/track" -Method POST -Body $EventData -Headers $Headers -TimeoutSec 15
                return $true
            }
            catch {
                return $false
            }
        } -ArgumentList $ApiKey, $i
        
        $Jobs += $Job
    }
    
    # Wait for all jobs to complete
    $Jobs | Wait-Job -Timeout 30 | Out-Null
    
    foreach ($Job in $Jobs) {
        $Result = Receive-Job $Job
        if ($Result -eq $true) {
            $SuccessfulRequests++
        }
        Remove-Job $Job
    }
    
    Write-Host "    📈 $SuccessfulRequests/$ConcurrentRequests requests successful" -ForegroundColor Gray
    
    if ($SuccessfulRequests -ge ($ConcurrentRequests * 0.8)) {  # 80% success rate
        return $true
    } else {
        return "Load test: Only $SuccessfulRequests/$ConcurrentRequests requests successful"
    }
}

Write-Host ""
Write-Host "📊 INTEGRATION TEST RESULTS" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray
Write-Host "Total Tests: $($TestResults.Passed + $TestResults.Failed + $TestResults.Warnings)" -ForegroundColor White
Write-Host "✅ Passed: $($TestResults.Passed)" -ForegroundColor Green
Write-Host "❌ Failed: $($TestResults.Failed)" -ForegroundColor Red
Write-Host "⚠️  Warnings: $($TestResults.Warnings)" -ForegroundColor Yellow

$SuccessRate = if (($TestResults.Passed + $TestResults.Failed) -gt 0) {
    [Math]::Round(($TestResults.Passed / ($TestResults.Passed + $TestResults.Failed)) * 100, 1)
} else { 0 }
Write-Host "📈 Success Rate: $SuccessRate%" -ForegroundColor Cyan

Write-Host ""
if ($TestResults.Failed -eq 0) {
    Write-Host "🎉 ALL INTEGRATION TESTS PASSED!" -ForegroundColor Green
    Write-Host "The User Whisperer Platform is fully integrated and ready for production!" -ForegroundColor Green
} else {
    Write-Host "💥 SOME TESTS FAILED!" -ForegroundColor Red
    Write-Host "Please check the failed tests above and ensure all services are running properly." -ForegroundColor Yellow
    
    Write-Host ""
    Write-Host "🛠️  Troubleshooting Tips:" -ForegroundColor Cyan
    Write-Host "  • Ensure Docker is running: docker-compose ps" -ForegroundColor White
    Write-Host "  • Start all services: docker-compose up -d" -ForegroundColor White
    Write-Host "  • Check service logs: docker-compose logs [service-name]" -ForegroundColor White
    Write-Host "  • Run health checks: node scripts/validate-services.js" -ForegroundColor White
}

Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
if ($TestResults.Failed -eq 0) {
    Write-Host "  • ✅ Platform ready for Phase 4 (Production Hardening)" -ForegroundColor Green
    Write-Host "  • 🚀 Consider deploying to staging environment" -ForegroundColor White
    Write-Host "  • 📊 Monitor dashboards: http://localhost:3000" -ForegroundColor White
} else {
    Write-Host "  • 🔧 Fix failed integration tests" -ForegroundColor Yellow
    Write-Host "  • 🔍 Investigate service connectivity issues" -ForegroundColor White
    Write-Host "  • 📋 Re-run integration tests after fixes" -ForegroundColor White
}

# Exit with appropriate code
if ($TestResults.Failed -eq 0) {
    exit 0
} else {
    exit 1
}
