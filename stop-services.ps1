# User Whisperer Platform - Stop All Services
Write-Host "🛑 Stopping User Whisperer Platform Services..." -ForegroundColor Red

# Stop all Node.js processes that might be our services
$nodeProcesses = Get-Process -Name "node" -ErrorAction SilentlyContinue

if ($nodeProcesses) {
    Write-Host "Stopping Node.js processes..." -ForegroundColor Yellow
    $nodeProcesses | ForEach-Object {
        try {
            Stop-Process -Id $_.Id -Force
            Write-Host "✅ Stopped process ID: $($_.Id)" -ForegroundColor Green
        } catch {
            Write-Host "⚠️ Could not stop process ID: $($_.Id)" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "ℹ️ No Node.js processes found running." -ForegroundColor Blue
}

Write-Host "`n🏁 All services stopped." -ForegroundColor Green
