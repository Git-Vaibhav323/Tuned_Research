# launch_streamlit.ps1 - ResearchPilot Streamlit launcher
# Run from PowerShell:  .\launch_streamlit.ps1

$PORT   = 8501
$SCRIPT = "streamlit_app\app.py"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  ResearchPilot - Streamlit Launcher" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Kill anything already on the port
$existing = Get-NetTCPConnection -LocalPort $PORT -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Stopping existing process on port $PORT..." -ForegroundColor Yellow
    $existing | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

Set-Location "E:\Tuned_Research"

Write-Host "  Starting on http://localhost:$PORT" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

py -3.12 -m streamlit run $SCRIPT `
    --server.port $PORT `
    --server.headless false `
    --browser.gatherUsageStats false
