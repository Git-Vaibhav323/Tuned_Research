@echo off
setlocal

echo.
echo =============================================
echo   ResearchPilot — AI Research Assistant
echo =============================================
echo.

cd /d E:\Tuned_Research

:: Kill anything already on port 8501
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8501 "') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo Starting Streamlit on http://localhost:8501
echo Press Ctrl+C to stop.
echo.

py -3.12 -m streamlit run streamlit_app\app.py ^
    --server.port 8501 ^
    --server.headless false ^
    --browser.gatherUsageStats false

pause
