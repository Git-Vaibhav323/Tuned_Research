@echo off
echo.
echo =============================================
echo   ResearchPilot — React Frontend
echo =============================================
echo.
echo Starting FastAPI backend on port 8000...
echo Starting React frontend on port 5173...
echo.
echo Backend API: http://localhost:8000/docs
echo Frontend:    http://localhost:5173
echo.
echo Press Ctrl+C to stop both servers.
echo.

cd /d E:\Tuned_Research

:: Start FastAPI in background
start "ResearchPilot API" cmd /k "py -3.12 -m uvicorn api.main:app --reload --port 8000 --host 127.0.0.1"

:: Wait for API to start
timeout /t 3 /nobreak >nul

:: Start Vite dev server
cd frontend
npm run dev
