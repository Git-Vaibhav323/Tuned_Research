@echo off
echo.
echo =============================================
echo   ResearchPilot Chatbot
echo =============================================
echo.
echo Starting Streamlit UI...
echo Browser will open automatically.
echo Press Ctrl+C to stop.
echo.
"C:\Users\LOQ\AppData\Local\Python\pythoncore-3.12-64\python.exe" -m streamlit run streamlit_app/app.py --server.port 8501 --server.headless false
pause
