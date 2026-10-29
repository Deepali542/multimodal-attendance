@echo off
title Multimodal Attendance & Engagement Agent
echo ===========================================================================
echo Starting Multimodal Attendance & Engagement Agent...
echo ===========================================================================
cd /d "%~dp0"

echo [1/2] Checking Python Environment...
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH!
    pause
    exit /b 1
)

echo [2/2] Running Classroom Simulation & Report Generation...
python main.py --mode simulation

echo.
echo ===========================================================================
echo Optional: Launch Streamlit Dashboard by running:
echo streamlit run src/ui/app.py
echo ===========================================================================
pause
