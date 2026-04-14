@echo off
setlocal
echo ========================================================
echo 🚀 Starting Unified AI Recommendation System
echo ========================================================
echo.

if not exist venv (
    echo ❌ Virtual environment 'venv' not found!
    echo Please install it first or run: python -m venv venv
    pause
    exit /b 1
)

echo [1] Activating Virtual Environment...
call venv\Scripts\activate

echo.
echo [2] Starting Unified Server (FastAPI + Flask)...
echo    - Address: http://localhost:5000
echo    - Health:  http://localhost:5000/health
echo.
echo ⚠️ PLEASE WAIT for the server to initialize models...
echo.

python unified_server.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ Server crashed or failed to start.
    echo Please check the error messages above.
)

pause
