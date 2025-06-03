@echo off
echo Starting Text Classification System with Redis Queue...
echo.

:: Check if Redis is running
echo Checking Redis connection...
redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo Redis is not running! Please start Redis first.
    echo Run: redis-server
    pause
    exit /b 1
)
echo Redis is running ✓
echo.

:: Start RQ Dashboard in background
echo Starting RQ Dashboard...
start "RQ Dashboard" cmd /c "rq-dashboard --port 9181"
echo RQ Dashboard started on http://localhost:9181
echo.

:: Start Workers
echo Starting RQ Workers...
start "RQ Workers" cmd /c "python start_workers.py --mode monitor"
echo Workers started ✓
echo.

:: Wait a moment for workers to initialize
timeout /t 3 /nobreak >nul

:: Start FastAPI Server
echo Starting FastAPI Server...
echo Server will be available at http://localhost:8000
echo Swagger UI at http://localhost:8000/docs
echo.
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

echo.
echo System stopped.
pause 