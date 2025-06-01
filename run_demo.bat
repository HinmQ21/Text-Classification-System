@echo off
echo ========================================
echo 🚀 Text Classification Demo Launcher
echo ========================================

echo.
echo 🔍 Checking requirements...

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed or not in PATH
    echo Please install Docker Desktop and try again
    pause
    exit /b 1
)
echo ✅ Docker is installed

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed or not in PATH
    echo Please install Docker Compose and try again
    pause
    exit /b 1
)
echo ✅ Docker Compose is installed

echo.
echo 🏗️ Building and starting services...
docker-compose up --build -d

if %errorlevel% neq 0 (
    echo ❌ Failed to start services
    echo Check the logs with: docker-compose logs
    pause
    exit /b 1
)

echo ✅ Services started successfully!

echo.
echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak >nul

echo.
echo 🌐 Opening demo in browser...
start http://localhost:3000

echo.
echo ========================================
echo 🎉 TEXT CLASSIFICATION DEMO IS RUNNING!
echo ========================================
echo 📱 Frontend:     http://localhost:3000
echo 🔧 Backend API:  http://localhost:8000
echo 📚 API Docs:     http://localhost:8000/docs
echo ❤️  Health Check: http://localhost:8000/health
echo ========================================
echo.
echo 🧪 Try these examples:
echo • Sentiment: 'I love this product!'
echo • Spam: 'FREE MONEY! Click now!'
echo • Topic: 'The new AI technology is amazing'
echo.
echo ⏹️  To stop: docker-compose down
echo 📋 To view logs: docker-compose logs
echo ========================================
echo.
echo 🎯 Demo is ready! Enjoy testing!
echo.
pause
