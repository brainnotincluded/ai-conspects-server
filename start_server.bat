@echo off
echo ====================================
echo AI Conspects Server - Windows Startup
echo ====================================

echo.
echo 🚀 Starting AI Conspects Server...
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found!
    echo Please run setup_windows.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if required packages are installed
python -c "import fastapi, uvicorn, sqlalchemy" 2>nul
if errorlevel 1 (
    echo ❌ Required packages not installed!
    echo Installing dependencies...
    pip install fastapi uvicorn sqlalchemy python-jose[cryptography] passlib[bcrypt] python-multipart
)

echo.
echo 🔧 Server Configuration:
echo ----------------------------------------
echo Host: 0.0.0.0
echo Port: 5000
echo Database: SQLite (ai_conspects.db)
echo Authentication: JWT with device registration
echo.

REM Check if port 5000 is available
netstat -an | find "0.0.0.0:5000" >nul
if %errorlevel%==0 (
    echo ⚠️ Port 5000 is already in use!
    echo Please stop the existing server or change the port.
    pause
    exit /b 1
)

echo ✅ Port 5000 is available
echo.

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs

echo 📁 Created required directories
echo.

echo 🎯 iOS App Endpoints Available:
echo ----------------------------------------
echo GET  /auth/generate-device-id
echo POST /auth/register
echo GET  /auth/me
echo POST /batches/upload-files
echo GET  /batches/{batch_id}
echo GET  /batches/{batch_id}/results
echo GET  /notes/
echo GET  /notes/{note_id}
echo POST /notes/{note_id}/toggle-favorite
echo POST /chat/query
echo POST /chat/study-plan
echo GET  /health
echo.

echo 🔥 Starting server with GPU support...
echo Press Ctrl+C to stop the server
echo.

REM Start the server
python ios_compatible_server.py

echo.
echo 👋 Server stopped.
pause