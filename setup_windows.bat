@echo off
echo ========================================
echo   AI Conspects Server - Windows Setup
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.9+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Checking Git installation...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git not found. Please install Git first.
    echo Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
if not exist venv (
    python -m venv venv
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Installing dependencies...
pip install -r requirements.txt
pip install pydantic-settings python-multipart

echo.
echo Running deployment analyzer...
python deploy_windows.py

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Install PostgreSQL and create 'ai_conspects' database
echo 2. Install and start Redis (or use Redis Cloud)
echo 3. Install Ollama and pull llama3.1:8b model
echo 4. Create .env file with your configuration
echo 5. Run: python run.py
echo.
echo See WINDOWS_SETUP.md for detailed instructions.
pause