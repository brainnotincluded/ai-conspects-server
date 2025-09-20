# 🪟 Windows Deployment Guide - AI Conspects Server

This guide will help you set up the full AI Conspects Server on your Windows PC with RTX 4090 GPU support.

## 🎯 Overview
Your server will run at `http://62.140.252.238:5000` with:
- **Whisper Large** for speech-to-text using your RTX 4090 GPU
- **Ollama** for local LLM processing
- **PostgreSQL** for data storage
- **Redis** for caching and task queues
- **Full API** with authentication, audio processing, and AI chat

## 📋 Prerequisites

### 1. Install Python 3.9+
Download and install Python from https://www.python.org/downloads/
Make sure to check "Add Python to PATH" during installation.

### 2. Install Git
Download and install Git from https://git-scm.com/download/win

### 3. Install CUDA (for RTX 4090)
- Download CUDA Toolkit 11.8 from NVIDIA
- Your RTX 4090 should already have appropriate drivers

## 🚀 Step-by-Step Setup

### Step 1: Clone the Repository
Open Command Prompt or PowerShell and run:

```powershell
# Navigate to your preferred directory
cd C:\
mkdir Projects
cd Projects

# Clone the repository
git clone https://github.com/brainnotincluded/ai-conspects-server.git
cd ai-conspects-server
```

### Step 2: Create Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 3: Install Dependencies
```powershell
# Install PyTorch with CUDA support (for RTX 4090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install server dependencies
pip install -r requirements.txt

# Install additional Windows-specific packages
pip install pydantic-settings python-multipart
```

### Step 4: Install PostgreSQL
1. Download PostgreSQL from https://www.postgresql.org/download/windows/
2. Install with these settings:
   - Port: 5432
   - Username: postgres
   - Password: (choose a secure password)
3. Create database:
   ```sql
   CREATE DATABASE ai_conspects;
   ```

### Step 5: Install Redis
**Option A: Redis on WSL2 (Recommended)**
```powershell
# Install WSL2 if not already installed
wsl --install

# In WSL2 terminal:
sudo apt update
sudo apt install redis-server
redis-server --daemonize yes
```

**Option B: Redis Cloud (Easier)**
- Sign up at https://redis.com/
- Create free database
- Get connection URL

### Step 6: Install Ollama
1. Download Ollama from https://ollama.ai/download
2. Install and start Ollama
3. Pull the required model:
   ```powershell
   ollama pull llama3.1:8b
   ```

### Step 7: Configure Environment
Create `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/ai_conspects
REDIS_URL=redis://localhost:6379

# AI Services (Optional - add your API keys)
OPENAI_API_KEY=your_openai_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# JWT Security
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production-make-it-long-and-random

# Server Configuration
HOST=0.0.0.0
PORT_WINDOWS=5000

# Processing Settings
MAX_BATCH_SIZE=10
MAX_RECORDING_DURATION=600
WHISPER_MODEL=large-v3

# Celery (for background tasks)
CELERY_BROKER_URL=redis://localhost:6379
CELERY_RESULT_BACKEND=redis://localhost:6379

# Storage (Optional - for AWS S3)
S3_BUCKET=ai-conspects-audio
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
S3_REGION=us-east-1
```

### Step 8: Initialize Database
```powershell
# Run the deployment analyzer first
python deploy_windows.py

# Initialize database tables
python -c "from database import init_db; import asyncio; asyncio.run(init_db())"
```

### Step 9: Start the Server
```powershell
# Make sure virtual environment is activated
venv\Scripts\activate

# Start the full server
python run.py
```

The server should start and show:
```
🚀 Starting AI Conspects Server on Windows - Port: 5000
📊 Configuration:
   - Host: 0.0.0.0
   - Port: 5000
   - Platform: Windows
   - Database: postgresql://...
   - Whisper Model: large-v3

🌐 Server will be available at: http://0.0.0.0:5000
📚 API Documentation: http://0.0.0.0:5000/docs
```

## ✅ Verification

### Test Basic Endpoints
```powershell
# Test health check
curl http://localhost:5000/health

# Test API documentation
# Open browser: http://localhost:5000/docs
```

### Test AI Features
```powershell
# Test authentication endpoint
curl -X POST http://localhost:5000/api/v1/auth/register -H "Content-Type: application/json" -d "{\"device_id\":\"test-123\"}"

# Test GPU is being used
# Check Task Manager -> Performance -> GPU for CUDA activity when processing audio
```

### Test External Access
From your Mac, test that the server is accessible:
```bash
curl http://62.140.252.238:5000/health
curl http://62.140.252.238:5000/docs
```

## 🔧 Troubleshooting

### Common Issues

**Port 5000 Already in Use:**
```powershell
# Kill process using port 5000
netstat -ano | findstr :5000
taskkill /PID [PID_NUMBER] /F
```

**CUDA Not Found:**
- Verify NVIDIA drivers are updated
- Reinstall PyTorch with CUDA support
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

**Database Connection Error:**
- Verify PostgreSQL is running: `services.msc`
- Check DATABASE_URL in .env file
- Test connection: `psql -h localhost -U postgres ai_conspects`

**Redis Connection Error:**
- If using WSL2: `wsl redis-server --daemonize yes`
- If using Redis Cloud: Update REDIS_URL in .env

**Ollama Not Found:**
- Verify Ollama is running: `ollama list`
- Pull model: `ollama pull llama3.1:8b`
- Check OLLAMA_BASE_URL in .env

## 🚦 Running in Production

### Using Windows Service
To run as a Windows service, consider using:
- **NSSM** (Non-Sucking Service Manager)
- **Windows Task Scheduler**

### Process Management
```powershell
# Create startup script: start_server.bat
@echo off
cd /d "C:\Projects\ai-conspects-server"
call venv\Scripts\activate.bat
python run.py
```

## 📊 Performance Monitoring

Monitor GPU usage during audio processing:
- Task Manager -> Performance -> GPU
- NVIDIA GPU-Z
- `nvidia-smi` command

## 🔄 Updates

To update the server:
```powershell
# Pull latest changes
git pull origin master

# Update dependencies if needed
pip install -r requirements.txt --upgrade

# Restart server
python run.py
```

## 🎯 Next Steps

Once running successfully:
1. ✅ Server accessible at `http://62.140.252.238:5000`
2. ✅ iOS app can connect and authenticate
3. ✅ Audio processing uses RTX 4090 GPU
4. ✅ AI features work with Ollama/OpenRouter
5. ✅ Database stores notes and relationships

Your AI Conspects app will now have full functionality with GPU acceleration!