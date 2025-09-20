#!/usr/bin/env python3
"""
Windows Deployment Helper for AI Conspects Server
This script helps identify what version is running and provides deployment instructions
"""

import sys
import platform
import subprocess
import json
from pathlib import Path

def check_current_server():
    """Check what version is currently running"""
    print("🔍 ANALYZING CURRENT SERVER SETUP")
    print("=" * 50)
    
    # Check if we're on the right machine
    system = platform.system()
    print(f"📊 Current Platform: {system}")
    
    # Check if main dependencies are available
    print("\n📦 CHECKING DEPENDENCIES:")
    
    deps = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server', 
        'sqlalchemy': 'Database ORM',
        'whisper': 'Speech-to-text (OpenAI Whisper)',
        'torch': 'PyTorch (for AI models)',
        'sentence-transformers': 'Embedding models',
        'redis': 'Redis client',
        'psycopg2': 'PostgreSQL client',
        'ollama': 'Local LLM client'
    }
    
    missing_deps = []
    for dep, desc in deps.items():
        try:
            __import__(dep)
            print(f"  ✅ {dep} - {desc}")
        except ImportError:
            print(f"  ❌ {dep} - {desc} (MISSING)")
            missing_deps.append(dep)
    
    # Check GPU availability
    print(f"\n🎮 GPU CHECK:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"  ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("  ❌ CUDA not available")
    except ImportError:
        print("  ❌ PyTorch not installed")
    
    return missing_deps

def check_server_files():
    """Check what server files exist"""
    print(f"\n📁 CHECKING SERVER FILES:")
    
    files_to_check = [
        'main.py',
        'simple_server.py', 
        'run.py',
        'auth.py',
        'audio_processing.py',
        'chat.py',
        'results.py',
        'config.py',
        'database.py',
        '.env'
    ]
    
    current_dir = Path.cwd()
    print(f"  📂 Current directory: {current_dir}")
    
    existing_files = []
    for file in files_to_check:
        file_path = current_dir / file
        if file_path.exists():
            print(f"  ✅ {file}")
            existing_files.append(file)
        else:
            print(f"  ❌ {file} (missing)")
    
    return existing_files

def generate_deployment_commands():
    """Generate deployment commands for Windows"""
    print(f"\n🚀 DEPLOYMENT INSTRUCTIONS FOR WINDOWS:")
    print("=" * 50)
    
    print("1. INSTALL MISSING DEPENDENCIES:")
    print("   pip install fastapi uvicorn sqlalchemy whisper-openai")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   pip install sentence-transformers redis psycopg2-binary python-jose[cryptography]")
    print("   pip install celery pydantic-settings python-multipart boto3")
    
    print("\n2. SETUP DATABASE (PostgreSQL):")
    print("   - Install PostgreSQL on Windows")
    print("   - Create database: ai_conspects") 
    print("   - Update DATABASE_URL in .env file")
    
    print("\n3. SETUP REDIS:")
    print("   - Install Redis on Windows (or use Redis Cloud)")
    print("   - Update REDIS_URL in .env file")
    
    print("\n4. SETUP OLLAMA (for local LLM):")
    print("   - Download Ollama from https://ollama.ai")
    print("   - Install and run: ollama run llama3.1:8b")
    
    print("\n5. CREATE .env FILE:")
    env_content = """
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ai_conspects
REDIS_URL=redis://localhost:6379

# AI Services (add your API keys)
OPENAI_API_KEY=your_openai_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# JWT Security
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production

# Server
HOST=0.0.0.0
PORT_WINDOWS=5000
"""
    print("   Create .env file with:")
    print(env_content)
    
    print("\n6. START THE FULL SERVER:")
    print("   python run.py")
    print("   OR")
    print("   python main.py")
    
    print("\n7. VERIFY DEPLOYMENT:")
    print("   Test endpoints:")
    print("   - http://localhost:5000/")
    print("   - http://localhost:5000/health") 
    print("   - http://localhost:5000/docs")
    print("   - http://localhost:5000/api/v1/auth/register")

def check_what_is_running():
    """Check what server version is actually running"""
    print(f"\n🔍 CHECKING CURRENT RUNNING SERVER:")
    print("=" * 50)
    
    try:
        import requests
        
        # Test basic endpoint
        response = requests.get("http://localhost:5000/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is running: {data.get('message', 'Unknown')}")
            print(f"✅ Version: {data.get('version', 'Unknown')}")
            print(f"✅ Platform: {data.get('platform', 'Unknown')}")
            
        # Test API endpoints
        endpoints_to_test = [
            "/api/v1/auth/register",
            "/api/v1/batches/submit", 
            "/api/v1/chat/query",
            "/api/v1/results/notes"
        ]
        
        print(f"\n🧪 Testing API endpoints:")
        for endpoint in endpoints_to_test:
            try:
                resp = requests.get(f"http://localhost:5000{endpoint}", timeout=2)
                if resp.status_code == 404:
                    print(f"  ❌ {endpoint} - Not Found (simple server)")
                else:
                    print(f"  ✅ {endpoint} - Available")
            except:
                print(f"  ❌ {endpoint} - Connection failed")
                
    except ImportError:
        print("❌ requests library not available")
    except Exception as e:
        print(f"❌ Cannot connect to local server: {e}")

def main():
    """Main deployment checker"""
    print("🤖 AI CONSPECTS SERVER - DEPLOYMENT ANALYZER")
    print("=" * 60)
    
    # Run all checks
    missing_deps = check_current_server()
    existing_files = check_server_files()
    check_what_is_running()
    
    print(f"\n📋 SUMMARY:")
    print("=" * 30)
    
    if 'main.py' in existing_files and 'auth.py' in existing_files:
        print("✅ Full server code is available")
        if missing_deps:
            print(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
            print("   → Run pip install commands below")
        else:
            print("✅ All dependencies seem to be available")
            print("   → Try running: python run.py")
    else:
        print("❌ Full server code is missing")
        print("   → Copy the complete server code to this directory")
    
    # Always show deployment instructions
    generate_deployment_commands()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Install missing dependencies")
    print("2. Setup database and Redis")
    print("3. Create .env file with your configuration")  
    print("4. Run: python run.py")
    print("5. Test at http://localhost:5000/docs")
    print("6. Update port forwarding if needed")

if __name__ == "__main__":
    main()