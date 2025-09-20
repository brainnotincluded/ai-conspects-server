"""
Simple Configuration for Windows Deployment
All settings needed to run the server on Windows
"""

import os
from pathlib import Path

class WindowsConfig:
    """Windows-specific configuration"""
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 5000
    
    # JWT Settings
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-in-production")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRE_MINUTES = 1440  # 24 hours
    
    # Database (SQLite for simplicity)
    DATABASE_URL = "sqlite:///ai_conspects.db"
    
    # File storage
    UPLOAD_DIR = Path("uploads")
    LOGS_DIR = Path("logs")
    
    # GPU settings (if available)
    USE_GPU = True  # Will automatically detect GPU availability
    WHISPER_MODEL = "large-v2"  # Use large model if GPU available, else base
    
    # AI API Keys (set these in environment variables)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE = 100
    
    # CORS settings
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # Allow all for development
    ]
    
    def __init__(self):
        # Create required directories
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.LOGS_DIR.mkdir(exist_ok=True)
        
        # Check GPU availability
        try:
            import torch
            self.GPU_AVAILABLE = torch.cuda.is_available()
            if self.GPU_AVAILABLE:
                self.GPU_NAME = torch.cuda.get_device_name(0)
                print(f"✅ GPU detected: {self.GPU_NAME}")
            else:
                print("⚠️ No GPU detected, using CPU")
        except ImportError:
            self.GPU_AVAILABLE = False
            print("⚠️ PyTorch not installed, GPU support disabled")

# Global config instance
config = WindowsConfig()