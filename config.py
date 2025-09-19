"""
Configuration settings for AI Conspects Server
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/ai_conspects"
    REDIS_URL: str = "redis://localhost:6379"
    
    # Storage
    S3_BUCKET: str = "ai-conspects-audio"
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    S3_REGION: str = "us-east-1"
    
    # AI Services
    OPENAI_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    PERPLEXITY_API_KEY: Optional[str] = None
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    
    # Authentication
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT_MAC: int = 8000
    PORT_WINDOWS: int = 5000
    
    # Processing Settings
    MAX_BATCH_SIZE: int = 10
    MAX_RECORDING_DURATION: int = 600  # 10 minutes
    WHISPER_MODEL: str = "large-v3"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
