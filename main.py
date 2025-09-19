"""
AI Conspects Server - Main Application
Supports Ollama and OpenRouter modes for note generation
Includes Whisper Large for speech-to-text and Perplexity API for research
"""

import os
import platform
import socket
import sys
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
import logging

from database import init_db
from auth import router as auth_router
from audio_processing import router as audio_router
from results import router as results_router
from chat import router as chat_router
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security = HTTPBearer()

def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False

def select_available_port(host: str) -> int:
    """Select the best available port (prefer 5000, fallback to 8000)"""
    preferred_port = 5000
    fallback_port = 8000
    
    if is_port_available(host, preferred_port):
        logger.info(f"✅ Port {preferred_port} is available - using preferred port")
        return preferred_port
    elif is_port_available(host, fallback_port):
        logger.info(f"⚠️  Port {preferred_port} is busy - using fallback port {fallback_port}")
        return fallback_port
    else:
        logger.error(f"❌ Both ports {preferred_port} and {fallback_port} are busy!")
        logger.error("   Please free up one of these ports or modify the port selection logic.")
        sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting AI Conspects Server...")
    await init_db()
    logger.info("Database initialized")
    
    # Log system info (port is already selected at this point)
    system = platform.system().lower()
    logger.info(f"Server configured for {system.title()}")
    logger.info("Server startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Conspects Server...")

# Create FastAPI app
app = FastAPI(
    title="AI Conspects Server",
    description="AI-powered note generation from audio recordings",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(audio_router, prefix="/api/v1")
app.include_router(results_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with server information"""
    system = platform.system().lower()
    # Note: Port is determined at startup, we'll show a generic response
    
    return {
        "message": "AI Conspects Server",
        "version": "1.0.0",
        "platform": system.title(),
        "port": "dynamic (5000 preferred, 8000 fallback)",
        "features": {
            "ollama_mode": True,
            "openrouter_mode": True,
            "whisper_large": True,
            "perplexity_research": True
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Server is running"}

if __name__ == "__main__":
    # Select available port automatically
    port = select_available_port(settings.HOST)
    system = platform.system().lower()
    
    logger.info(f"Starting server on {settings.HOST}:{port} ({system.title()})")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=port,
        reload=True,
        log_level="info"
    )
