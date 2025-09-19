"""
Simple AI Conspects Server for testing without database
"""

import platform
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Conspects Server",
    description="AI-powered note generation from audio recordings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with server information"""
    system = platform.system().lower()
    port = 8000 if system == "darwin" else 5000
    
    return {
        "message": "AI Conspects Server",
        "version": "1.0.0",
        "platform": system.title(),
        "port": port,
        "status": "running",
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

@app.get("/api/v1/test")
async def test_endpoint():
    """Test endpoint for iOS app"""
    return {
        "message": "AI Conspects Server is ready!",
        "endpoints": {
            "auth": "/api/v1/auth/register",
            "audio": "/api/v1/batches/submit", 
            "results": "/api/v1/results/notes",
            "chat": "/api/v1/chat/query"
        },
        "llm_modes": ["ollama", "openrouter"],
        "whisper_model": "large-v3"
    }

if __name__ == "__main__":
    # Determine port based on platform
    system = platform.system().lower()
    port = 8000 if system == "darwin" else 5000
    
    logger.info(f"🚀 Starting AI Conspects Server on {system.title()} - Port: {port}")
    logger.info(f"📊 Server will be available at: http://localhost:{port}")
    logger.info(f"📚 API Documentation: http://localhost:{port}/docs")
    logger.info(f"❤️  Health Check: http://localhost:{port}/health")
    
    uvicorn.run(
        "simple_server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
