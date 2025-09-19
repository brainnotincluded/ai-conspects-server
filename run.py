#!/usr/bin/env python3
"""
Run script for AI Conspects Server
Handles platform detection and port selection with automatic fallback
"""

import os
import sys
import platform
import socket
import uvicorn
from config import settings

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
        print(f"✅ Port {preferred_port} is available - using preferred port")
        return preferred_port
    elif is_port_available(host, fallback_port):
        print(f"⚠️  Port {preferred_port} is busy - using fallback port {fallback_port}")
        return fallback_port
    else:
        print(f"❌ Both ports {preferred_port} and {fallback_port} are busy!")
        print("   Please free up one of these ports or modify the port selection logic.")
        sys.exit(1)

def main():
    """Main entry point"""
    
    # Select available port automatically
    port = select_available_port(settings.HOST)
    system = platform.system().lower()
    print(f"🚀 Starting AI Conspects Server on {system.title()} - Port: {port}")
    
    # Print configuration
    print(f"📊 Configuration:")
    print(f"   - Host: {settings.HOST}")
    print(f"   - Port: {port}")
    print(f"   - Platform: {system.title()}")
    print(f"   - Database: {settings.DATABASE_URL}")
    print(f"   - Redis: {settings.REDIS_URL}")
    print(f"   - Whisper Model: {settings.WHISPER_MODEL}")
    print(f"   - Max Batch Size: {settings.MAX_BATCH_SIZE}")
    
    # Check for required environment variables
    missing_vars = []
    if not settings.OPENAI_API_KEY and not settings.OPENROUTER_API_KEY:
        missing_vars.append("OPENAI_API_KEY or OPENROUTER_API_KEY")
    if not settings.JWT_SECRET_KEY or settings.JWT_SECRET_KEY == "your-secret-key-change-in-production":
        missing_vars.append("JWT_SECRET_KEY")
    
    if missing_vars:
        print(f"⚠️  Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("   Some features may not work properly.")
    
    print(f"\n🌐 Server will be available at: http://{settings.HOST}:{port}")
    print(f"📚 API Documentation: http://{settings.HOST}:{port}/docs")
    print(f"❤️  Health Check: http://{settings.HOST}:{port}/health")
    print("\n" + "="*50)
    
    # Start the server
    try:
        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=port,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down AI Conspects Server...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
