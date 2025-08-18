#!/usr/bin/env python3
"""
Simple startup script for PDF Intelligence Platform
Run this script to start the FastAPI application
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    
    print("🚀 Starting PDF Intelligence Platform...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"📝 Log Level: {log_level}")
    print("=" * 50)
    
    # Start the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )
