#!/usr/bin/env python3
"""
Startup script for PDF Intelligence Platform
This script can be used by PM2 to start the FastAPI application
"""

import uvicorn
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    
    print(f"Starting PDF Intelligence Platform on {host}:{port}")
    print(f"Reload mode: {reload}")
    print(f"Log level: {log_level}")
    
    # Start the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )
