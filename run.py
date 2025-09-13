#!/usr/bin/env python3
"""
Simple startup script for PDF Intelligence Platform
Run this script to start the FastAPI application

Version: 0.1
"""

import uvicorn
import os
import sys

# Force CPU mode before importing any ML libraries
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "debug")
    
    print("🚀 Starting PDF Intelligence Platform...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"📝 Log Level: {log_level}")
    print("✅ CPU mode enforced - All operations will use CPU")
    print("=" * 50)
    
    # Start the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=["."] if reload else None,  # Only watch current directory
        reload_excludes=[
            "vector_db/**",  # Exclude vector database files
            "processed/**",  # Exclude processed PDF files
            "uploads/**",    # Exclude uploaded files
            "pdf_extract_kit_models/**",  # Exclude model files
            "*.log",         # Exclude log files
            "*.tmp",         # Exclude temporary files
            "__pycache__/**", # Exclude Python cache
            "*.pyc",         # Exclude compiled Python files
            ".git/**",       # Exclude git files
            "node_modules/**" # Exclude node modules if any
        ] if reload else None,
        log_level=log_level
    )
