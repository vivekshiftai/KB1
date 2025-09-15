"""
PDF Intelligence Platform - Main Application
FastAPI application for PDF processing and intelligent querying

Version: 0.1
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import settings
from utils.helpers import setup_logging
from utils.cpu_optimizer import optimize_for_cpu
from endpoints import upload, query, pdfs, rules, maintenance, safety, images


# Setup thread-safe logging
logger = setup_logging(settings.log_level)

# Test logging immediately
logger.info("=== APPLICATION STARTING - Thread-Safe Logging Active ===")
logger.info(f"Log level configured: {settings.log_level}")
logger.info(f"Thread safety: ENABLED")
logger.info(f"Concurrent API handling: ENABLED")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with thread safety verification"""
    import threading
    
    logger.info("=== STARTING PDF INTELLIGENCE PLATFORM ===")
    logger.info(f"Thread: {threading.current_thread().name}")
    logger.info(f"Thread ID: {threading.get_ident()}")
    print("🚀 PDF Intelligence Platform starting...")  # Also print to console
    
    # Force CPU mode for all operations
    optimize_for_cpu()
    
    logger.info(f"Vector DB Type: {settings.vector_db_type}")
    logger.info(f"Models Directory: {settings.models_dir}")
    logger.info(f"Upload Directory: {settings.upload_dir}")
    logger.info(f"Output Directory: {settings.output_dir}")
    logger.info("✅ CPU mode enforced - All operations will use CPU")
    
    # Test logging from services to verify thread-safe logging
    from services.llm_service import logger as llm_logger
    from services.vector_db import logger as vector_logger
    llm_logger.info("LLM Service logger is working - Thread-safe")
    vector_logger.info("Vector DB Service logger is working - Thread-safe")
    
    # Log thread safety features
    logger.info("=== THREAD SAFETY FEATURES ACTIVE ===")
    logger.info("✅ Thread-safe logging enabled")
    logger.info("✅ Concurrent API request handling enabled")
    logger.info("✅ Per-collection locks for vector DB operations")
    logger.info("✅ Per-model locks for LLM service operations")
    logger.info("✅ API request semaphore limiting (max 5 concurrent)")
    logger.info("✅ Thread-safe embedding model singleton")
    
    yield
    
    logger.info("=== SHUTTING DOWN PDF INTELLIGENCE PLATFORM ===")
    logger.info("Thread-safe cleanup completed")

# Initialize FastAPI app
app = FastAPI(
    title="PDF Intelligence Platform",
    description="A comprehensive backend API system that processes PDF manuals, stores them in vector databases, and provides intelligent querying capabilities for IoT device documentation, rules generation, maintenance schedules, and safety information.",
    version="0.1",
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
app.include_router(upload.router)
app.include_router(query.router)
app.include_router(pdfs.router)
app.include_router(rules.router)
app.include_router(maintenance.router)
app.include_router(safety.router)
app.include_router(images.router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "PDF Intelligence Platform",
        "version": "0.1",
        "status": "operational",
        "endpoints": {
            "upload": "/upload-pdf",
            "query": "/query",
            "list_pdfs": "/pdfs",
            "generate_rules": "/generate-rules/{pdf_name}",
            "generate_maintenance": "/generate-maintenance/{pdf_name}",
            "generate_safety": "/generate-safety/{pdf_name}",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Global health check endpoint"""
    return {
        "status": "healthy",
        "service": "PDF Intelligence Platform",
        "version": "0.1",
        "components": {
            "upload": "healthy",
            "query": "healthy",
            "pdfs": "healthy",
            "rules": "healthy",
            "maintenance": "healthy",
            "safety": "healthy"
        }
    }

@app.get("/test-logs")
async def test_logs():
    """Test endpoint to verify thread-safe logging is working"""
    import threading
    import asyncio
    
    current_thread = threading.current_thread()
    logger.info(f"=== TESTING THREAD-SAFE LOGGING ===")
    logger.info(f"Current thread: {current_thread.name} (ID: {threading.get_ident()})")
    logger.info("Test INFO message from main logger")
    logger.warning("Test WARNING message from main logger")
    logger.error("Test ERROR message from main logger")
    
    # Test service loggers
    from services.llm_service import logger as llm_logger
    from services.vector_db import logger as vector_logger
    
    llm_logger.info(f"Test INFO message from LLM service (Thread: {current_thread.name})")
    vector_logger.info(f"Test INFO message from Vector DB service (Thread: {current_thread.name})")
    
    # Test concurrent logging simulation
    async def concurrent_log_test():
        await asyncio.sleep(0.1)  # Simulate some async work
        logger.info(f"Concurrent log from async function (Thread: {threading.current_thread().name})")
    
    # Run concurrent logging test
    await concurrent_log_test()
    
    return {
        "message": "Thread-safe test log messages sent - check console and app.log file",
        "log_level": settings.log_level,
        "log_file": "app.log",
        "thread_safety": "ENABLED",
        "current_thread": current_thread.name,
        "thread_id": threading.get_ident()
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with thread-safe logging"""
    import threading
    
    current_thread = threading.current_thread()
    logger.error(f"Unhandled exception in thread {current_thread.name} (ID: {threading.get_ident()}): {str(exc)}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "detail": str(exc) if settings.log_level.upper() == "DEBUG" else "An unexpected error occurred",
            "thread_info": {
                "thread_name": current_thread.name,
                "thread_id": threading.get_ident()
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."],  # Only watch current directory
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
        ],
        log_level=settings.log_level.lower()
    )