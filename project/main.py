"""
PDF Intelligence Platform - Main Application
FastAPI application for PDF processing and intelligent querying

Version: 0.1
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import settings
from utils.helpers import setup_logging
from utils.cpu_optimizer import optimize_for_cpu
from endpoints import upload, query, pdfs, rules, maintenance, safety, images


# Setup logging
logger = setup_logging(settings.log_level)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting PDF Intelligence Platform...")
    
    # Force CPU mode for all operations
    optimize_for_cpu()
    
    logger.info(f"Vector DB Type: {settings.vector_db_type}")
    logger.info(f"Models Directory: {settings.models_dir}")
    logger.info(f"Upload Directory: {settings.upload_dir}")
    logger.info(f"Output Directory: {settings.output_dir}")
    logger.info("✅ CPU mode enforced - All operations will use CPU")
    
    yield
    
    logger.info("Shutting down PDF Intelligence Platform...")

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

# Timeout middleware
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Add timeout handling to all requests"""
    try:
        # Set timeout based on endpoint
        timeout = settings.request_timeout
        if "upload" in request.url.path:
            timeout = settings.upload_timeout
        elif "generate-rules" in request.url.path or "generate-maintenance" in request.url.path or "generate-safety" in request.url.path:
            timeout = settings.llm_timeout
        
        logger.info(f"Request to {request.url.path} with timeout: {timeout}s")
        
        # Execute request with timeout
        response = await asyncio.wait_for(call_next(request), timeout=timeout)
        return response
        
    except asyncio.TimeoutError:
        logger.error(f"Request to {request.url.path} timed out after {timeout}s")
        return JSONResponse(
            status_code=408,
            content={
                "success": False,
                "message": "Request timeout",
                "detail": f"Request took longer than {timeout} seconds to complete"
            }
        )
    except Exception as e:
        logger.error(f"Error in timeout middleware: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "detail": str(e) if settings.log_level.upper() == "DEBUG" else "An unexpected error occurred"
            }
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

@app.get("/performance")
async def performance_metrics():
    """Get LLM performance metrics"""
    from utils.helpers import llm_monitor
    
    stats = llm_monitor.get_performance_stats()
    is_degrading = llm_monitor.is_performance_degrading()
    
    return {
        "llm_performance": {
            "total_requests": stats["total_requests"],
            "average_response_time_seconds": round(stats["average_response_time"], 2),
            "error_rate_percent": round(stats["error_rate"], 2),
            "timeout_rate_percent": round(stats["timeout_rate"], 2),
            "performance_degrading": is_degrading,
            "recent_response_times": [round(t, 2) for t in stats["recent_response_times"]]
        },
        "recommendations": {
            "optimize_chunks": is_degrading or stats["average_response_time"] > 30,
            "reduce_context_size": stats["average_response_time"] > 45,
            "check_azure_service": stats["timeout_rate"] > 10
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "detail": str(exc) if settings.log_level.upper() == "DEBUG" else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )