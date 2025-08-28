"""
PDF Intelligence Platform - Maintenance Generation Endpoint
Handles generation of maintenance schedules from PDF content

Version: 0.1
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Path
from services.vector_db import VectorDatabase
from services.llm_service import LLMService
from utils.helpers import sanitize_filename, calculate_processing_time
from models.schemas import MaintenanceResponse
from config import settings

router = APIRouter(prefix="/generate-maintenance", tags=["maintenance"])
logger = logging.getLogger(__name__)

@router.post("/{pdf_name}", response_model=MaintenanceResponse)
async def generate_maintenance_schedule(pdf_name: str = Path(..., description="Name of the PDF file")):
    """Generate maintenance schedule from PDF content"""
    start_time = time.time()
    
    logger.info(f"Generating maintenance schedule for PDF: {pdf_name}")
    
    try:
        # Initialize services
        vector_db = VectorDatabase()
        llm_service = LLMService()
        
        # Generate collection name
        collection_name = sanitize_filename(pdf_name)
        
        # Check if collection exists
        if not vector_db.collection_exists(collection_name):
            raise HTTPException(
                status_code=404,
                detail=f"PDF '{pdf_name}' not found. Please upload the PDF first."
            )
        
        # Get chunks from vector database with smart filtering for maintenance content
        logger.info("Retrieving maintenance-relevant chunks from vector database...")
        
        # Define maintenance-related keywords for smart filtering
        maintenance_keywords = [
            "maintenance", "service", "inspection", "check", "clean", "lubricate", "calibrate",
            "replace", "repair", "adjust", "tighten", "monitor", "test", "verify", "examine",
            "schedule", "routine", "preventive", "periodic", "daily", "weekly", "monthly", "annual",
            "filter", "oil", "grease", "bearing", "belt", "motor", "pump", "valve", "sensor",
            "temperature", "pressure", "vibration", "noise", "wear", "damage", "failure",
            "safety", "warning", "caution", "procedure", "instruction", "manual", "guide"
        ]
        
        # Query for maintenance-relevant chunks using keywords
        maintenance_chunks = []
        for keyword in maintenance_keywords[:10]:  # Use top 10 keywords for comprehensive coverage
            try:
                keyword_chunks = await vector_db.query_chunks(
                    collection_name=collection_name,
                    query=keyword,
                    top_k=9  # Fetch top 9 chunks per keyword
                )
                maintenance_chunks.extend(keyword_chunks)
                logger.info(f"Found {len(keyword_chunks)} chunks for keyword: {keyword}")
            except Exception as e:
                logger.warning(f"Error querying for keyword '{keyword}': {str(e)}")
                continue
        
        # Remove duplicates and get top chunks
        unique_chunks = []
        seen_chunks = set()
        for chunk in maintenance_chunks:
            chunk_id = chunk.get("metadata", {}).get("chunk_index", "")
            if chunk_id not in seen_chunks:
                unique_chunks.append(chunk)
                seen_chunks.add(chunk_id)
        
        # Sort by relevance (lower distance = higher relevance) and take top 9
        unique_chunks.sort(key=lambda x: x.get("distance", 1.0))
        top_maintenance_chunks = unique_chunks[:9]  # Take top 9 chunks
        
        logger.info(f"Found {len(top_maintenance_chunks)} unique maintenance-relevant chunks out of {len(maintenance_chunks)} total chunks")
        
        if not top_maintenance_chunks:
            logger.warning("No maintenance-relevant chunks found, falling back to general chunks")
            # Fallback to general chunks if no maintenance-specific content found
            top_maintenance_chunks = await vector_db.get_all_chunks(
                collection_name=collection_name,
                limit=9
            )
        
        if not top_maintenance_chunks:
            raise HTTPException(
                status_code=404,
                detail="No content found in PDF"
            )
        
        logger.info(f"Processing {len(top_maintenance_chunks)} chunks for maintenance schedule generation")
        
        # Generate maintenance schedule using LLM with enhanced prompt
        logger.info("Generating maintenance schedule with LLM...")
        maintenance_tasks = await llm_service.generate_maintenance_schedule(top_maintenance_chunks)
        
        processing_time = calculate_processing_time(start_time)
        
        logger.info(f"Generated {len(maintenance_tasks)} maintenance tasks in {processing_time}")
        
        return MaintenanceResponse(
            success=True,
            pdf_name=pdf_name,
            maintenance_tasks=maintenance_tasks,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating maintenance schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Maintenance schedule generation failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Maintenance Generation"}