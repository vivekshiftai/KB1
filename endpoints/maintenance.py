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
        
        # Define maintenance-specific keywords for smart filtering
        maintenance_keywords = [
            "maintenance list", "maintenance tasks", "maintenance schedules", "maintenance procedure"
        ]
        
        # Create combined query from all keywords
        combined_query = " ".join(maintenance_keywords)
        logger.info(f"Searching with combined query: {combined_query}")
        
        # Query for maintenance-relevant chunks using combined keywords
        try:
            maintenance_chunks = await vector_db.query_chunks(
                collection_name=collection_name,
                query=combined_query,
                top_k=12  # Fetch top 12 chunks in single query
            )
            logger.info(f"Found {len(maintenance_chunks)} chunks for combined maintenance query")
        except Exception as e:
            logger.warning(f"Error querying with combined keywords: {str(e)}")
            maintenance_chunks = []
        
        # Take top 12 chunks (no deduplication needed since single query)
        top_maintenance_chunks = maintenance_chunks[:12]
        
        logger.info(f"Selected {len(top_maintenance_chunks)} maintenance-relevant chunks from single query")
        
        if not top_maintenance_chunks:
            logger.warning("No maintenance-relevant chunks found, falling back to general chunks")
            # Fallback to general chunks if no maintenance-specific content found
            top_maintenance_chunks = await vector_db.get_all_chunks(
                collection_name=collection_name,
                limit=10
            )
        
        if not top_maintenance_chunks:
            raise HTTPException(
                status_code=404,
                detail="No content found in PDF"
            )
        
        logger.info(f"Processing {len(top_maintenance_chunks)} chunks for maintenance schedule generation")
        
        # Process chunks in batches of 4
        all_maintenance_tasks = []
        chunk_batches = [top_maintenance_chunks[i:i+4] for i in range(0, len(top_maintenance_chunks), 4)]
        
        logger.info(f"Processing {len(chunk_batches)} batches of 4 chunks each")
        
        for batch_num, chunk_batch in enumerate(chunk_batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(chunk_batches)} with {len(chunk_batch)} chunks")
            
            # Generate maintenance schedule for this batch
            batch_tasks = await llm_service.generate_maintenance_schedule(chunk_batch)
            
            if batch_tasks:
                all_maintenance_tasks.extend(batch_tasks)
                logger.info(f"Batch {batch_num} generated {len(batch_tasks)} tasks")
            else:
                logger.warning(f"Batch {batch_num} generated no tasks")
        
        maintenance_tasks_data = all_maintenance_tasks
        
        # Log the generated tasks for debugging
        logger.info(f"Generated {len(maintenance_tasks_data)} maintenance tasks")
        for i, task in enumerate(maintenance_tasks_data):
            logger.info(f"Task {i+1}: {task.get('task', 'Unknown')} (Category: {task.get('category', 'Unknown')}, Priority: {task.get('priority', 'Unknown')})")
        
        processing_time = calculate_processing_time(start_time)
        
        logger.info(f"Generated {len(maintenance_tasks_data)} maintenance tasks in {processing_time}")
        
        return MaintenanceResponse(
            success=True,
            message="Maintenance tasks generated successfully from PDF analysis",
            maintenance_tasks=maintenance_tasks_data,
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