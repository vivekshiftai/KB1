"""
PDF Intelligence Platform - Rules Generation Endpoint
Handles generation of IoT monitoring rules from PDF content

Version: 0.1
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Path
from services.vector_db import VectorDatabase
from services.llm_service import LLMService
from utils.helpers import sanitize_filename, calculate_processing_time, optimize_chunks_for_llm, estimate_llm_response_time
from models.schemas import RulesResponse
from config import settings

router = APIRouter(prefix="/generate-rules", tags=["rules"])
logger = logging.getLogger(__name__)

@router.post("/{pdf_name}", response_model=RulesResponse)
async def generate_rules(pdf_name: str = Path(..., description="Name of the PDF file")):
    """Generate IoT monitoring rules from PDF content"""
    start_time = time.time()
    
    logger.info(f"Generating rules for PDF: {pdf_name}")
    
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
        
        # Get chunks from vector database
        logger.info("Retrieving chunks from vector database...")
        all_chunks = await vector_db.get_all_chunks(
            collection_name=collection_name,
            limit=settings.max_chunks_per_batch
        )
        
        if not all_chunks:
            raise HTTPException(
                status_code=404,
                detail="No content found in PDF"
            )
        
        logger.info(f"Retrieved {len(all_chunks)} chunks from vector database")
        
        # Optimize chunks for LLM processing to reduce latency
        logger.info("Optimizing chunks for LLM processing...")
        optimized_chunks = optimize_chunks_for_llm(
            chunks=all_chunks,
            max_chunks=10,  # Limit to 10 chunks for faster processing
            max_tokens_per_chunk=2000  # Limit chunk size
        )
        
        logger.info(f"Optimized to {len(optimized_chunks)} chunks for processing")
        
        # Estimate response time
        estimated_time = estimate_llm_response_time(optimized_chunks)
        logger.info(f"Estimated LLM response time: {estimated_time:.1f}s")
        
        # Generate rules using LLM with optimized chunks
        logger.info("Generating rules with LLM...")
        rules = await llm_service.generate_rules(optimized_chunks)
        
        processing_time = calculate_processing_time(start_time)
        
        logger.info(f"Generated {len(rules)} rules in {processing_time}")
        
        return RulesResponse(
            success=True,
            pdf_name=pdf_name,
            rules=rules,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating rules: {str(e)}")
        processing_time = calculate_processing_time(start_time)
        
        # Check if it's a timeout error
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            raise HTTPException(
                status_code=408, 
                detail=f"Request timed out after {processing_time}. The LLM service took too long to respond. Please try again with a smaller document or contact support if the issue persists."
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Rule generation failed: {str(e)}"
            )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Rules Generation"}