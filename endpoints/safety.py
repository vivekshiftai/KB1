"""
PDF Intelligence Platform - Safety Information Generation Endpoint
Handles generation of safety information from PDF content

Version: 0.1
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Path
from services.vector_db import VectorDatabase
from services.llm_service import LLMService
from utils.helpers import sanitize_filename, calculate_processing_time
from models.schemas import SafetyResponse
from config import settings

router = APIRouter(prefix="/generate-safety", tags=["safety"])
logger = logging.getLogger(__name__)

@router.post("/{pdf_name}", response_model=SafetyResponse)
async def generate_safety_information(pdf_name: str = Path(..., description="Name of the PDF file")):
    """Generate safety information from PDF content"""
    start_time = time.time()
    
    logger.info(f"Generating safety information for PDF: {pdf_name}")
    
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
        
        # Get chunks from vector database with smart filtering for safety content
        logger.info("Retrieving safety-relevant chunks from vector database...")
        
        # Define safety-specific keywords for smart filtering
        safety_keywords = [
            "error codes precaution alert safety", "warning danger hazard risk", 
            "emergency stop prohibited caution", "protection equipment procedure"
        ]
        
        # Create combined query from all keywords
        combined_query = " ".join(safety_keywords)
        logger.info(f"Searching with combined query: {combined_query}")
        
        # Query for safety-relevant chunks using combined keywords
        try:
            safety_chunks = await vector_db.query_chunks(
                collection_name=collection_name,
                query=combined_query,
                top_k=12  # Fetch top 12 chunks in single query
            )
            logger.info(f"Found {len(safety_chunks)} chunks for combined safety query")
        except Exception as e:
            logger.warning(f"Error querying with combined keywords: {str(e)}")
            safety_chunks = []
        
        # Take top 12 chunks (no deduplication needed since single query)
        top_safety_chunks = safety_chunks[:12]
        
        logger.info(f"Selected {len(top_safety_chunks)} safety-relevant chunks from single query")
        
        if not top_safety_chunks:
            logger.warning("No safety-relevant chunks found, falling back to general chunks")
            # Fallback to general chunks if no safety-specific content found
            top_safety_chunks = await vector_db.get_all_chunks(
                collection_name=collection_name,
                limit=12
            )
        
        if not top_safety_chunks:
            raise HTTPException(
                status_code=404,
                detail="No content found in PDF"
            )
        
        logger.info(f"Processing {len(top_safety_chunks)} chunks for safety information generation")
        
        # Process chunks in batches of 4
        all_safety_data = {"safety_precautions": [], "safety_information": []}
        chunk_batches = [top_safety_chunks[i:i+4] for i in range(0, len(top_safety_chunks), 4)]
        
        logger.info(f"Processing {len(chunk_batches)} batches of 4 chunks each")
        
        for batch_num, chunk_batch in enumerate(chunk_batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(chunk_batches)} with {len(chunk_batch)} chunks")
            
            # Generate safety information for this batch
            batch_safety = await llm_service.generate_safety_information(chunk_batch)
            
            if batch_safety:
                # Merge safety data from this batch
                if "safety_precautions" in batch_safety:
                    all_safety_data["safety_precautions"].extend(batch_safety["safety_precautions"])
                if "safety_information" in batch_safety:
                    all_safety_data["safety_information"].extend(batch_safety["safety_information"])
                
                batch_precautions = len(batch_safety.get("safety_precautions", []))
                batch_info = len(batch_safety.get("safety_information", []))
                logger.info(f"Batch {batch_num} generated {batch_precautions} precautions and {batch_info} info items")
            else:
                logger.warning(f"Batch {batch_num} generated no safety data")
        
        safety_data = all_safety_data
        
        processing_time = calculate_processing_time(start_time)
        
        safety_precautions = safety_data.get("safety_precautions", [])
        safety_information = safety_data.get("safety_information", [])
        
        logger.info(f"Generated {len(safety_precautions)} safety precautions and {len(safety_information)} safety information items in {processing_time}")
        
        return SafetyResponse(
            success=True,
            message="Safety precautions generated successfully from PDF analysis",
            safety_precautions=safety_precautions,
            safety_information=safety_information,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating safety information: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Safety information generation failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Safety Generation"}