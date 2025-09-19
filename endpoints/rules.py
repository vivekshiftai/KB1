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
from utils.helpers import sanitize_filename, calculate_processing_time
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
        
        # Get chunks from vector database with smart filtering for IoT/monitoring content
        logger.info("Retrieving IoT/monitoring-relevant chunks from vector database...")
        
        # Define IoT/monitoring-related keywords for smart filtering
        iot_keywords = [
            "temperature pressure vibration speed", "monitor sensor threshold limit", 
            "alarm alert warning critical", "measurement reading value range"
        ]
        
        # Create combined query from all keywords
        combined_query = " ".join(iot_keywords)
        logger.info(f"Searching with combined query: {combined_query}")
        
        # Query for IoT/monitoring-relevant chunks using combined keywords
        try:
            iot_chunks = await vector_db.query_chunks(
                collection_name=collection_name,
                query=combined_query,
                top_k=12  # Fetch top 12 chunks in single query
            )
            logger.info(f"Found {len(iot_chunks)} chunks for combined IoT query")
        except Exception as e:
            logger.warning(f"Error querying with combined keywords: {str(e)}")
            iot_chunks = []
        
        # Take top 12 chunks (no deduplication needed since single query)
        top_iot_chunks = iot_chunks[:12]
        
        logger.info(f"Selected {len(top_iot_chunks)} IoT/monitoring-relevant chunks from single query")
        
        if not top_iot_chunks:
            logger.warning("No IoT/monitoring-relevant chunks found, falling back to general chunks")
            # Fallback to general chunks if no IoT-specific content found
            top_iot_chunks = await vector_db.get_all_chunks(
                collection_name=collection_name,
                limit=12
            )
        
        if not top_iot_chunks:
            raise HTTPException(
                status_code=404,
                detail="No content found in PDF"
            )
        
        logger.info(f"Processing {len(top_iot_chunks)} chunks for rule generation")
        
        # Process chunks in batches of 4
        all_rules = []
        chunk_batches = [top_iot_chunks[i:i+4] for i in range(0, len(top_iot_chunks), 4)]
        
        logger.info(f"Processing {len(chunk_batches)} batches of 4 chunks each")
        
        for batch_num, chunk_batch in enumerate(chunk_batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(chunk_batches)} with {len(chunk_batch)} chunks")
            
            # Generate rules for this batch
            batch_rules = await llm_service.generate_rules(chunk_batch)
            
            if batch_rules:
                all_rules.extend(batch_rules)
                logger.info(f"Batch {batch_num} generated {len(batch_rules)} rules")
            else:
                logger.warning(f"Batch {batch_num} generated no rules")
        
        rules_data = all_rules
        
        processing_time = calculate_processing_time(start_time)
        
        logger.info(f"Generated {len(rules_data)} rules in {processing_time}")
        
        return RulesResponse(
            success=True,
            message="Rules generated successfully from PDF analysis",
            rules=rules_data,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating rules: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Rule generation failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Rules Generation"}