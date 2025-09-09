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
            "temperature", "pressure", "vibration", "speed", "flow", "level", "voltage", "current",
            "monitor", "sensor", "threshold", "limit", "alarm", "alert", "warning", "critical",
            "measurement", "reading", "value", "range", "maximum", "minimum", "normal", "abnormal",
            "performance", "efficiency", "output", "input", "signal", "data", "status", "condition",
            "operational", "functional", "working", "running", "stopped", "error", "fault", "failure"
        ]
        
        # Query for IoT/monitoring-relevant chunks using keywords
        iot_chunks = []
        for keyword in iot_keywords[:8]:  # Use top 8 keywords to get diverse content
            try:
                keyword_chunks = await vector_db.query_chunks(
                    collection_name=collection_name,
                    query=keyword,
                    top_k=3
                )
                iot_chunks.extend(keyword_chunks)
                logger.info(f"Found {len(keyword_chunks)} chunks for keyword: {keyword}")
            except Exception as e:
                logger.warning(f"Error querying for keyword '{keyword}': {str(e)}")
                continue
        
        # Remove duplicates and get top 10 most relevant chunks
        unique_chunks = []
        seen_chunks = set()
        for chunk in iot_chunks:
            chunk_id = chunk.get("metadata", {}).get("chunk_index", "")
            if chunk_id not in seen_chunks:
                unique_chunks.append(chunk)
                seen_chunks.add(chunk_id)
        
        # Sort by relevance (lower distance = higher relevance) and take top 10
        unique_chunks.sort(key=lambda x: x.get("distance", 1.0))
        top_iot_chunks = unique_chunks[:10]
        
        logger.info(f"Found {len(top_iot_chunks)} unique IoT/monitoring-relevant chunks out of {len(iot_chunks)} total chunks")
        
        if not top_iot_chunks:
            logger.warning("No IoT/monitoring-relevant chunks found, falling back to general chunks")
            # Fallback to general chunks if no IoT-specific content found
            top_iot_chunks = await vector_db.get_all_chunks(
                collection_name=collection_name,
                limit=10
            )
        
        if not top_iot_chunks:
            raise HTTPException(
                status_code=404,
                detail="No content found in PDF"
            )
        
        logger.info(f"Processing {len(top_iot_chunks)} chunks for rule generation")
        
        # Generate rules using LLM with enhanced prompt
        logger.info("Generating rules with LLM...")
        rules = await llm_service.generate_rules(top_iot_chunks)
        
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
        raise HTTPException(status_code=500, detail=f"Rule generation failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Rules Generation"}