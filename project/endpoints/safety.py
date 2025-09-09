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
        
        # Define safety-related keywords for smart filtering
        safety_keywords = [
            "safety", "warning", "caution", "danger", "hazard", "risk", "emergency", "protective",
            "injury", "damage", "failure", "malfunction", "overheating", "overload", "pressure",
            "temperature", "voltage", "current", "shock", "burn", "cut", "crush", "fall",
            "ppe", "helmet", "gloves", "goggles", "mask", "vest", "boots", "ear protection",
            "lockout", "tagout", "isolation", "depressurize", "de-energize", "ventilation",
            "flammable", "explosive", "toxic", "corrosive", "radiation", "noise", "vibration",
            "maintenance", "repair", "installation", "operation", "startup", "shutdown",
            "procedure", "instruction", "manual", "guide", "protocol", "standard", "regulation"
        ]
        
        # Query for safety-relevant chunks using keywords
        safety_chunks = []
        for keyword in safety_keywords[:12]:  # Use top 12 keywords to get diverse content
            try:
                keyword_chunks = await vector_db.query_chunks(
                    collection_name=collection_name,
                    query=keyword,
                    top_k=4
                )
                safety_chunks.extend(keyword_chunks)
                logger.info(f"Found {len(keyword_chunks)} chunks for keyword: {keyword}")
            except Exception as e:
                logger.warning(f"Error querying for keyword '{keyword}': {str(e)}")
                continue
        
        # Remove duplicates and get top 20 most relevant chunks
        unique_chunks = []
        seen_chunks = set()
        for chunk in safety_chunks:
            chunk_id = chunk.get("metadata", {}).get("chunk_index", "")
            if chunk_id not in seen_chunks:
                unique_chunks.append(chunk)
                seen_chunks.add(chunk_id)
        
        # Sort by relevance (lower distance = higher relevance) and take top 20
        unique_chunks.sort(key=lambda x: x.get("distance", 1.0))
        top_safety_chunks = unique_chunks[:20]
        
        logger.info(f"Found {len(top_safety_chunks)} unique safety-relevant chunks out of {len(safety_chunks)} total chunks")
        
        if not top_safety_chunks:
            logger.warning("No safety-relevant chunks found, falling back to general chunks")
            # Fallback to general chunks if no safety-specific content found
            top_safety_chunks = await vector_db.get_all_chunks(
                collection_name=collection_name,
                limit=20
            )
        
        if not top_safety_chunks:
            raise HTTPException(
                status_code=404,
                detail="No content found in PDF"
            )
        
        logger.info(f"Processing {len(top_safety_chunks)} chunks for safety information generation")
        
        # Generate safety information using LLM with enhanced prompt
        logger.info("Generating safety information with LLM...")
        safety_information = await llm_service.generate_safety_information(top_safety_chunks)
        
        processing_time = calculate_processing_time(start_time)
        
        logger.info(f"Generated {len(safety_information)} safety items in {processing_time}")
        
        return SafetyResponse(
            success=True,
            pdf_name=pdf_name,
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