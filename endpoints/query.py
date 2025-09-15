"""
PDF Intelligence Platform - Query Endpoint
Handles intelligent querying of PDF content with LangGraph workflow

Version: 0.1
"""

import time
import logging
from fastapi import APIRouter, HTTPException
from services.vector_db import VectorDatabase
from services.llm_service import LLMService
from services.langgraph_query_processor import LangGraphQueryProcessor
from utils.helpers import sanitize_filename, calculate_processing_time
from models.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger(__name__)

@router.post("", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    """Query PDF content with LangGraph workflow and response validation
    
    Note: All query lengths are accepted, including queries with 10 characters or less.
    No length restrictions or filters are applied.
    """
    start_time = time.time()
    
    logger.info(f"Processing query for PDF: {request.pdf_name}")
    logger.info(f"Query: {request.query}")
    logger.info(f"Query length: {len(request.query)} characters")
    logger.info(f"Query validation: ACCEPTED (no length restrictions)")
    
    try:
        # Initialize LangGraph query processor
        processor = LangGraphQueryProcessor()
        
        # Process query using LangGraph workflow
        response = await processor.process_query(request)
        
        logger.info(f"Query processed successfully in {response.processing_time}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Query"}