import time
import logging
from fastapi import APIRouter, HTTPException
from services.vector_db import VectorDatabase
from services.llm_service import LLMService
from utils.helpers import sanitize_filename, calculate_processing_time
from models.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger(__name__)

@router.post("", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    """Query PDF content with intelligent response generation"""
    start_time = time.time()
    
    logger.info(f"Processing query for PDF: {request.pdf_name}")
    logger.info(f"Query: {request.query}")
    
    try:
        # Initialize services
        vector_db = VectorDatabase()
        llm_service = LLMService()
        
        # Generate collection name
        collection_name = sanitize_filename(request.pdf_name)
        
        # Check if collection exists
        if not vector_db.collection_exists(collection_name):
            raise HTTPException(
                status_code=404, 
                detail=f"PDF '{request.pdf_name}' not found. Please upload the PDF first."
            )
        
        # Query vector database
        logger.info("Querying vector database...")
        chunks = await vector_db.query_chunks(
            collection_name=collection_name,
            query=request.query,
            top_k=request.top_k
        )
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant content found for the query"
            )
        
        # Generate response using LLM
        logger.info("Generating response with LLM...")
        llm_result = await llm_service.query_with_context(chunks, request.query)
        
        # Collect all images and tables from used chunks
        all_images = []
        all_tables = []
        
        for chunk in chunks:
            all_images.extend(chunk.get("images", []))
            all_tables.extend(chunk.get("tables", []))
        
        # Remove duplicates
        all_images = list(set(all_images))
        all_tables = list(set(all_tables))
        
        processing_time = calculate_processing_time(start_time)
        
        logger.info(f"Query processed successfully in {processing_time}")
        
        return QueryResponse(
            success=True,
            message="Query processed successfully",
            response=llm_result["response"],
            chunks_used=llm_result["chunks_used"],
            images=all_images,
            tables=all_tables,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Query"}