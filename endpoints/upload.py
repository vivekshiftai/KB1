import os
import time
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.pdf_processor import PDFProcessor
from services.vector_db import VectorDatabase
from utils.helpers import sanitize_filename, calculate_processing_time, validate_pdf_file
from models.schemas import PDFUploadResponse
from config import settings

router = APIRouter(prefix="/upload", tags=["upload"])
logger = logging.getLogger(__name__)

@router.post("/pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file"""
    start_time = time.time()
    
    # Validate file
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if file.size and file.size > settings.max_file_size:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size: {settings.max_file_size} bytes")
    
    logger.info(f"Processing PDF upload: {file.filename}")
    logger.info(f"File size: {file.size} bytes")
    logger.info(f"Content type: {file.content_type}")
    
    # Sanitize filename to handle special characters
    safe_filename = sanitize_filename(file.filename) + ".pdf"
    upload_path = Path(settings.upload_dir) / safe_filename
    
    try:
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"PDF saved to: {upload_path}")
        
        # Validate PDF
        if not validate_pdf_file(str(upload_path)):
            upload_path.unlink()  # Delete invalid file
            raise HTTPException(status_code=400, detail="Invalid PDF file - please ensure the file is a valid PDF document")
        
        logger.info(f"PDF validation successful for: {file.filename}")
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        if upload_path.exists():
            upload_path.unlink()  # Clean up on error
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    
    try:
        # Initialize services
        pdf_processor = PDFProcessor()
        vector_db = VectorDatabase()
        
        # Generate collection name from original filename
        collection_name = sanitize_filename(file.filename)
        
        # Check if collection already exists
        if vector_db.collection_exists(collection_name):
            logger.info(f"Collection {collection_name} already exists, skipping processing")
            processing_time = calculate_processing_time(start_time)
            return PDFUploadResponse(
                success=True,
                message="PDF already processed",
                pdf_name=file.filename,
                chunks_processed=0,
                processing_time=processing_time,
                collection_name=collection_name
            )
        
        # Prepare output directory
        output_dir = Path(settings.output_dir) / Path(file.filename).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process PDF
        logger.info("Starting PDF processing with MinerU...")
        chunks = await pdf_processor.process_pdf(str(upload_path), str(output_dir))
        
        if not chunks:
            raise HTTPException(status_code=500, detail="No content could be extracted from PDF")
        
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Store in vector database
        logger.info("Storing chunks in vector database...")
        chunks_stored = await vector_db.store_chunks(chunks, collection_name)
        
        # Cleanup uploaded file
        upload_path.unlink()
        
        processing_time = calculate_processing_time(start_time)
        
        logger.info(f"PDF processing completed successfully in {processing_time}")
        
        return PDFUploadResponse(
            success=True,
            message="PDF processed and stored successfully",
            pdf_name=file.filename,
            chunks_processed=chunks_stored,
            processing_time=processing_time,
            collection_name=collection_name
        )
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        
        # Cleanup on error
        if upload_path.exists():
            upload_path.unlink()
        
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PDF Upload"}