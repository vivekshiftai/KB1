import logging
from fastapi import APIRouter, HTTPException, Query
from services.vector_db import VectorDatabase
from models.schemas import PDFListResponse

router = APIRouter(prefix="/pdfs", tags=["pdfs"])
logger = logging.getLogger(__name__)

@router.get("", response_model=PDFListResponse)
async def list_pdfs(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page")
):
    """List all processed PDFs with pagination"""
    logger.info(f"Listing PDFs - page: {page}, limit: {limit}")
    
    try:
        # Initialize vector database
        vector_db = VectorDatabase()
        
        # Get all PDF collections
        all_pdfs = await vector_db.list_pdf_collections()
        
        # Sort by creation date (newest first)
        all_pdfs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_pdfs = all_pdfs[start_idx:end_idx]
        
        logger.info(f"Retrieved {len(paginated_pdfs)} PDFs (total: {len(all_pdfs)})")
        
        return PDFListResponse(
            success=True,
            pdfs=paginated_pdfs,
            total_count=len(all_pdfs)
        )
        
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list PDFs: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PDFs"}