"""
PDF Intelligence Platform - Image Serving Endpoint
Handles serving images from vector database

Version: 0.1
"""

import base64
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from services.vector_db import VectorDatabase
from utils.helpers import sanitize_filename

router = APIRouter(prefix="/images", tags=["images"])
logger = logging.getLogger(__name__)

@router.get("/{pdf_name}/{image_path:path}")
async def get_image(pdf_name: str, image_path: str):
    """Serve image from vector database"""
    logger.info(f"Serving image {image_path} for PDF {pdf_name}")
    
    try:
        # Generate collection name
        collection_name = sanitize_filename(pdf_name)
        
        # Initialize vector database
        vector_db = VectorDatabase()
        
        # Get image from vector database
        image_data = await vector_db.get_image(collection_name, image_path)
        
        if not image_data:
            raise HTTPException(
                status_code=404,
                detail=f"Image '{image_path}' not found for PDF '{pdf_name}'"
            )
        
        # Decode base64 image data
        try:
            image_bytes = base64.b64decode(image_data["data"])
        except Exception as e:
            logger.error(f"Error decoding image data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error processing image data"
            )
        
        # Return image response
        return Response(
            content=image_bytes,
            media_type=image_data["content_type"],
            headers={
                "Content-Disposition": f"inline; filename={image_data['metadata']['filename']}",
                "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to serve image: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Image Serving"}
