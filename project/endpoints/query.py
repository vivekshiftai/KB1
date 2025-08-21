"""
PDF Intelligence Platform - Query Endpoint
Handles intelligent querying of PDF content

Version: 0.1
"""

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
        
        # Generate collection name - use exact same pattern as upload
        collection_name = sanitize_filename(request.pdf_name)
        logger.info(f"Looking for collection: '{collection_name}'")
        
        # Store the original collection name for image retrieval BEFORE any fallback logic
        original_collection_name = collection_name
        
        # Check if collection exists
        if not vector_db.collection_exists(collection_name):
            raise HTTPException(
                status_code=404, 
                detail=f"PDF '{request.pdf_name}' not found. Please upload the PDF first."
            )
        
        # Check collection type to ensure we're querying a document collection
        collection_type = vector_db.get_collection_type(collection_name)
        logger.info(f"Collection '{collection_name}' type: {collection_type}")
        
        # If the first collection is an image collection, this indicates a naming issue
        if collection_type == "image":
            logger.error(f"Collection '{collection_name}' is an image collection, not a document collection")
            logger.error(f"This suggests the PDF '{request.pdf_name}' was not uploaded correctly or the collection name is wrong")
            
            # Try to find the correct document collection
            logger.info(f"Attempting to find the correct document collection...")
            
            # Check if there's a document collection without the _images suffix
            base_collection_name = collection_name.replace("_images", "")
            if vector_db.collection_exists(base_collection_name):
                base_type = vector_db.get_collection_type(base_collection_name)
                logger.info(f"Found base collection '{base_collection_name}' with type: {base_type}")
                if base_type == "document":
                    collection_name = base_collection_name
                    collection_type = base_type
                    logger.info(f"Using base collection: '{collection_name}'")
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Collection '{collection_name}' is an image collection and base collection '{base_collection_name}' is not a document collection."
                    )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Collection '{collection_name}' is an image collection. The PDF '{request.pdf_name}' may not have been uploaded correctly."
                )
        
        # Also check if the images collection exists for this PDF
        images_collection_name = f"{original_collection_name}_images"
        has_images_collection = vector_db.collection_exists(images_collection_name)
        logger.info(f"Original collection name: '{original_collection_name}'")
        logger.info(f"Images collection '{images_collection_name}' exists: {has_images_collection}")
        logger.info(f"Document collection name: '{collection_name}'")
        
        if collection_type == "unknown":
            logger.warning(f"Collection '{collection_name}' type is unknown, proceeding with caution")
        
        # Query vector database
        logger.info("Querying vector database...")
        chunks = await vector_db.query_chunks(
            collection_name=collection_name,
            query=request.query,
            top_k=request.top_k
        )
        
        # Debug: Log chunk structure
        if chunks:
            logger.info(f"Retrieved {len(chunks)} chunks")
            logger.info(f"First chunk structure: {list(chunks[0].keys())}")
            if 'metadata' in chunks[0]:
                logger.info(f"First chunk metadata: {list(chunks[0]['metadata'].keys())}")
            if 'document' in chunks[0]:
                logger.info(f"First chunk document preview: {chunks[0]['document'][:200]}...")
                logger.info(f"First chunk document length: {len(chunks[0]['document'])} characters")
            logger.info(f"First chunk heading: {chunks[0].get('metadata', {}).get('heading', 'No heading')}")
        else:
            logger.warning("No chunks retrieved from vector database")
        
        # If no chunks found, try to get some general content from the PDF
        if not chunks:
            logger.warning("No relevant chunks found, trying to get general content...")
            try:
                # Try to get some general chunks without specific query
                chunks = await vector_db.query_chunks(
                    collection_name=collection_name,
                    query="",  # Empty query to get general content
                    top_k=3
                )
                if chunks:
                    logger.info(f"Retrieved {len(chunks)} general chunks as fallback")
                else:
                    logger.error("No content available in the PDF")
                    raise HTTPException(
                        status_code=404,
                        detail="No content found in the PDF. Please ensure the PDF was uploaded and processed correctly."
                    )
            except Exception as e:
                logger.error(f"Error getting fallback chunks: {str(e)}")
                raise HTTPException(
                    status_code=404,
                    detail="No relevant content found for the query and no fallback content available."
                )
        
        # Generate response using LLM
        logger.info("Generating response with LLM...")
        try:
            llm_result = await llm_service.query_with_context(chunks, request.query)
            logger.info(f"LLM result keys: {list(llm_result.keys()) if isinstance(llm_result, dict) else 'Not a dict'}")
        except Exception as e:
            logger.error(f"LLM service error: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response: {str(e)}"
            )
        
        # Step 1: Find chunks that were actually used by the LLM
        logger.info(f"LLM used chunks: {llm_result['chunks_used']}")
        logger.info(f"Total chunks available: {len(chunks)}")
        
        # Find chunks that were actually used by the LLM
        used_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                # Handle both possible chunk structures
                if "metadata" in chunk and "document" in chunk:
                    # Vector DB format
                    heading = chunk.get("metadata", {}).get("heading", "")
                else:
                    # Fallback format
                    heading = chunk.get("heading", "")
                
                # Check if this chunk's heading is in the LLM's referenced sections
                # Use case-insensitive matching for better reliability
                chunk_used = False
                for referenced_heading in llm_result['chunks_used']:
                    # Check for exact heading match
                    if heading.lower() == referenced_heading.lower():
                        chunk_used = True
                        break
                    # Check for partial heading match
                    elif heading.lower() in referenced_heading.lower() or referenced_heading.lower() in heading.lower():
                        chunk_used = True
                        break
                    # Check for chunk index match (e.g., "Chunk 1", "Chunk 2")
                    elif referenced_heading.lower() == f"chunk {i+1}":
                        chunk_used = True
                        break
                
                if chunk_used:
                    used_chunks.append(chunk)
                    logger.info(f"Found used chunk: '{heading}' with {len(chunk.get('images', []))} images, {len(chunk.get('tables', []))} tables")
            except Exception as e:
                logger.warning(f"Error processing chunk for image collection: {str(e)}")
                continue
        
        logger.info(f"Found {len(used_chunks)} chunks that were actually used by the LLM")
        
        # If no chunks were matched by LLM, use all available chunks (fallback)
        if not used_chunks and chunks:
            logger.warning("No chunks matched by LLM, using all available chunks as fallback")
            used_chunks = chunks
            # Update the LLM result to reflect that all chunks were used
            llm_result['chunks_used'] = [f"Chunk {i+1}" for i in range(len(chunks))]
        
        # Step 2: Collect image paths from used chunks
        all_image_paths = []
        all_tables = []
        
        for i, chunk in enumerate(used_chunks):
            chunk_images = chunk.get("images", [])
            chunk_tables = chunk.get("tables", [])
            
            logger.info(f"Used chunk {i}: {len(chunk_images)} images, {len(chunk_tables)} tables")
            if chunk_images:
                logger.info(f"Used chunk {i} images: {chunk_images}")
            if chunk_tables:
                logger.info(f"Used chunk {i} tables: {chunk_tables}")
            
            all_image_paths.extend(chunk_images)
            all_tables.extend(chunk_tables)
        
        # Remove duplicates
        all_image_paths = list(set(all_image_paths))
        all_tables = list(set(all_tables))
        
        logger.info(f"Total unique image paths from used chunks: {len(all_image_paths)}")
        logger.info(f"Image paths: {all_image_paths}")
        
        # Step 3: Fetch actual image data from images collection using image paths
        from models.schemas import ImageData
        image_data_list = []
        
        if all_image_paths:
            logger.info(f"Fetching {len(all_image_paths)} images from images collection...")
            
            for image_path in all_image_paths:
                try:
                    # Get image data from images collection using the original collection name
                    image_data = await vector_db.get_image(original_collection_name, image_path)
                    if image_data:
                        # Create ImageData object
                        image_obj = ImageData(
                            filename=image_data["metadata"]["filename"],
                            data=image_data["data"],  # Base64 encoded image
                            mime_type=image_data["content_type"],
                            size=image_data["metadata"]["size"]
                        )
                        image_data_list.append(image_obj)
                        logger.info(f"Fetched image: {image_path} ({image_data['metadata']['size']} bytes)")
                    else:
                        logger.warning(f"Image not found in images collection: {image_path}")
                except Exception as e:
                    logger.error(f"Error fetching image {image_path}: {str(e)}")
                    continue
        else:
            logger.info("No image paths found in used chunks")
        
        logger.info(f"Total unique image paths from used chunks: {len(all_image_paths)}")
        logger.info(f"Successfully fetched {len(image_data_list)} images from images collection")
        logger.info(f"Total unique tables from used chunks: {len(all_tables)}")
        if all_tables:
            logger.info(f"Tables from used chunks: {all_tables}")
        
        processing_time = calculate_processing_time(start_time)
        
        logger.info(f"Query processed successfully in {processing_time}")
        
        return QueryResponse(
            success=True,
            message="Query processed successfully",
            response=llm_result["response"],
            chunks_used=llm_result["chunks_used"],
            images=image_data_list,
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