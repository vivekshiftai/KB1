"""
PDF Intelligence Platform - Debug Endpoint
Helps debug and troubleshoot issues with the system

Version: 0.1
"""

import time
import logging
from fastapi import APIRouter, HTTPException
from services.vector_db import VectorDatabase
from services.llm_service import LLMService
from utils.helpers import sanitize_filename
from models.schemas import StandardResponse

router = APIRouter(prefix="/debug", tags=["debug"])
logger = logging.getLogger(__name__)

@router.get("/collection/{pdf_name}")
async def debug_collection(pdf_name: str):
    """Debug a specific PDF collection to see its contents"""
    try:
        vector_db = VectorDatabase()
        collection_name = sanitize_filename(pdf_name)
        
        if not vector_db.collection_exists(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        # Get all chunks from the collection
        all_chunks = await vector_db.get_all_chunks(collection_name)
        
        # Also check if there's an images collection
        images_collection_name = f"{collection_name}_images"
        has_images_collection = vector_db.collection_exists(images_collection_name)
        
        logger.info(f"Collection: {collection_name}, has {len(all_chunks)} chunks")
        logger.info(f"Images collection exists: {has_images_collection}")
        
        # Analyze the chunks
        chunk_analysis = []
        total_content_length = 0
        
        for i, chunk in enumerate(all_chunks):
            doc_content = chunk.get("document", "")
            heading = chunk.get("metadata", {}).get("heading", f"Chunk {i}")
            content_length = len(doc_content)
            total_content_length += content_length
            
            chunk_info = {
                "chunk_index": i,
                "heading": heading,
                "content_length": content_length,
                "content_preview": doc_content[:200] + "..." if len(doc_content) > 200 else doc_content,
                "has_content": bool(doc_content.strip()),
                "images": chunk.get("images", []),
                "tables": chunk.get("tables", [])
            }
            chunk_analysis.append(chunk_info)
        
        return StandardResponse(
            success=True,
            message=f"Collection analysis for {collection_name}",
            data={
                "collection_name": collection_name,
                "has_images_collection": has_images_collection,
                "total_chunks": len(all_chunks),
                "total_content_length": total_content_length,
                "average_content_length": total_content_length / len(all_chunks) if all_chunks else 0,
                "chunks_with_content": sum(1 for c in chunk_analysis if c["has_content"]),
                "chunks_analysis": chunk_analysis
            }
        )
        
    except Exception as e:
        logger.error(f"Error debugging collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@router.post("/test-query/{pdf_name}")
async def test_query(pdf_name: str, query: str = "test query"):
    """Test the query pipeline step by step"""
    try:
        vector_db = VectorDatabase()
        llm_service = LLMService()
        collection_name = sanitize_filename(pdf_name)
        
        if not vector_db.collection_exists(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        # Step 1: Get all chunks first to see what's in the collection
        logger.info(f"Step 0: Getting all chunks from collection: {collection_name}")
        all_chunks = await vector_db.get_all_chunks(collection_name)
        logger.info(f"Collection has {len(all_chunks)} total chunks")
        
        # Step 1: Query vector database
        logger.info(f"Step 1: Querying vector database with query: '{query}'")
        chunks = await vector_db.query_chunks(collection_name, query, top_k=5)
        
        # Step 2: Analyze retrieved chunks
        chunks_analysis = []
        for i, chunk in enumerate(chunks):
            doc_content = chunk.get("document", "")
            heading = chunk.get("metadata", {}).get("heading", f"Chunk {i}")
            distance = chunk.get("distance", 0)
            
            chunk_info = {
                "chunk_index": i,
                "heading": heading,
                "content_length": len(doc_content),
                "content_preview": doc_content[:200] + "..." if len(doc_content) > 200 else doc_content,
                "distance": distance,
                "has_content": bool(doc_content.strip())
            }
            chunks_analysis.append(chunk_info)
        
        # Step 3: Test LLM processing
        logger.info(f"Step 3: Testing LLM processing with {len(chunks)} chunks")
        llm_result = None
        llm_error = None
        
        try:
            llm_result = await llm_service.query_with_context(chunks, query)
        except Exception as e:
            llm_error = str(e)
            logger.error(f"LLM processing failed: {llm_error}")
        
        return StandardResponse(
            success=True,
            message="Query pipeline test completed",
            data={
                "query": query,
                "collection_name": collection_name,
                "total_chunks_in_collection": len(all_chunks),
                "chunks_retrieved": len(chunks),
                "chunks_analysis": chunks_analysis,
                "llm_result": llm_result,
                "llm_error": llm_error
            }
        )
        
    except Exception as e:
        logger.error(f"Error testing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@router.get("/collections")
async def list_collections():
    """List all available collections"""
    try:
        vector_db = VectorDatabase()
        collections = await vector_db.list_pdf_collections()
        
        return StandardResponse(
            success=True,
            message="Collections listed successfully",
            data={
                "total_collections": len(collections),
                "collections": [
                    {
                        "collection_name": coll.collection_name,
                        "pdf_name": coll.pdf_name,
                        "chunk_count": coll.chunk_count,
                        "created_at": coll.created_at.isoformat()
                    }
                    for coll in collections
                ]
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@router.get("/test-images/{pdf_name}")
async def test_images(pdf_name: str):
    """Test image storage and retrieval for a specific PDF"""
    try:
        vector_db = VectorDatabase()
        collection_name = sanitize_filename(pdf_name)
        
        # Check document collection
        doc_collection_info = None
        if vector_db.collection_exists(collection_name):
            doc_collection = vector_db.client.get_collection(name=collection_name)
            doc_count = doc_collection.count()
            doc_sample = doc_collection.get(include=["metadatas"], limit=3)
            
            # Extract image paths from document chunks
            image_paths = []
            for metadata in doc_sample["metadatas"]:
                if "images" in metadata:
                    try:
                        import json
                        chunk_images = json.loads(metadata["images"])
                        image_paths.extend(chunk_images)
                    except:
                        pass
            
            doc_collection_info = {
                "collection_name": collection_name,
                "count": doc_count,
                "sample_metadata": doc_sample["metadatas"][:3] if doc_sample["metadatas"] else [],
                "image_paths_found": list(set(image_paths))
            }
        
        # Check images collection
        images_collection_name = f"{collection_name}_images"
        images_collection_info = None
        if vector_db.collection_exists(images_collection_name):
            images_collection = vector_db.client.get_collection(name=images_collection_name)
            images_count = images_collection.count()
            images_sample = images_collection.get(include=["metadatas"], limit=5)
            
            images_collection_info = {
                "collection_name": images_collection_name,
                "count": images_count,
                "sample_metadata": images_sample["metadatas"][:5] if images_sample["metadatas"] else [],
                "all_image_paths": [meta.get("path", "") for meta in images_sample["metadatas"]] if images_sample["metadatas"] else []
            }
        
        # Test image retrieval for a sample image path
        test_image_retrieval = None
        if doc_collection_info and doc_collection_info["image_paths_found"]:
            test_path = doc_collection_info["image_paths_found"][0]
            try:
                image_data = await vector_db.get_image(collection_name, test_path)
                test_image_retrieval = {
                    "test_path": test_path,
                    "retrieved": image_data is not None,
                    "metadata": image_data["metadata"] if image_data else None
                }
            except Exception as e:
                test_image_retrieval = {
                    "test_path": test_path,
                    "retrieved": False,
                    "error": str(e)
                }
        
        return {
            "document_collection": doc_collection_info,
            "images_collection": images_collection_info,
            "test_image_retrieval": test_image_retrieval,
            "all_collections": [coll.name for coll in vector_db.client.list_collections()]
        }
        
    except Exception as e:
        return {"error": str(e)}

@router.get("/health")
async def debug_health():
    """Comprehensive health check"""
    try:
        vector_db = VectorDatabase()
        llm_service = LLMService()
        
        health_status = {
            "vector_database": "unknown",
            "llm_service": "unknown",
            "collections": "unknown"
        }
        
        # Test vector database
        try:
            collections = await vector_db.list_pdf_collections()
            health_status["vector_database"] = "healthy"
            health_status["collections"] = f"{len(collections)} collections found"
        except Exception as e:
            health_status["vector_database"] = f"error: {str(e)}"
        
        # Test LLM service
        try:
            # Test with a simple query
            test_chunks = [{"document": "test content", "metadata": {"heading": "test"}}]
            result = await llm_service.query_with_context(test_chunks, "test query")
            health_status["llm_service"] = "healthy"
        except Exception as e:
            health_status["llm_service"] = f"error: {str(e)}"
        
        return StandardResponse(
            success=True,
            message="Health check completed",
            data=health_status
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/test-collection/{pdf_name}")
async def test_collection_directly(pdf_name: str):
    """Test a collection directly without querying"""
    try:
        vector_db = VectorDatabase()
        collection_name = sanitize_filename(pdf_name)
        
        if not vector_db.collection_exists(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        # Get collection type
        collection_type = vector_db.get_collection_type(collection_name)
        
        # Get the collection directly
        collection = vector_db.client.get_collection(name=collection_name)
        
        # Get all documents without querying
        results = collection.get(
            include=["documents", "metadatas"],
            limit=10
        )
        
        # Analyze the results
        analysis = []
        for i in range(len(results["documents"])):
            doc_content = results["documents"][i]
            metadata = results["metadatas"][i]
            
            analysis.append({
                "index": i,
                "content_length": len(doc_content),
                "content_preview": doc_content[:200] + "..." if len(doc_content) > 200 else doc_content,
                "metadata_keys": list(metadata.keys()),
                "metadata": metadata
            })
        
        return StandardResponse(
            success=True,
            message=f"Direct collection test for {collection_name}",
            data={
                "collection_name": collection_name,
                "collection_type": collection_type,
                "total_documents": len(results["documents"]),
                "analysis": analysis
            }
        )
        
    except Exception as e:
        logger.error(f"Error testing collection directly: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@router.get("/fix-collection/{pdf_name}")
async def fix_collection_issue(pdf_name: str):
    """Attempt to fix collection issues by identifying the correct collection"""
    try:
        vector_db = VectorDatabase()
        base_collection_name = sanitize_filename(pdf_name)
        
        # Check different possible collection names
        possible_collections = [
            base_collection_name,
            f"pdf_{base_collection_name}",
            f"{base_collection_name}_documents",
            f"{base_collection_name}_chunks"
        ]
        
        collection_info = {}
        
        for coll_name in possible_collections:
            if vector_db.collection_exists(coll_name):
                coll_type = vector_db.get_collection_type(coll_name)
                collection_info[coll_name] = {
                    "exists": True,
                    "type": coll_type
                }
            else:
                collection_info[coll_name] = {
                    "exists": False,
                    "type": "none"
                }
        
        # Also check for images collection
        images_collection_name = f"{base_collection_name}_images"
        if vector_db.collection_exists(images_collection_name):
            collection_info[images_collection_name] = {
                "exists": True,
                "type": "image"
            }
        
        return StandardResponse(
            success=True,
            message=f"Collection analysis for {pdf_name}",
            data={
                "base_name": base_collection_name,
                "collections": collection_info,
                "recommendation": "Use the collection with type 'document' for queries"
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
