"""
Vector Database Service
Handles vector database operations with ChromaDB

Version: 0.1
"""

import os
import json
import logging
import threading
import asyncio
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from datetime import datetime
from config import settings
from models.schemas import ChunkData, PDFListItem, ImageData
from utils.helpers import sanitize_filename

logger = logging.getLogger(__name__)

# Thread-safe singleton pattern for embedding model
_EMBEDDING_MODEL_SINGLETON: Optional[SentenceTransformer] = None
_EMBEDDING_MODEL_LOCK = threading.Lock()


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    global _EMBEDDING_MODEL_SINGLETON
    if _EMBEDDING_MODEL_SINGLETON is None:
        with _EMBEDDING_MODEL_LOCK:
            # Double-check locking pattern
            if _EMBEDDING_MODEL_SINGLETON is None:
                # Force CPU usage for sentence transformers
                _EMBEDDING_MODEL_SINGLETON = SentenceTransformer(model_name, device='cpu')
                logger.info(f"Initialized SentenceTransformer model '{model_name}' on CPU")
    return _EMBEDDING_MODEL_SINGLETON


class VectorDatabase:
    def __init__(self):
        self.db_type = settings.vector_db_type
        self.embedding_model = _get_embedding_model(settings.embedding_model)
        self.client = self._initialize_client()
        # Thread-safe locks for concurrent operations
        self._collection_locks = {}  # Per-collection locks
        self._global_lock = threading.Lock()  # For collection management
        
        logger.info("Vector Database thread safety initialized:")
        logger.info(f"  - Per-collection locks: ENABLED")
        logger.info(f"  - Global lock for collection management: ENABLED")
        logger.info(f"  - Thread-safe embedding model singleton: ENABLED")
        logger.info(f"  - Thread-safe ChromaDB client: ENABLED")
    
    def _get_collection_lock(self, collection_name: str) -> threading.Lock:
        """Get or create a lock for a specific collection"""
        with self._global_lock:
            if collection_name not in self._collection_locks:
                self._collection_locks[collection_name] = threading.Lock()
                logger.info(f"Created new collection lock for: {collection_name} (Thread: {threading.current_thread().name})")
            return self._collection_locks[collection_name]
    
    def _initialize_client(self):
        """Initialize vector database client"""
        if self.db_type == "azure" and settings.azure_openai_key:
            logger.info("Initializing Azure Vector Search client")
            return self._setup_azure_client()
        else:
            logger.info("Initializing ChromaDB client")
            return self._setup_chromadb_client()
    
    def _setup_chromadb_client(self):
        """Setup ChromaDB client"""
        try:
            client = chromadb.PersistentClient(
                path=settings.chromadb_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB client initialized at: {settings.chromadb_path}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise e
    
    def _setup_azure_client(self):
        """Setup Azure Vector Search client (placeholder)"""
        # Implementation for Azure Vector Search would go here
        logger.warning("Azure Vector Search not implemented, falling back to ChromaDB")
        return self._setup_chromadb_client()
    
    async def store_chunks(self, chunks: List[ChunkData], collection_name: str, output_dir: str = None) -> int:
        """Store chunks in vector database"""
        logger.info(f"Storing {len(chunks)} chunks in collection: {collection_name}")
        
        try:
            # Get or create collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"created_at": datetime.now().isoformat()}
            )
            
            # Store images if output_dir is provided
            if output_dir:
                await self._store_images(collection_name, output_dir)
            
            # Prepare data for batch insertion
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            # Debug: Log chunk information
            total_images = 0
            total_tables = 0
            total_embedded_images = 0
            
            logger.info(f"Storing {len(chunks)} chunks with embedded images and tables")
            
            for i, chunk in enumerate(chunks):
                # Log image and table information for each chunk
                if chunk.image_paths or chunk.embedded_images or chunk.tables:
                    logger.info(f"Chunk {i} ('{chunk.heading}'): {len(chunk.image_paths)} image paths, {len(chunk.embedded_images)} embedded images, {len(chunk.tables)} tables")
                    if chunk.image_paths:
                        logger.info(f"  Image paths: {chunk.image_paths}")
                    if chunk.embedded_images:
                        logger.info(f"  Embedded images: {[img.filename for img in chunk.embedded_images]}")
                        total_embedded_images += len(chunk.embedded_images)
                    if chunk.tables:
                        logger.info(f"  Tables: {chunk.tables}")
                
                total_images += len(chunk.image_paths)
                total_embedded_images += len(chunk.embedded_images)
                total_tables += len(chunk.tables)
                # Create combined text for embedding
                combined_text = f"{chunk.heading}\n{chunk.text}"
                
                # Generate embedding
                embedding = self.embedding_model.encode(combined_text).tolist()
                
                # Prepare data
                ids.append(f"chunk-{i}")
                embeddings.append(embedding)
                documents.append(chunk.text)
                
                # Debug: Log document content
                logger.info(f"Storing chunk {i}: heading='{chunk.heading}', text_length={len(chunk.text)}")
                if len(chunk.text) < 100:
                    logger.warning(f"Chunk {i} has very short text: '{chunk.text}'")
                
                # Convert embedded images to serializable format
                embedded_images_data = []
                for img in chunk.embedded_images:
                    embedded_images_data.append({
                        "filename": img.filename,
                        "data": img.data,  # Base64 data
                        "mime_type": img.mime_type,
                        "size": img.size
                    })
                
                # Prepare metadata
                metadata = {
                    "heading": chunk.heading,
                    "image_paths": json.dumps(chunk.image_paths),
                    "embedded_images": json.dumps(embedded_images_data),
                    "tables": json.dumps(chunk.tables),
                    "chunk_index": i,
                    "created_at": datetime.now().isoformat()
                }
                metadatas.append(metadata)
            
            # Batch insert
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Successfully stored {len(chunks)} chunks")
            logger.info(f"Total image paths stored: {total_images}")
            logger.info(f"Total embedded images stored: {total_embedded_images}")
            logger.info(f"Total tables stored: {total_tables}")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            raise e
    
    async def _store_images(self, collection_name: str, output_dir: str):
        """Store images from MinerU output in the collection"""
        logger.info(f"Storing images from {output_dir} for collection {collection_name}")
        
        try:
            import base64
            from pathlib import Path
            
            # Find all image files in the output directory
            image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(output_dir).glob(f"**/*{ext}"))
                image_files.extend(Path(output_dir).glob(f"**/*{ext.upper()}"))
            
            logger.info(f"Found {len(image_files)} image files in {output_dir}")
            
            if not image_files:
                logger.warning(f"No image files found in {output_dir}")
                return
            
            # Create images collection using only PDF name
            images_collection_name = f"{collection_name}_images"
            images_collection = self.client.get_or_create_collection(
                name=images_collection_name,
                metadata={"created_at": datetime.now().isoformat()}
            )
            
            logger.info(f"Created images collection: {images_collection_name}")
            
            # Store each image in the collection
            stored_count = 0
            for image_file in image_files:
                try:
                    # Read image file and encode as base64
                    with open(image_file, "rb") as f:
                        image_data = f.read()
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                    
                    # Get relative path from output_dir for consistent naming
                    relative_path = str(image_file.relative_to(Path(output_dir)))
                    
                    # Store image metadata
                    image_metadata = {
                        "type": "image",
                        "filename": image_file.name,
                        "path": relative_path,
                        "size": len(image_data),
                        "format": image_file.suffix.lower(),
                        "collection": collection_name,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Store image with base64 data using filename as ID for easy retrieval
                    image_id = image_file.name
                    images_collection.add(
                        ids=[image_id],
                        documents=[image_base64],
                        metadatas=[image_metadata]
                    )
                    
                    logger.info(f"Stored image: {image_file.name} (path: {relative_path}, {len(image_data)} bytes)")
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing image {image_file}: {str(e)}")
                    continue
            
            logger.info(f"Successfully stored {stored_count} images in collection {images_collection_name}")
            
            # Verify storage by checking collection
            try:
                verify_collection = self.client.get_collection(name=images_collection_name)
                actual_count = verify_collection.count()
                logger.info(f"Verified: Images collection {images_collection_name} contains {actual_count} images")
                
                if actual_count != stored_count:
                    logger.warning(f"Storage count mismatch: stored {stored_count}, but collection has {actual_count}")
            except Exception as e:
                logger.warning(f"Could not verify image storage: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in image storage: {str(e)}")
            raise e
    
    async def get_image(self, collection_name: str, image_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve image data by path from the collection"""
        logger.info(f"Retrieving image {image_path} from collection {collection_name}")
        
        try:
            # Try multiple possible collection names for image retrieval
            possible_collection_names = [
                f"{collection_name}_images",  # e.g., "test_images"
                collection_name,  # e.g., "test" (in case images are stored in main collection)
                f"{collection_name.replace('_images', '')}_images",  # Handle double _images suffix
                f"{collection_name.replace('_images', '')}"  # Remove _images if present
            ]
            
            images_collection = None
            successful_collection_name = None
            
            for coll_name in possible_collection_names:
                try:
                    logger.info(f"Trying to get image from collection: {coll_name}")
                    images_collection = self.client.get_collection(name=coll_name)
                    successful_collection_name = coll_name
                    logger.info(f"Successfully accessed collection: {coll_name}")
                    break
                except Exception as e:
                    logger.debug(f"Collection {coll_name} not found: {str(e)}")
                    continue
            
            if images_collection is None:
                logger.error(f"Could not find any valid collection for image retrieval")
                logger.error(f"Tried collections: {possible_collection_names}")
                return None
            
            # Debug: Log what's in the images collection
            logger.info(f"Searching for image in collection: {successful_collection_name}")
            logger.info(f"Looking for image path: {image_path}")
            logger.info(f"Image filename to search for: {image_filename}")
            
            # Try to get image directly by filename first (more efficient)
            # Extract filename from path (e.g., "images/abc123.jpg" -> "abc123.jpg")
            image_filename = image_path.split('/')[-1] if '/' in image_path else image_path
            
            try:
                results = images_collection.get(
                    ids=[image_filename],
                    include=["documents", "metadatas"]
                )
                
                if results["ids"] and len(results["ids"]) > 0:
                    image_data = results["documents"][0]
                    metadata = results["metadatas"][0]
                    
                    return {
                        "data": image_data,  # Base64 encoded image
                        "metadata": metadata,
                        "content_type": self._get_content_type(metadata.get("format", ""))
                    }
            except Exception as e:
                logger.debug(f"Direct lookup failed for {image_id}, trying query method: {str(e)}")
            
            # Debug: Get all IDs in the collection to see what's available
            try:
                all_ids = images_collection.get(include=["metadatas"], limit=1000)
                logger.info(f"Available image filenames in collection: {all_ids['ids'][:10]}...")  # Show first 10 filenames
                logger.info(f"Total images in collection: {len(all_ids['ids'])}")
                
                # Show sample metadata to verify storage
                if all_ids["metadatas"]:
                    sample_metadata = all_ids["metadatas"][:3]
                    logger.info(f"Sample image metadata: {sample_metadata}")
            except Exception as e:
                logger.warning(f"Could not get all IDs from collection: {str(e)}")
            
            # Fallback: Query for the specific image by path
            results = images_collection.query(
                query_texts=["image"],  # Dummy query to get all images
                n_results=1000,  # Get all images
                include=["documents", "metadatas"]
            )
            
            # Find the image by filename (fallback)
            for i, metadata in enumerate(results["metadatas"][0]):
                stored_filename = metadata.get("filename", "")
                stored_path = metadata.get("path", "")
                
                # Try to match by filename first, then by path
                if stored_filename == image_filename or stored_path == image_path:
                    image_data = results["documents"][0][i]
                    return {
                        "data": image_data,  # Base64 encoded image
                        "metadata": metadata,
                        "content_type": self._get_content_type(metadata.get("format", ""))
                    }
            
            logger.warning(f"Image {image_path} not found in collection {collection_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving image {image_path}: {str(e)}")
            return None
    
    def _get_content_type(self, file_extension: str) -> str:
        """Get MIME content type from file extension"""
        content_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff'
        }
        return content_types.get(file_extension.lower(), 'application/octet-stream')
    
    async def query_chunks(self, collection_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query chunks from vector database"""
        logger.info(f"Querying collection: {collection_name} with query: {query[:100]}... (Thread: {threading.current_thread().name})")
        
        try:
            # Get collection with thread safety logging
            logger.info(f"Acquiring collection: {collection_name}")
            collection = self.client.get_collection(name=collection_name)
            logger.info(f"Collection acquired successfully: {collection_name}")
            
            # Check if this is an images collection (ends with _images)
            if collection_name.endswith("_images"):
                logger.warning(f"Attempting to query images collection: {collection_name}")
                logger.warning("This should not happen - images collections should not be queried for text content")
                return []
            
            # Validate that this is a document collection by checking metadata structure
            # Get a sample to check metadata structure
            sample_results = collection.get(include=["metadatas"], limit=1)
            if sample_results["metadatas"]:
                sample_metadata = sample_results["metadatas"][0]
                if "type" in sample_metadata and sample_metadata["type"] == "image":
                    logger.error(f"Collection {collection_name} contains image data, not document chunks")
                    logger.error(f"Sample metadata: {sample_metadata}")
                    return []
                if "heading" not in sample_metadata:
                    logger.warning(f"Collection {collection_name} metadata missing 'heading' field")
                    logger.warning(f"Sample metadata keys: {list(sample_metadata.keys())}")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            total_images = 0
            total_tables = 0
            
            for i in range(len(results["documents"][0])):
                # Parse image paths, embedded images, and tables from metadata
                image_paths = json.loads(results["metadatas"][0][i].get("image_paths", "[]"))
                embedded_images_data = json.loads(results["metadatas"][0][i].get("embedded_images", "[]"))
                tables = json.loads(results["metadatas"][0][i].get("tables", "[]"))
                
                # Convert embedded images back to ImageData objects
                embedded_images = []
                for img_data in embedded_images_data:
                    try:
                        embedded_images.append(ImageData(
                            filename=img_data["filename"],
                            data=img_data["data"],  # Base64 data
                            mime_type=img_data["mime_type"],
                            size=img_data["size"]
                        ))
                    except Exception as e:
                        logger.warning(f"Error parsing embedded image data: {str(e)}")
                
                # Log image and table information
                if image_paths or embedded_images or tables:
                    heading = results["metadatas"][0][i].get("heading", f"Chunk {i}")
                    logger.info(f"Result {i} ('{heading}'): {len(image_paths)} image paths, {len(embedded_images)} embedded images, {len(tables)} tables")
                    if image_paths:
                        logger.info(f"  Image paths: {image_paths}")
                    if embedded_images:
                        logger.info(f"  Embedded images: {[img.filename for img in embedded_images]}")
                    if tables:
                        logger.info(f"  Tables: {tables}")
                
                total_images += len(embedded_images)
                total_tables += len(tables)
                
                # Skip image chunks
                metadata = results["metadatas"][0][i]
                if "type" in metadata and metadata["type"] == "image":
                    logger.warning(f"Skipping image chunk: {metadata.get('filename', 'unknown')}")
                    continue
                
                result = {
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "image_paths": image_paths,
                    "embedded_images": embedded_images,
                    "tables": tables
                }
                formatted_results.append(result)
                
                # Debug: Log retrieved document content
                doc_content = results["documents"][0][i]
                heading = results["metadatas"][0][i].get("heading", f"Chunk {i}")
                logger.info(f"Retrieved chunk {i}: heading='{heading}', content_length={len(doc_content)}")
                
                # Validate that this is a document chunk, not an image chunk
                metadata = results["metadatas"][0][i]
                if "type" in metadata and metadata["type"] == "image":
                    logger.warning(f"Retrieved image chunk instead of document chunk: {metadata}")
                    continue
                
                # Include all chunks - let the LLM decide relevance based on query
            
            logger.info(f"Total images in results: {total_images}")
            logger.info(f"Total tables in results: {total_tables}")
            
            logger.info(f"Retrieved {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying chunks: {str(e)}")
            raise e
    
    async def list_pdf_collections(self) -> List[PDFListItem]:
        """List all PDF collections"""
        logger.info("Listing all PDF collections")
        
        try:
            collections = self.client.list_collections()
            pdf_collections = []
            
            for collection in collections:
                # Skip image collections (they end with _images)
                if collection.name.endswith("_images"):
                    continue
                
                # Check if this is a document collection by examining metadata
                try:
                    coll = self.client.get_collection(collection.name)
                    sample_results = coll.get(include=["metadatas"], limit=1)
                    
                    if sample_results["metadatas"]:
                        sample_metadata = sample_results["metadatas"][0]
                        # Skip if this is an image collection (has type: "image")
                        if "type" in sample_metadata and sample_metadata["type"] == "image":
                            continue
                        # Include if this has document structure (has heading field)
                        if "heading" in sample_metadata:
                            count = coll.count()
                            
                            # Use collection name as PDF name (this matches the upload pattern)
                            pdf_name = collection.name.replace("_", " ")
                            
                            # Get creation date from metadata with robust fallback
                            created_at = None
                            meta = getattr(collection, "metadata", None) or {}
                            if isinstance(meta, dict):
                                created_at = meta.get("created_at")
                            if not created_at:
                                try:
                                    created_at = sample_metadata.get("created_at")
                                except Exception:
                                    created_at = None
                            if created_at:
                                created_at = datetime.fromisoformat(created_at)
                            else:
                                created_at = datetime.now()
                            
                            pdf_collections.append(PDFListItem(
                                collection_name=collection.name,
                                pdf_name=pdf_name,
                                created_at=created_at,
                                chunk_count=count
                            ))
                            logger.info(f"Found PDF collection: {collection.name} -> {pdf_name}")
                except Exception as e:
                    logger.warning(f"Error examining collection {collection.name}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(pdf_collections)} PDF collections")
            return pdf_collections
            
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            raise e
    
    async def get_all_chunks(self, collection_name: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get all chunks from a collection"""
        logger.info(f"Getting all chunks from collection: {collection_name}")
        
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Get all documents
            results = collection.get(
                include=["documents", "metadatas"],
                limit=limit
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"])):
                result = {
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "images": json.loads(results["metadatas"][i].get("images", "[]")),
                    "tables": json.loads(results["metadatas"][i].get("tables", "[]"))
                }
                formatted_results.append(result)
            
            logger.info(f"Retrieved {len(formatted_results)} chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting all chunks: {str(e)}")
            raise e
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists (thread-safe)"""
        collection_lock = self._get_collection_lock(collection_name)
        with collection_lock:
            try:
                self.client.get_collection(collection_name)
                return True
            except Exception:
                return False
    
    def get_collection_type(self, collection_name: str) -> str:
        """Get the type of collection (document or image)"""
        try:
            collection = self.client.get_collection(collection_name)
            sample_results = collection.get(include=["metadatas"], limit=1)
            if sample_results["metadatas"]:
                sample_metadata = sample_results["metadatas"][0]
                if "type" in sample_metadata and sample_metadata["type"] == "image":
                    return "image"
                elif "heading" in sample_metadata:
                    return "document"
                else:
                    return "unknown"
            return "unknown"
        except Exception as e:
            logger.error(f"Error getting collection type for {collection_name}: {str(e)}")
            return "unknown"
    
    async def delete_pdf_collections(self, pdf_name: str) -> bool:
        """Delete both document and image collections for a PDF"""
        logger.info(f"Deleting collections for PDF: {pdf_name}")
        
        try:
            # Generate collection name using the same pattern as upload
            collection_name = sanitize_filename(pdf_name)
            images_collection_name = f"{collection_name}_images"
            
            deleted_collections = []
            
            # Delete document collection
            if self.collection_exists(collection_name):
                try:
                    self.client.delete_collection(name=collection_name)
                    deleted_collections.append(collection_name)
                    logger.info(f"Deleted document collection: {collection_name}")
                except Exception as e:
                    logger.error(f"Error deleting document collection {collection_name}: {str(e)}")
                    raise e
            else:
                logger.warning(f"Document collection {collection_name} does not exist")
            
            # Delete images collection
            if self.collection_exists(images_collection_name):
                try:
                    self.client.delete_collection(name=images_collection_name)
                    deleted_collections.append(images_collection_name)
                    logger.info(f"Deleted images collection: {images_collection_name}")
                except Exception as e:
                    logger.error(f"Error deleting images collection {images_collection_name}: {str(e)}")
                    raise e
            else:
                logger.warning(f"Images collection {images_collection_name} does not exist")
            
            logger.info(f"Successfully deleted {len(deleted_collections)} collections: {deleted_collections}")
            return len(deleted_collections) > 0
            
        except Exception as e:
            logger.error(f"Error deleting PDF collections: {str(e)}")
            raise e