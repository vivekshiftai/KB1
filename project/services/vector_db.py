"""
Vector Database Service
Handles vector database operations with ChromaDB

Version: 0.1
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from datetime import datetime
from config import settings
from models.schemas import ChunkData, PDFListItem

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_SINGLETON: Optional[SentenceTransformer] = None


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    global _EMBEDDING_MODEL_SINGLETON
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
            
            logger.info(f"Storing {len(chunks)} chunks with images and tables")
            
            for i, chunk in enumerate(chunks):
                # Log image and table information for each chunk
                if chunk.images or chunk.tables:
                    logger.info(f"Chunk {i} ('{chunk.heading}'): {len(chunk.images)} images, {len(chunk.tables)} tables")
                    if chunk.images:
                        logger.info(f"  Images: {chunk.images}")
                    if chunk.tables:
                        logger.info(f"  Tables: {chunk.tables}")
                
                total_images += len(chunk.images)
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
                
                # Prepare metadata
                metadata = {
                    "heading": chunk.heading,
                    "images": json.dumps(chunk.images),
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
            logger.info(f"Total images stored: {total_images}")
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
            
            # Store each image in the collection
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
                    
                    # Store in a separate images collection
                    images_collection_name = f"{collection_name}_images"
                    images_collection = self.client.get_or_create_collection(
                        name=images_collection_name,
                        metadata={"created_at": datetime.now().isoformat()}
                    )
                    
                    # Store image with base64 data
                    images_collection.add(
                        ids=[f"img_{relative_path}"],
                        documents=[image_base64],
                        metadatas=[image_metadata]
                    )
                    
                    logger.info(f"Stored image: {relative_path} ({len(image_data)} bytes)")
                    
                except Exception as e:
                    logger.error(f"Error storing image {image_file}: {str(e)}")
                    continue
            
            logger.info(f"Successfully stored {len(image_files)} images for collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Error in image storage: {str(e)}")
            raise e
    
    async def get_image(self, collection_name: str, image_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve image data by path from the collection"""
        logger.info(f"Retrieving image {image_path} from collection {collection_name}")
        
        try:
            # Get images collection
            images_collection_name = f"{collection_name}_images"
            images_collection = self.client.get_collection(name=images_collection_name)
            
            # Try to get image directly by ID first (more efficient)
            image_id = f"img_{image_path}"
            try:
                results = images_collection.get(
                    ids=[image_id],
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
            
            # Fallback: Query for the specific image by path
            results = images_collection.query(
                query_texts=["image"],  # Dummy query to get all images
                n_results=1000,  # Get all images
                include=["documents", "metadatas"]
            )
            
            # Find the image by path
            for i, metadata in enumerate(results["metadatas"][0]):
                if metadata.get("path") == image_path:
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
        logger.info(f"Querying collection: {collection_name} with query: {query[:100]}...")
        
        try:
            # Get collection
            collection = self.client.get_collection(name=collection_name)
            
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
                # Parse images and tables from metadata
                images = json.loads(results["metadatas"][0][i].get("images", "[]"))
                tables = json.loads(results["metadatas"][0][i].get("tables", "[]"))
                
                # Log image and table information
                if images or tables:
                    heading = results["metadatas"][0][i].get("heading", f"Chunk {i}")
                    logger.info(f"Result {i} ('{heading}'): {len(images)} images, {len(tables)} tables")
                    if images:
                        logger.info(f"  Images: {images}")
                    if tables:
                        logger.info(f"  Tables: {tables}")
                
                total_images += len(images)
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
                    "images": images,
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
                
                if len(doc_content) < 100:
                    logger.warning(f"Retrieved chunk {i} has very short content: '{doc_content}'")
            
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
                if collection.name.startswith("pdf_"):
                    # Get collection info
                    coll = self.client.get_collection(collection.name)
                    count = coll.count()
                    
                    # Extract PDF name from collection name
                    pdf_name = collection.name.replace("pdf_", "").replace("_", " ")
                    
                    # Get creation date from metadata with robust fallback
                    created_at = None
                    meta = getattr(collection, "metadata", None) or {}
                    if isinstance(meta, dict):
                        created_at = meta.get("created_at")
                    if not created_at:
                        try:
                            sample = coll.get(include=["metadatas"], limit=1)
                            if sample and sample.get("metadatas"):
                                created_at = sample["metadatas"][0].get("created_at")
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
        """Check if collection exists"""
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