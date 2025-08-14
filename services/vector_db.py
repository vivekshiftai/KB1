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
        _EMBEDDING_MODEL_SINGLETON = SentenceTransformer(model_name)
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
    
    async def store_chunks(self, chunks: List[ChunkData], collection_name: str) -> int:
        """Store chunks in vector database"""
        logger.info(f"Storing {len(chunks)} chunks in collection: {collection_name}")
        
        try:
            # Get or create collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"created_at": datetime.now().isoformat()}
            )
            
            # Prepare data for batch insertion
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for i, chunk in enumerate(chunks):
                # Create combined text for embedding
                combined_text = f"{chunk.heading}\n{chunk.text}"
                
                # Generate embedding
                embedding = self.embedding_model.encode(combined_text).tolist()
                
                # Prepare data
                ids.append(f"chunk-{i}")
                embeddings.append(embedding)
                documents.append(chunk.text)
                
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
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            raise e
    
    async def query_chunks(self, collection_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query chunks from vector database"""
        logger.info(f"Querying collection: {collection_name} with query: {query[:100]}...")
        
        try:
            # Get collection
            collection = self.client.get_collection(name=collection_name)
            
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
            for i in range(len(results["documents"][0])):
                result = {
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "images": json.loads(results["metadatas"][0][i].get("images", "[]")),
                    "tables": json.loads(results["metadatas"][0][i].get("tables", "[]"))
                }
                formatted_results.append(result)
            
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