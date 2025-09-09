# Embeddings Implementation Guide

## Technical Implementation Details

This guide provides detailed technical information about implementing and extending the embeddings system in the PDF Intelligence Platform.

## 1. Core Components

### 1.1 Embedding Model Initialization

```python
# services/vector_db.py
from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL_SINGLETON: Optional[SentenceTransformer] = None

def _get_embedding_model(model_name: str) -> SentenceTransformer:
    global _EMBEDDING_MODEL_SINGLETON
    if _EMBEDDING_MODEL_SINGLETON is None:
        # Force CPU usage for sentence transformers
        _EMBEDDING_MODEL_SINGLETON = SentenceTransformer(model_name, device='cpu')
        logger.info(f"Initialized SentenceTransformer model '{model_name}' on CPU")
    return _EMBEDDING_MODEL_SINGLETON
```

**Key Features:**
- Singleton pattern for memory efficiency
- CPU-only deployment for compatibility
- Automatic model loading and caching

### 1.2 Chunking Implementation

```python
# services/chunking.py
class MarkdownChunker:
    def __init__(self):
        self.heading_pattern = re.compile(r"^#+\s+.*")
        self.image_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
        self.table_pattern = re.compile(r"(<table>.*?</table>)", re.DOTALL | re.IGNORECASE)
    
    def chunk_markdown_with_headings(self, md_path: str) -> List[ChunkData]:
        chunks = []
        heading = None
        content_lines = []
        images = []
        tables = []
        
        with open(md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            stripped = line.strip()
            
            # Detect heading
            if stripped.startswith("#"):
                # Save previous chunk
                if heading is not None:
                    chunk_text = "".join(content_lines).strip()
                    if chunk_text or images or tables:
                        chunks.append(ChunkData(
                            heading=heading.strip(),
                            text=chunk_text,
                            images=images.copy(),
                            tables=tables.copy()
                        ))
                
                # Start new chunk
                heading = line.strip()
                content_lines.clear()
                images.clear()
                tables.clear()
                continue
            
            # Extract images and tables
            img_matches = self.image_pattern.findall(line)
            table_matches = self.table_pattern.findall(line)
            
            if img_matches:
                images.extend(img_matches)
            if table_matches:
                tables.extend(table_matches)
            
            if heading is not None:
                content_lines.append(line)
        
        # Save final chunk
        if heading is not None:
            chunk_text = "".join(content_lines).strip()
            if chunk_text or images or tables:
                chunks.append(ChunkData(
                    heading=heading.strip(),
                    text=chunk_text,
                    images=images.copy(),
                    tables=tables.copy()
                ))
        
        return chunks
```

## 2. Vector Database Operations

### 2.1 ChromaDB Setup

```python
# services/vector_db.py
def _setup_chromadb_client(self):
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
```

### 2.2 Chunk Storage

```python
async def store_chunks(self, chunks: List[ChunkData], collection_name: str, output_dir: str = None) -> int:
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
    
    return len(chunks)
```

### 2.3 Query Processing

```python
async def query_chunks(self, collection_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
        # Parse images and tables from metadata
        images = json.loads(results["metadatas"][0][i].get("images", "[]"))
        tables = json.loads(results["metadatas"][0][i].get("tables", "[]"))
        
        result = {
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "images": images,
            "tables": tables
        }
        formatted_results.append(result)
    
    return formatted_results
```

## 3. LLM Integration

### 3.1 Token Management

```python
# services/llm_service.py
def truncate_chunks_to_fit_context(self, chunks: List[Dict[str, Any]], system_prompt: str = "", user_prompt_template: str = "") -> List[Dict[str, Any]]:
    # Calculate available tokens for chunks
    system_tokens = self.count_tokens(system_prompt)
    user_template_tokens = self.count_tokens(user_prompt_template)
    reserved_tokens = system_tokens + user_template_tokens + 200  # Buffer
    
    available_tokens = self.max_context_tokens - reserved_tokens
    
    selected_chunks = []
    current_tokens = 0
    
    for i, chunk in enumerate(chunks):
        # Handle both possible chunk structures
        if "metadata" in chunk and "document" in chunk:
            heading = chunk.get("metadata", {}).get("heading", "")
            content = chunk.get("document", "")
        else:
            heading = chunk.get("heading", "")
            content = chunk.get("text", chunk.get("content", ""))
        
        chunk_text = f"**{heading}**\n{content}"
        chunk_tokens = self.count_tokens(chunk_text)
        
        if current_tokens + chunk_tokens <= available_tokens:
            selected_chunks.append(chunk)
            current_tokens += chunk_tokens
        else:
            break
    
    return selected_chunks
```

### 3.2 Context Assembly

```python
async def query_with_context(self, chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    # System prompt
    system_prompt = "You are a technical documentation assistant. Provide accurate, detailed answers based on the provided manual sections."
    
    # User prompt template
    user_prompt_template = """Answer the user's query based on the provided context.

Context:
{context}

Query: {query}

Provide a clear answer. At the end, add "REFERENCES:" followed by the exact section headings you used."""
    
    # Truncate chunks to fit within token limit
    selected_chunks = self.truncate_chunks_to_fit_context(chunks, system_prompt, user_prompt_template)
    
    # Prepare context from selected chunks
    context_parts = []
    for chunk in selected_chunks:
        if "metadata" in chunk and "document" in chunk:
            heading = chunk.get("metadata", {}).get("heading", "")
            content = chunk.get("document", "")
        else:
            heading = chunk.get("heading", "")
            content = chunk.get("text", chunk.get("content", ""))
        
        if content:
            context_parts.append(f"**{heading}**\n{content}")
    
    context = "\n\n".join(context_parts)
    
    # Format the user prompt with context
    user_prompt = user_prompt_template.format(context=context, query=query)
    
    # Call LLM
    response = self.client.complete(
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt)
        ],
        max_tokens=self.max_completion_tokens,
        temperature=0.1,
        top_p=0.1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        model=self.model_name
    )
    
    answer = response.choices[0].message.content
    
    # Parse referenced sections
    chunks_used = self._extract_referenced_sections(answer, chunks)
    
    return {
        "response": answer,
        "chunks_used": chunks_used
    }
```

## 4. Image Storage Implementation

### 4.1 Image Processing

```python
async def _store_images(self, collection_name: str, output_dir: str):
    import base64
    from pathlib import Path
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(output_dir).glob(f"**/*{ext}"))
        image_files.extend(Path(output_dir).glob(f"**/*{ext.upper()}"))
    
    # Store each image
    for image_file in image_files:
        try:
            # Read and encode image
            with open(image_file, "rb") as f:
                image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Get relative path
            relative_path = str(image_file.relative_to(Path(output_dir)))
            
            # Prepare metadata
            image_metadata = {
                "type": "image",
                "filename": image_file.name,
                "path": relative_path,
                "size": len(image_data),
                "format": image_file.suffix.lower(),
                "collection": collection_name,
                "created_at": datetime.now().isoformat()
            }
            
            # Store in separate collection
            images_collection_name = f"{collection_name}_images"
            images_collection = self.client.get_or_create_collection(
                name=images_collection_name,
                metadata={"created_at": datetime.now().isoformat()}
            )
            
            images_collection.add(
                ids=[f"img_{relative_path}"],
                documents=[image_base64],
                metadatas=[image_metadata]
            )
            
        except Exception as e:
            logger.error(f"Error storing image {image_file}: {str(e)}")
            continue
```

### 4.2 Image Retrieval

```python
async def get_image(self, collection_name: str, image_path: str) -> Optional[Dict[str, Any]]:
    try:
        # Get images collection
        images_collection_name = f"{collection_name}_images"
        images_collection = self.client.get_collection(name=images_collection_name)
        
        # Try direct lookup
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
                    "data": image_data,
                    "metadata": metadata,
                    "content_type": self._get_content_type(metadata.get("format", ""))
                }
        except Exception:
            pass
        
        # Fallback: Query method
        results = images_collection.query(
            query_texts=["image"],
            n_results=1000,
            include=["documents", "metadatas"]
        )
        
        # Find image by path
        for i, metadata in enumerate(results["metadatas"][0]):
            if metadata.get("path") == image_path:
                image_data = results["documents"][0][i]
                return {
                    "data": image_data,
                    "metadata": metadata,
                    "content_type": self._get_content_type(metadata.get("format", ""))
                }
        
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving image {image_path}: {str(e)}")
        return None
```

## 5. Configuration Management

### 5.1 Settings Configuration

```python
# config.py
class Settings(BaseSettings):
    # Vector Database Configuration
    vector_db_type: str = "chromadb"
    chromadb_path: str = "./vector_db"
    
    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Processing Settings
    max_chunks_per_batch: int = 25
    device_mode: str = "cpu"
    
    # CPU Optimization
    force_cpu: bool = True
    torch_device: str = "cpu"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

# Global settings instance
settings = Settings()

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
```

### 5.2 Environment Variables

```bash
# .env file
AZURE_OPENAI_KEY=your_azure_key_here
VECTOR_DB_TYPE=chromadb
CHROMADB_PATH=./vector_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_CHUNKS_PER_BATCH=25
DEVICE_MODE=cpu
FORCE_CPU=true
TORCH_DEVICE=cpu
```

## 6. Error Handling and Logging

### 6.1 Comprehensive Error Handling

```python
def safe_embedding_generation(self, text: str) -> Optional[List[float]]:
    """Safely generate embeddings with error handling"""
    try:
        embedding = self.embedding_model.encode(text).tolist()
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return None

def safe_chunk_storage(self, chunks: List[ChunkData], collection_name: str) -> int:
    """Safely store chunks with error handling"""
    try:
        return await self.store_chunks(chunks, collection_name)
    except Exception as e:
        logger.error(f"Error storing chunks: {str(e)}")
        raise e

def safe_query_processing(self, collection_name: str, query: str) -> List[Dict[str, Any]]:
    """Safely process queries with error handling"""
    try:
        return await self.query_chunks(collection_name, query)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return []
```

### 6.2 Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embeddings.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

## 7. Performance Optimization

### 7.1 Batch Processing

```python
def process_chunks_in_batches(self, chunks: List[ChunkData], batch_size: int = 25):
    """Process chunks in batches for memory efficiency"""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        yield batch

async def store_chunks_batched(self, chunks: List[ChunkData], collection_name: str):
    """Store chunks in batches"""
    total_stored = 0
    for batch in self.process_chunks_in_batches(chunks):
        stored = await self.store_chunks(batch, collection_name)
        total_stored += stored
        logger.info(f"Stored batch: {stored} chunks, total: {total_stored}")
    return total_stored
```

### 7.2 Memory Management

```python
def optimize_memory_usage(self):
    """Optimize memory usage for large documents"""
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Clear model cache if needed
    if hasattr(self.embedding_model, 'clear_cache'):
        self.embedding_model.clear_cache()
    
    # Monitor memory usage
    import psutil
    memory_usage = psutil.virtual_memory()
    logger.info(f"Memory usage: {memory_usage.percent}%")
```

## 8. Testing and Validation

### 8.1 Embedding Quality Tests

```python
def test_embedding_quality(self):
    """Test embedding quality with sample queries"""
    test_queries = [
        "temperature monitoring",
        "maintenance procedures",
        "safety warnings",
        "operational parameters"
    ]
    
    for query in test_queries:
        embedding = self.embedding_model.encode(query).tolist()
        assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"
        assert all(isinstance(x, float) for x in embedding), "All values should be floats"
    
    logger.info("Embedding quality tests passed")

def test_similarity_scoring(self):
    """Test semantic similarity between related concepts"""
    related_pairs = [
        ("temperature", "heat"),
        ("maintenance", "repair"),
        ("safety", "protection"),
        ("operation", "function")
    ]
    
    for pair in related_pairs:
        emb1 = self.embedding_model.encode(pair[0])
        emb2 = self.embedding_model.encode(pair[1])
        
        similarity = self.cosine_similarity(emb1, emb2)
        logger.info(f"Similarity between '{pair[0]}' and '{pair[1]}': {similarity:.3f}")
```

### 8.2 Integration Tests

```python
async def test_full_pipeline(self):
    """Test the complete embedding pipeline"""
    # Test chunking
    chunks = self.chunker.chunk_markdown_with_headings("test.md")
    assert len(chunks) > 0, "Should generate chunks"
    
    # Test embedding generation
    for chunk in chunks:
        embedding = self.embedding_model.encode(f"{chunk.heading}\n{chunk.text}").tolist()
        assert len(embedding) == 384, "Should generate 384-dimensional embeddings"
    
    # Test storage
    collection_name = "test_collection"
    stored_count = await self.store_chunks(chunks, collection_name)
    assert stored_count == len(chunks), "Should store all chunks"
    
    # Test querying
    results = await self.query_chunks(collection_name, "test query")
    assert len(results) > 0, "Should return query results"
    
    logger.info("Full pipeline test passed")
```

## 9. Monitoring and Metrics

### 9.1 Performance Metrics

```python
import time
from typing import Dict, Any

class EmbeddingMetrics:
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = {"start": time.time()}
    
    def end_timer(self, operation: str) -> float:
        """End timing and return duration"""
        if operation in self.metrics:
            duration = time.time() - self.metrics[operation]["start"]
            self.metrics[operation]["duration"] = duration
            self.metrics[operation]["end"] = time.time()
            return duration
        return 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return self.metrics.copy()

# Usage
metrics = EmbeddingMetrics()

# Time embedding generation
metrics.start_timer("embedding_generation")
embedding = model.encode(text)
duration = metrics.end_timer("embedding_generation")
logger.info(f"Embedding generation took {duration:.3f} seconds")
```

### 9.2 Health Checks

```python
async def health_check(self) -> Dict[str, Any]:
    """Perform system health check"""
    health_status = {
        "embedding_model": "unknown",
        "vector_database": "unknown",
        "memory_usage": "unknown",
        "disk_space": "unknown"
    }
    
    try:
        # Test embedding model
        test_embedding = self.embedding_model.encode("test").tolist()
        health_status["embedding_model"] = "healthy"
    except Exception as e:
        health_status["embedding_model"] = f"error: {str(e)}"
    
    try:
        # Test vector database
        collections = self.client.list_collections()
        health_status["vector_database"] = "healthy"
    except Exception as e:
        health_status["vector_database"] = f"error: {str(e)}"
    
    try:
        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        health_status["memory_usage"] = f"{memory.percent}%"
    except Exception as e:
        health_status["memory_usage"] = f"error: {str(e)}"
    
    try:
        # Check disk space
        import psutil
        disk = psutil.disk_usage(settings.chromadb_path)
        health_status["disk_space"] = f"{disk.percent}% used"
    except Exception as e:
        health_status["disk_space"] = f"error: {str(e)}"
    
    return health_status
```

## 10. Troubleshooting Guide

### 10.1 Common Issues

#### Memory Issues
```python
# Solution: Reduce batch size
settings.max_chunks_per_batch = 10  # Reduce from 25

# Solution: Force garbage collection
import gc
gc.collect()
```

#### Slow Embedding Generation
```python
# Solution: Use smaller model
settings.embedding_model = "all-MiniLM-L6-v2"  # Already optimized

# Solution: Reduce text length
def truncate_text_for_embedding(self, text: str, max_length: int = 1000) -> str:
    return text[:max_length] if len(text) > max_length else text
```

#### Vector Database Errors
```python
# Solution: Reset database
import shutil
shutil.rmtree(settings.chromadb_path)
os.makedirs(settings.chromadb_path, exist_ok=True)

# Solution: Check disk space
import psutil
disk = psutil.disk_usage(settings.chromadb_path)
if disk.percent > 90:
    logger.warning("Low disk space detected")
```

### 10.2 Debug Mode

```python
# Enable debug logging
logging.getLogger().setLevel(logging.DEBUG)

# Add debug information to operations
def debug_embedding_generation(self, text: str):
    logger.debug(f"Generating embedding for text: {text[:100]}...")
    embedding = self.embedding_model.encode(text).tolist()
    logger.debug(f"Generated embedding with {len(embedding)} dimensions")
    return embedding
```

This implementation guide provides the technical foundation for understanding, maintaining, and extending the embeddings system. Each section includes practical code examples and best practices for production deployment.
