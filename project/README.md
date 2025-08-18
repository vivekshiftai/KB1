# PDF Intelligence Platform

A comprehensive backend API system that processes PDF manuals using MinerU, stores them in vector databases, and provides intelligent querying capabilities for IoT device documentation, rules generation, maintenance schedules, and safety information.

## Features

- **PDF Processing**: Upload and process PDF manuals using MinerU with CPU-based local inference
- **Intelligent Chunking**: Heading-based content chunking with image and table extraction
- **Vector Storage**: ChromaDB integration with efficient similarity search
- **Smart Querying**: GPT-4 powered intelligent responses with context awareness
- **Rules Generation**: Automated IoT monitoring rule creation from technical documentation
- **Maintenance Scheduling**: Extract and structure maintenance tasks from manuals
- **Safety Information**: Comprehensive safety guideline generation
- **Production Ready**: Comprehensive error handling, detailed logging, and scalable architecture
- **VM Optimized**: CPU-only processing optimized for virtual machine environments

## Quick Start

1. **Environment Setup**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

2. **Install Dependencies**
   ```bash
   # For production (recommended)
   pip install -r requirements.txt
   
   # For minimal installation (basic functionality only)
   pip install -r requirements-minimal.txt
   
   # For development (includes testing and linting tools)
   pip install -r requirements-dev.txt
   ```

3. **Run the Application**
   ```bash
   # Option 1: Simple startup script (recommended)
   chmod +x start.sh
   ./start.sh
   
   # Option 2: Python startup script
   python3 run.py
   
   # Option 3: Direct Python execution
   python main.py
   
   # Option 4: Using uvicorn directly
   uvicorn main:app --host 0.0.0.0 --port 8000
   
   # Option 5: For development with auto-reload
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access API Documentation**
   Open http://localhost:8000/docs for interactive API documentation

## API Endpoints

### Core Operations
- `POST /upload-pdf` - Upload and process PDF files with MinerU extraction
- `POST /query` - Query PDF content with intelligent responses and context
- `GET /pdfs` - List all processed PDFs with pagination

### Generation Endpoints
- `POST /generate-rules/{pdf_name}` - Generate IoT monitoring rules from PDF content
- `POST /generate-maintenance/{pdf_name}` - Extract maintenance schedules and tasks
- `POST /generate-safety/{pdf_name}` - Generate safety information and procedures

### Utility
- `GET /health` - Health check endpoint
- `GET /` - Service information and available endpoints

## Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_key_here

# Optional (Azure OpenAI fallback)
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_optional
AZURE_OPENAI_KEY=your_azure_key_optional

# Database Configuration
VECTOR_DB_TYPE=chromadb
CHROMADB_PATH=./vector_db

# File Storage
MODELS_DIR=./pdf_extract_kit_models
UPLOAD_DIR=./uploads
OUTPUT_DIR=./processed

# Processing Configuration
MAX_FILE_SIZE=52428800  # 50MB in bytes
MAX_CHUNKS_PER_BATCH=25
EMBEDDING_MODEL=all-MiniLM-L6-v2

# MinerU Configuration
DEVICE_MODE=cpu  # cpu or cuda
FORMULA_ENABLE=true
TABLE_ENABLE=true

# Logging
LOG_LEVEL=INFO
```

## Architecture

The platform follows a modular architecture:

- **Services**: Core business logic (PDF processing, vector database, LLM integration)
- **Endpoints**: API route handlers with validation and error handling
- **Models**: Pydantic schemas for request/response validation
- **Utils**: Helper functions and utilities
- **Config**: Centralized configuration management

## Key Features

### PDF Processing
- Downloads PDF-Extract-Kit-1.0 models from HuggingFace
- Processes PDFs using MinerU with local inference
- Extracts text, images, and tables
- Implements heading-based chunking strategy

### Vector Database
- Primary support for ChromaDB with Azure Vector Search fallback
- Efficient similarity search with sentence-transformers embeddings
- Separate collections per PDF for organized storage

### LLM Integration
- GPT-4 powered intelligent responses
- Context-aware query processing
- Structured data generation for rules, maintenance, and safety

### Production Features
- Comprehensive error handling and detailed logging with structured output
- Request/response validation with Pydantic schemas
- Async/await patterns for optimal performance
- Health checks and monitoring endpoints
- CORS support and security considerations
- Detailed processing logs for debugging and monitoring

## Usage Examples

### Upload PDF
```bash
curl -X POST "http://localhost:8000/upload-pdf" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@manual.pdf"
```

### Query Content
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "pdf_name": "manual.pdf", 
       "query": "How to install the device?",
       "top_k": 5
     }'
```

### List PDFs
```bash
curl -X GET "http://localhost:8000/pdfs?page=1&limit=10"
```

### Generate Rules
```bash
curl -X POST "http://localhost:8000/generate-rules/manual.pdf"
```

### Generate Maintenance Schedule
```bash
curl -X POST "http://localhost:8000/generate-maintenance/manual.pdf"
```

### Generate Safety Information
```bash
curl -X POST "http://localhost:8000/generate-safety/manual.pdf"
```

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive logging and error handling
3. Include proper type hints and documentation
4. Test endpoints thoroughly
5. Update README for new features

## License

This project is licensed under the MIT License.