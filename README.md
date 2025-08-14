# PDF Intelligence Platform

A comprehensive backend API system that processes PDF manuals using MinerU, stores them in vector databases, and provides intelligent querying capabilities for IoT device documentation, rules generation, maintenance schedules, and safety information.

## Features

- **PDF Processing**: Upload and process PDF manuals using MinerU with local model inference
- **Intelligent Chunking**: Heading-based content chunking with image and table extraction
- **Vector Storage**: ChromaDB integration with Azure Vector Search fallback
- **Smart Querying**: GPT-4 powered intelligent responses with context awareness
- **Rules Generation**: Automated IoT monitoring rule creation from technical documentation
- **Maintenance Scheduling**: Extract and structure maintenance tasks from manuals
- **Safety Information**: Comprehensive safety guideline generation
- **Production Ready**: Comprehensive error handling, logging, and scalable architecture

## Quick Start

1. **Environment Setup**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python main.py
   ```

4. **Access API Documentation**
   Open http://localhost:8000/docs for interactive API documentation

## API Endpoints

### Core Operations
- `POST /upload/pdf` - Upload and process PDF files
- `POST /query` - Query PDF content with intelligent responses
- `GET /pdfs` - List all processed PDFs

### Generation Endpoints
- `POST /generate-rules/{pdf_name}` - Generate IoT monitoring rules
- `POST /generate-maintenance/{pdf_name}` - Extract maintenance schedules
- `POST /generate-safety/{pdf_name}` - Generate safety information

### Utility
- `GET /health` - Health check endpoint
- `GET /` - Service information and available endpoints

## Environment Variables

```
OPENAI_API_KEY=your_openai_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_optional
AZURE_OPENAI_KEY=your_azure_key_optional
VECTOR_DB_TYPE=chromadb
CHROMADB_PATH=./vector_db
MODELS_DIR=./pdf_extract_kit_models
UPLOAD_DIR=./uploads
OUTPUT_DIR=./processed
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
- Comprehensive error handling and logging
- Request/response validation with Pydantic
- Async/await patterns for optimal performance
- Health checks and monitoring endpoints
- CORS support and security considerations

## Usage Examples

### Upload PDF
```bash
curl -X POST "http://localhost:8000/upload/pdf" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@manual.pdf"
```

### Query Content
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"pdf_name": "manual.pdf", "query": "How to install the device?"}'
```

### Generate Rules
```bash
curl -X POST "http://localhost:8000/generate-rules/manual.pdf"
```

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive logging and error handling
3. Include proper type hints and documentation
4. Test endpoints thoroughly
5. Update README for new features

## License

This project is licensed under the MIT License.