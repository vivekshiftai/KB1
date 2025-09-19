# PDF Intelligence Platform

A comprehensive backend API system that processes PDF manuals using MinerU, stores them in vector databases, and provides intelligent querying capabilities with advanced image processing and inline image references for IoT device documentation, rules generation, maintenance schedules, and safety information.

## 🚀 Features

- **PDF Processing**: Upload and process PDF manuals using MinerU with CPU-based local inference
- **Intelligent Chunking**: Heading-based content chunking with image and table extraction
- **Advanced Image Processing**: Automatic image labeling with customizable fonts and sizes
- **Inline Image References**: LLM generates natural language with embedded image markers
- **Vector Storage**: ChromaDB integration with efficient similarity search
- **Smart Querying**: GPT-4 powered intelligent responses with context awareness and image integration
- **LangGraph Workflow**: Advanced query processing with validation and retry logic
- **Rules Generation**: Automated IoT monitoring rule creation from technical documentation
- **Maintenance Scheduling**: Extract and structure maintenance tasks from manuals
- **Safety Information**: Comprehensive safety guideline generation
- **Production Ready**: Comprehensive error handling, detailed logging, and scalable architecture
- **VM Optimized**: CPU-only processing optimized for virtual machine environments

## 📋 Prerequisites

- Python 3.8+
- Azure OpenAI API key (or OpenAI API key)
- Git
- PIL/Pillow for image processing

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd AGE
   ```

2. **Navigate to the project directory**
   ```bash
   cd project
   ```

3. **Create and activate virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

## ⚙️ Configuration

Create a `.env` file in the `project` directory with the following variables:

```bash
# Required - Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
AZURE_OPENAI_KEY=your_azure_key_here

# Model Configuration
MAINTENANCE_MODEL_NAME=your_model_name
RULES_MODEL_NAME=your_model_name
SAFETY_MODEL_NAME=your_model_name
QUERY_MODEL_NAME=your_model_name
ANALYSIS_MODEL_NAME=your_model_name

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

## 🚀 Quick Start

1. **Start the application**
   ```bash
   python main.py
   ```

2. **Access API documentation**
   Open http://localhost:8000/docs for interactive API documentation

3. **Test the health endpoint**
   ```bash
   curl http://localhost:8000/health
   ```

## 📚 API Endpoints

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

## 💡 Usage Examples

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

## 🖼️ Image Integration

The system provides advanced image processing and integration:

### Response Format
Queries return responses with inline image references:

```json
{
  "success": true,
  "response": "Turn the main switch as shown in below image\n[IMAGE:image 1.jpg]\nCheck the control panel refer to below image\n[IMAGE:image 2.jpg]",
  "images": [
    {
      "filename": "image 1.jpg",
      "data": "base64-encoded-image-data",
      "mime_type": "image/jpeg",
      "size": 12345
    }
  ],
  "suggested_images": ["image 1", "image 2"]
}
```

### Image Features
- **Pre-labeled Images**: Images are labeled before LLM processing
- **Natural References**: Uses "as shown in below image" phrasing
- **Image Markers**: `[IMAGE:filename.jpg]` markers for frontend parsing
- **Clean Filenames**: Simple names like "image 1.jpg", "image 2.jpg"
- **Bold Labels**: 28px bold text labels on white background extensions

## 🏗️ Architecture

The platform follows a modular architecture:

```
project/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── endpoints/           # API route handlers
│   ├── upload.py       # PDF upload and processing
│   ├── query.py        # Content querying
│   ├── pdfs.py         # PDF listing
│   ├── rules.py        # Rules generation
│   ├── maintenance.py  # Maintenance scheduling
│   └── safety.py       # Safety information
├── services/           # Core business logic
│   ├── pdf_processor.py        # PDF processing with MinerU
│   ├── vector_db.py           # Vector database operations
│   ├── chunking.py            # Content chunking logic
│   ├── llm_service.py         # LLM integration with image processing
│   └── langgraph_query_processor.py # Advanced query workflow
├── models/             # Pydantic schemas
│   └── schemas.py      # Request/response models
└── utils/              # Helper functions
    ├── helpers.py      # Utility functions
    ├── cpu_optimizer.py # CPU optimization utilities
    └── image_processor.py # Image labeling and processing
```

## 🔧 Key Features

### PDF Processing
- Downloads PDF-Extract-Kit-1.0 models from HuggingFace
- Processes PDFs using MinerU with local inference
- Extracts text, images, and tables
- Implements heading-based chunking strategy

### Vector Database
- Primary support for ChromaDB with Azure Vector Search fallback
- Efficient similarity search with sentence-transformers embeddings
- Separate collections per PDF for organized storage

### Image Processing & Integration
- Automatic image labeling with customizable fonts and sizes (28px bold text)
- Pre-labeling of images before LLM processing for consistency
- Clean filename generation (image 1.jpg, image 2.jpg) for user-friendly responses
- Inline image reference markers for frontend integration
- Natural language references with embedded image markers

### LLM Integration
- Azure OpenAI GPT-4 powered intelligent responses
- Context-aware query processing with image analysis
- LangGraph workflow for advanced query processing with validation and retry logic
- Structured data generation for rules, maintenance, and safety
- Hardcoded 3-chunk initial processing for faster responses

### Production Features
- Comprehensive error handling and detailed logging with structured output
- Request/response validation with Pydantic schemas
- Async/await patterns for optimal performance
- Health checks and monitoring endpoints
- CORS support and security considerations
- Detailed processing logs for debugging and monitoring

## 🐛 Troubleshooting

### Common Issues

1. **MinerU not found**
   ```bash
   pip install magic-pdf
   ```

2. **Azure OpenAI API key not set**
   - Ensure your `.env` file contains valid `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_KEY`

3. **Port already in use**
   - Change the port in `main.py` or kill the process using port 8000

4. **Memory issues with large PDFs**
   - Reduce `MAX_CHUNKS_PER_BATCH` in your `.env` file
   - Ensure sufficient RAM (recommended: 8GB+)

5. **Image processing issues**
   - Ensure PIL/Pillow is installed: `pip install Pillow`
   - Check logs for font loading issues
   - Images will use fallback fonts if system fonts unavailable

### Logs
- Application logs are written to `app.log` in the project directory
- Check logs for detailed error information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow the existing code structure and patterns
- Add comprehensive logging and error handling
- Include proper type hints and documentation
- Test endpoints thoroughly
- Update README for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MinerU](https://github.com/opendatalab/PDF-Extract-Kit) for PDF processing capabilities
- [ChromaDB](https://www.trychroma.com/) for vector database functionality
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenAI](https://openai.com/) for GPT-4 integration

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the logs for detailed error information
- Review the API documentation at `/docs` when the server is running