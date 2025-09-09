# Azure AI Setup Guide

This guide will help you set up Azure AI integration for the PDF Intelligence Platform.

## Prerequisites

1. **Azure Account**: You need an active Azure account
2. **Azure AI Service**: Access to Azure AI services with the Llama-3.2-90B-Vision-Instruct model
3. **API Key**: Your Azure AI API key

## Configuration Steps

### 1. Get Your Azure AI API Key

1. Log into the [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure AI service
3. Go to "Keys and Endpoint" section
4. Copy your API key

### 2. Configure Environment Variables

Create a `.env` file in the project root with the following content:

```bash
# Azure AI Configuration (Required)
AZURE_OPENAI_KEY=your_azure_ai_api_key_here
AZURE_OPENAI_ENDPOINT=https://chgai.services.ai.azure.com/models

# Optional - Legacy OpenAI Configuration (for fallback)
OPENAI_API_KEY=your_openai_key_here

# Database Configuration
VECTOR_DB_TYPE=chromadb
CHROMADB_PATH=./vector_db

# File Storage
MODELS_DIR=./pdf_extract_kit_models
UPLOAD_DIR=./uploads
OUTPUT_DIR=./processed

# Processing Configuration
MAX_FILE_SIZE=52428800
MAX_CHUNKS_PER_BATCH=25
EMBEDDING_MODEL=all-MiniLM-L6-v2

# MinerU Configuration
DEVICE_MODE=cpu
FORMULA_ENABLE=true
TABLE_ENABLE=true
IMAGE_ENABLE=true

# Logging
LOG_LEVEL=INFO
```

### 3. Install Dependencies

```bash
# Install Azure AI dependencies
pip install azure-ai-inference>=1.0.0 azure-core>=1.29.0

# Or install all requirements
pip install -r requirements.txt
```

### 4. Test the Integration

Run the test script to verify your Azure AI setup:

```bash
python test_setup.py
```

You should see output like:
```
PDF Intelligence Platform - Setup Test
==================================================
Testing Environment Setup...
✓ .env file found
✓ AZURE_OPENAI_KEY is set
  Key length: 32 characters
✓ AZURE_OPENAI_ENDPOINT is set: https://chgai.services.ai.azure.com/models

Testing Module Imports...
✓ FastAPI imported successfully
✓ Uvicorn imported successfully
✓ Azure AI Inference imported successfully
✓ Azure Core imported successfully
✓ Tiktoken imported successfully
✓ ChromaDB imported successfully
✓ Sentence Transformers imported successfully

Testing Configuration...
✓ Configuration loaded successfully
✓ Azure AI key configured: Yes
✓ Azure AI endpoint: https://chgai.services.ai.azure.com/models
✓ Vector DB type: chromadb
✓ Upload directory: ./uploads
✓ Output directory: ./processed

Testing LLM Service...
✓ LLM Service module imported successfully
✓ LLM Service initialized successfully

==================================================
Test completed!
```

## Azure AI Configuration Details

The platform uses the following Azure AI configuration:

- **Endpoint**: `https://chgai.services.ai.azure.com/models`
- **Model**: `Llama-3.2-90B-Vision-Instruct`
- **API Version**: `2024-05-01-preview`
- **Max Tokens**: 8192
- **Temperature**: 0.1 (for consistent responses)

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your `AZURE_OPENAI_KEY` is correctly set in the `.env` file
2. **Endpoint Error**: Verify the endpoint URL is correct
3. **Model Not Found**: Ensure you have access to the Llama-3.2-90B-Vision-Instruct model
4. **Rate Limiting**: Azure AI has rate limits - check your service tier

### Error Messages

- `AuthenticationError`: Check your API key
- `ModelNotFoundError`: Verify model access
- `RateLimitError`: Reduce request frequency or upgrade service tier

### Support

If you encounter issues:
1. Check the Azure AI service status
2. Verify your API key and endpoint
3. Review the Azure AI documentation
4. Check the application logs for detailed error messages

## Migration from OpenAI

If you're migrating from OpenAI to Azure AI:

1. Update your `.env` file with Azure AI credentials
2. The platform will automatically use Azure AI instead of OpenAI
3. All existing functionality remains the same
4. Test the integration using `python test_azure_ai.py`

## Performance Notes

- Azure AI responses may have slightly different characteristics than OpenAI
- The Llama-3.2-90B-Vision-Instruct model is optimized for technical documentation
- Token limits and response times may vary based on your Azure AI service tier
