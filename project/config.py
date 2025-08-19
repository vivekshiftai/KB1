"""
PDF Intelligence Platform - Configuration Management
Centralized configuration settings for the application

Version: 0.1
"""

import os
from pydantic_settings import BaseSettings
from pydantic import validator
from typing import Optional

class Settings(BaseSettings):
    # Azure AI Configuration (replacing OpenAI)
    azure_endpoint: str = "https://chgai.services.ai.azure.com/models"
    azure_api_key: str = ""
    azure_model_name: str = "Llama-3.2-90B-Vision-Instruct"
    azure_api_version: str = "2024-05-01-preview"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Map environment variables to field names
        fields = {
            "azure_endpoint": {"env": "AZURE_ENDPOINT"},
            "azure_api_key": {"env": "AZURE_API_KEY"},
            "azure_model_name": {"env": "AZURE_MODEL_NAME"},
            "azure_api_version": {"env": "AZURE_API_VERSION"},
            "chroma_db_path": {"env": "CHROMA_DB_PATH"},
            "embedding_model": {"env": "EMBEDDING_MODEL"},
            "upload_dir": {"env": "UPLOAD_DIR"},
            "output_dir": {"env": "OUTPUT_DIR"},
            "max_file_size": {"env": "MAX_FILE_SIZE"},
            "device_mode": {"env": "DEVICE_MODE"},
            "formula_enable": {"env": "FORMULA_ENABLE"},
            "table_enable": {"env": "TABLE_ENABLE"},
            "image_enable": {"env": "IMAGE_ENABLE"},
            "force_cpu": {"env": "FORCE_CPU"},
            "torch_device": {"env": "TORCH_DEVICE"},
            "max_tokens": {"env": "MAX_TOKENS"},
            "max_completion_tokens": {"env": "MAX_COMPLETION_TOKENS"},
            "max_context_tokens": {"env": "MAX_CONTEXT_TOKENS"},
            "log_level": {"env": "LOG_LEVEL"}
        }
    
    # Vector Database Configuration
    chroma_db_path: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # PDF Processing Configuration
    upload_dir: str = "./uploads"
    output_dir: str = "./outputs"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    
    # MinerU Configuration
    device_mode: str = "cpu"  # Force CPU mode
    formula_enable: bool = True
    table_enable: bool = True
    image_enable: bool = True  # Enable image extraction
    
    # CPU Optimization Settings
    force_cpu: bool = True
    torch_device: str = "cpu"
    
    # Token Management
    max_tokens: int = 4000
    max_completion_tokens: int = 1500
    max_context_tokens: int = 8000
    
    # Logging
    log_level: str = "INFO"
    
    @validator('azure_api_key')
    def validate_azure_api_key(cls, v):
        if not v or v == "your_azure_api_key_here":
            raise ValueError("Azure API key is required. Please set AZURE_API_KEY in your .env file")
        return v

# Create settings instance
settings = Settings()

# Force CPU mode globally
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"