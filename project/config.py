"""
PDF Intelligence Platform - Configuration Management
Centralized configuration settings for the application

Version: 0.1
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Azure AI Configuration (Primary)
    azure_openai_key: str
    azure_openai_endpoint: Optional[str] = "https://chgai.services.ai.azure.com/models"
    
    # OpenAI Configuration (Legacy - Optional)
    openai_api_key: Optional[str] = None
    
    # Vector Database Configuration
    vector_db_type: str = "chromadb"
    chromadb_path: str = "./vector_db"
    
    # File Paths
    models_dir: str = "./pdf_extract_kit_models"
    upload_dir: str = "./uploads"
    output_dir: str = "./processed"
    
    # Application Settings
    log_level: str = "INFO"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    max_chunks_per_batch: int = 25
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Timeout Settings
    request_timeout: int = 300  # 5 minutes for request timeout
    llm_timeout: int = 180      # 3 minutes for LLM operations
    upload_timeout: int = 600   # 10 minutes for file uploads
    
    # MinerU Configuration
    device_mode: str = "cpu"  # Force CPU mode
    formula_enable: bool = True
    table_enable: bool = True
    image_enable: bool = True  # Enable image extraction
    
    # CPU Optimization Settings
    force_cpu: bool = True  # Force all operations to use CPU
    torch_device: str = "cpu"  # PyTorch device setting
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields to be ignored

# Global settings instance
settings = Settings()

# Force CPU usage for all operations
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads for CPU optimization

# Ensure directories exist
Path(settings.upload_dir).mkdir(exist_ok=True)
Path(settings.output_dir).mkdir(exist_ok=True)
Path(settings.models_dir).mkdir(exist_ok=True)
Path(settings.chromadb_path).mkdir(exist_ok=True)