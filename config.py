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
    # Azure AI Configuration
    azure_openai_key: str
    azure_openai_endpoint: Optional[str] = "https://chgai.services.ai.azure.com/models"
    
    # O3-mini specific configuration
    o3_azure_endpoint: Optional[str] = "https://chgai.cognitiveservices.azure.com/"
    o3_deployment_name: str = "o3-mini"
    
    # Model Configuration for Different Use Cases
    # Maintenance Generation Model
    maintenance_model_name: str = "o3-mini"
    maintenance_model_endpoint: Optional[str] = "https://chgai.cognitiveservices.azure.com/"
    
    # Rules Generation Model
    rules_model_name: str = "o3-mini"
    rules_model_endpoint: Optional[str] = "https://chgai.cognitiveservices.azure.com/"
    
    # Safety Generation Model
    safety_model_name: str = "o3-mini"
    safety_model_endpoint: Optional[str] = "https://chgai.cognitiveservices.azure.com/"
    
    # Query Processing Model
    query_model_name: str = "gpt-4o"
    query_model_endpoint: Optional[str] = "https://chgai.services.ai.azure.com/models"
    
    # Query Analysis Model
    analysis_model_name: str = "gpt-4o"
    analysis_model_endpoint: Optional[str] = "https://chgai.services.ai.azure.com/models"
    
    # Vector Database Configuration
    vector_db_type: str = "chromadb"
    chromadb_path: str = "./vector_db"
    
    # File Paths
    models_dir: str = "./pdf_extract_kit_models"
    upload_dir: str = "./uploads"
    output_dir: str = "./processed"
    
    # Application Settings
    log_level: str = "INFO"  # Force INFO level for visibility
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    max_chunks_per_batch: int = 25
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # MinerU Configuration
    device_mode: str = "cpu"  # Force CPU mode
    formula_enable: bool = True
    table_enable: bool = True
    image_enable: bool = True  # Enable image extraction
    
    # Image Compression Settings
    image_compression_enabled: bool = True  # Enable image compression
    image_max_dimension: int = 1200  # Maximum width/height for compression
    image_quality: int = 85  # JPEG quality (85 is good balance)
    image_max_size: int = 1024 * 1024  # 1MB limit before compression
    
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