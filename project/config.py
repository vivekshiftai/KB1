"""
PDF Intelligence Platform - Configuration Management
Centralized configuration settings for the application

Version: 0.1
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Azure AI Configuration (replacing OpenAI)
    azure_endpoint: str = "https://chgai.services.ai.azure.com/models"
    azure_api_key: str = ""
    azure_model_name: str = "Llama-3.2-90B-Vision-Instruct"
    azure_api_version: str = "2024-05-01-preview"
    
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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()

# Force CPU mode globally
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"