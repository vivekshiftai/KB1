import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str
    azure_openai_endpoint: Optional[str] = None
    azure_openai_key: Optional[str] = None
    
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
    
    # MinerU Configuration
    device_mode: str = "cuda"  # or "cpu"
    formula_enable: bool = True
    table_enable: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Ensure directories exist
Path(settings.upload_dir).mkdir(exist_ok=True)
Path(settings.output_dir).mkdir(exist_ok=True)
Path(settings.models_dir).mkdir(exist_ok=True)
Path(settings.chromadb_path).mkdir(exist_ok=True)