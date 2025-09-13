"""
Helper functions and utilities

Version: 0.1
"""

import re
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Any, Dict
from functools import wraps

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create handlers
    file_handler = logging.FileHandler("app.log", mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure specific service loggers
    service_loggers = [
        'services.llm_service',
        'services.vector_db', 
        'services.pdf_processor',
        'services.chunking',
        'services.langgraph_query_processor'
    ]
    
    for logger_name in service_loggers:
        service_logger = logging.getLogger(logger_name)
        service_logger.setLevel(logging.INFO)
        service_logger.propagate = True  # Ensure messages propagate to root logger
    
    # Test logging immediately
    test_logger = logging.getLogger(__name__)
    test_logger.info("=== LOGGING SYSTEM INITIALIZED ===")
    test_logger.info(f"Log file: app.log")
    test_logger.info(f"Log level: INFO")
    
    # Force flush to ensure logs are written immediately
    for handler in root_logger.handlers:
        handler.flush()
    
    return test_logger

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for use as collection name"""
    # Remove extension and special characters, use only the PDF name
    base_name = Path(filename).stem
    sanitized = re.sub(r'[^\w\-_]', '_', base_name.lower())
    
    # Return clean PDF name without any prefixes
    return sanitized

def calculate_processing_time(start_time: float) -> str:
    """Calculate processing time in seconds"""
    return f"{time.time() - start_time:.2f}s"

def generate_file_hash(file_path: str) -> str:
    """Generate MD5 hash of file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def timing_decorator(func):
    """Decorator to measure execution time"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        processing_time = calculate_processing_time(start_time)
        if isinstance(result, dict) and 'processing_time' not in result:
            result['processing_time'] = processing_time
        return result
    return wrapper

def extract_images_from_markdown(content: str) -> list:
    """Extract image paths from markdown content"""
    image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    return image_pattern.findall(content)

def extract_tables_from_markdown(content: str) -> list:
    """Extract HTML tables from markdown content"""
    table_pattern = re.compile(r'(<table>.*?</table>)', re.DOTALL | re.IGNORECASE)
    return table_pattern.findall(content)

def validate_pdf_file(file_path: str) -> bool:
    """Validate if file is a valid PDF"""
    try:
        # First check if file exists and has content
        if not Path(file_path).exists():
            return False
        
        file_size = Path(file_path).stat().st_size
        if file_size == 0:
            return False
        
        # Check PDF magic number (first 4 bytes should be %PDF)
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False
        
        # Basic validation passed (magic number check)
        logger.info(f"PDF validation successful: {file_path}")
        return True
            
    except Exception as e:
        logger.error(f"PDF validation error for {file_path}: {str(e)}")
        return False

# Initialize logger at module level
logger = logging.getLogger(__name__)