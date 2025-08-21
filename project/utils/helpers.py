"""
Helper functions and utilities

Version: 0.1
"""

import os
import time
import logging
from pathlib import Path
import re

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    # Limit length
    if len(sanitized) > 100:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:95] + ext
    return sanitized

def calculate_processing_time(start_time: float) -> str:
    """Calculate processing time in seconds"""
    return f"{time.time() - start_time:.2f}s"

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
        
        # Try to open with PyMuPDF for additional validation
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
            
            # Consider it valid if we can open it and it has at least one page
            return page_count > 0
        except Exception as e:
            # Log the specific error for debugging
            logging.warning(f"PyMuPDF validation failed for {file_path}: {str(e)}")
            # Still return True if magic number check passed
            return True
        
    except Exception as e:
        logging.error(f"PDF validation error for {file_path}: {str(e)}")
        return False