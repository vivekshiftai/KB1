"""
CPU Optimization Utility for PDF Intelligence Platform
Ensures all operations use CPU instead of GPU

Version: 0.1
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def force_cpu_mode():
    """Force all operations to use CPU instead of GPU"""
    
    # Disable CUDA/GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_DEVICE"] = "cpu"
    
    # Disable tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Limit thread usage for CPU optimization
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    
    # Allow online model downloads
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    
    # Additional CPU optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    logger.info("CPU mode enforced - CUDA disabled, thread limits set")

def get_cpu_environment() -> dict:
    """Get environment variables for CPU-only operation"""
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "TORCH_DEVICE": "cpu",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "TRANSFORMERS_OFFLINE": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
    }

def verify_cpu_mode():
    """Verify that CPU mode is active"""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    torch_device = os.environ.get("TORCH_DEVICE", "")
    
    if cuda_visible == "" and torch_device == "cpu":
        logger.info("✅ CPU mode verified - CUDA disabled, PyTorch set to CPU")
        return True
    else:
        logger.warning("⚠️ CPU mode not fully enforced")
        return False

def optimize_for_cpu():
    """Apply CPU optimizations for better performance"""
    
    # Force CPU mode
    force_cpu_mode()
    
    # Verify CPU mode
    verify_cpu_mode()
    
    logger.info("CPU optimizations applied successfully")
