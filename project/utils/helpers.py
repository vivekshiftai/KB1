"""
Helper functions and utilities

Version: 0.1
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
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
    """Calculate and format processing time"""
    elapsed = time.time() - start_time
    if elapsed < 60:
        return f"{elapsed:.2f}s"
    elif elapsed < 3600:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.2f}s"
    else:
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        return f"{hours}h {minutes}m"

def optimize_chunks_for_llm(chunks: List[Dict[str, Any]], max_chunks: int = 10, max_tokens_per_chunk: int = 2000) -> List[Dict[str, Any]]:
    """
    Optimize chunks for LLM processing to reduce latency
    
    Args:
        chunks: List of document chunks
        max_chunks: Maximum number of chunks to process
        max_tokens_per_chunk: Maximum tokens per chunk
    
    Returns:
        Optimized list of chunks
    """
    if not chunks:
        return []
    
    # Sort chunks by relevance (if available) or use first chunks
    sorted_chunks = chunks[:max_chunks]
    
    # Truncate chunks that are too long
    optimized_chunks = []
    for chunk in sorted_chunks:
        try:
            # Handle both possible chunk structures
            if "metadata" in chunk and "document" in chunk:
                # Vector DB format
                heading = chunk.get("metadata", {}).get("heading", "")
                content = chunk.get("document", "")
            else:
                # Fallback format
                heading = chunk.get("heading", "")
                content = chunk.get("text", chunk.get("content", ""))
            
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(content) // 4
            
            if estimated_tokens > max_tokens_per_chunk:
                # Truncate content
                max_chars = max_tokens_per_chunk * 4
                truncated_content = content[:max_chars] + " [Content truncated]"
                
                # Update chunk with truncated content
                if "metadata" in chunk and "document" in chunk:
                    chunk["document"] = truncated_content
                else:
                    chunk["text"] = truncated_content
                    chunk["content"] = truncated_content
            
            optimized_chunks.append(chunk)
            
        except Exception as e:
            logging.warning(f"Error optimizing chunk: {str(e)}")
            continue
    
    return optimized_chunks

def estimate_llm_response_time(chunks: List[Dict[str, Any]], query_length: int = 100) -> float:
    """
    Estimate LLM response time based on input size
    
    Args:
        chunks: List of document chunks
        query_length: Length of user query
    
    Returns:
        Estimated response time in seconds
    """
    if not chunks:
        return 5.0  # Base time for simple queries
    
    # Calculate total input size
    total_chars = query_length
    for chunk in chunks:
        try:
            if "metadata" in chunk and "document" in chunk:
                content = chunk.get("document", "")
            else:
                content = chunk.get("text", chunk.get("content", ""))
            total_chars += len(content)
        except:
            continue
    
    # Rough estimation: 1 second per 1000 characters + base time
    estimated_time = (total_chars / 1000) + 3.0
    
    # Cap at reasonable maximum
    return min(estimated_time, 60.0)

def create_optimized_prompt(context: str, query: str, max_context_length: int = 8000) -> str:
    """
    Create an optimized prompt for LLM processing
    
    Args:
        context: Context information
        query: User query
        max_context_length: Maximum context length in characters
    
    Returns:
        Optimized prompt
    """
    # Truncate context if too long
    if len(context) > max_context_length:
        context = context[:max_context_length] + " [Context truncated]"
    
    # Create concise prompt
    prompt = f"""Answer this query based on the provided context:

Context: {context}

Query: {query}

Provide a clear, concise answer. Include relevant details from the context."""
    
    return prompt

class LLMPerformanceMonitor:
    """Monitor LLM performance and latency"""
    
    def __init__(self):
        self.response_times = []
        self.error_count = 0
        self.timeout_count = 0
        self.total_requests = 0
    
    def record_response_time(self, response_time: float):
        """Record response time for monitoring"""
        self.response_times.append(response_time)
        self.total_requests += 1
    
    def record_error(self, error_type: str = "general"):
        """Record error for monitoring"""
        self.error_count += 1
        if "timeout" in error_type.lower():
            self.timeout_count += 1
    
    def get_average_response_time(self) -> float:
        """Get average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = self.get_average_response_time()
        return {
            "total_requests": self.total_requests,
            "average_response_time": avg_time,
            "error_rate": (self.error_count / self.total_requests * 100) if self.total_requests > 0 else 0,
            "timeout_rate": (self.timeout_count / self.total_requests * 100) if self.total_requests > 0 else 0,
            "recent_response_times": self.response_times[-10:] if self.response_times else []
        }
    
    def is_performance_degrading(self) -> bool:
        """Check if performance is degrading"""
        if len(self.response_times) < 5:
            return False
        
        recent_avg = sum(self.response_times[-5:]) / 5
        overall_avg = self.get_average_response_time()
        
        # If recent average is 50% higher than overall average, performance is degrading
        return recent_avg > overall_avg * 1.5

# Global performance monitor instance
llm_monitor = LLMPerformanceMonitor()