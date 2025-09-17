"""
Pydantic schemas for request/response models

Version: 0.1
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class PDFUploadResponse(BaseModel):
    success: bool
    message: str
    pdf_name: str
    chunks_processed: int
    processing_time: str
    collection_name: str

class ImageData(BaseModel):
    """Image data with metadata"""
    filename: str
    data: str  # Base64 encoded image data
    mime_type: str
    size: int

class QueryRequest(BaseModel):
    pdf_name: str = Field(..., description="Name of the PDF file to query")
    query: str = Field(..., description="Query text")
    top_k: int = Field(default=5, description="Number of top results to return")

class QueryResponse(BaseModel):
    success: bool
    message: str
    response: str
    chunks_used: List[str]
    images: List[ImageData]  # Actual image data instead of URLs
    tables: List[str]
    processing_time: str
    # Image suggestions and usage tracking
    suggested_images: Optional[List[str]] = None  # Names of images suggested by LLM
    images_used_for_response: Optional[List[str]] = None  # Names of images used in response generation
    # Dynamic processing metadata
    processing_stages: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    initial_chunks_count: Optional[int] = None
    total_chunks_count: Optional[int] = None
    collection_used: Optional[str] = None

class PDFListItem(BaseModel):
    collection_name: str
    pdf_name: str
    created_at: datetime
    chunk_count: int

class PDFListResponse(BaseModel):
    success: bool
    pdfs: List[PDFListItem]
    total_count: int

class Rule(BaseModel):
    name: str
    description: str
    metric: str
    metric_value: str
    threshold: str
    consequence: str
    condition: str
    action: str
    priority: str

class RulesResponse(BaseModel):
    success: bool
    message: str
    processing_time: str
    rules: List[Rule]

class MaintenanceTask(BaseModel):
    task: str
    task_name: str
    description: str
    frequency: str
    priority: str
    estimated_duration: str
    required_tools: str
    category: str
    safety_notes: str

class MaintenanceResponse(BaseModel):
    success: bool
    message: str
    processing_time: str
    maintenance_tasks: List[MaintenanceTask]

class SafetyInfo(BaseModel):
    title: str
    description: str
    category: str
    severity: str
    mitigation: str
    about_reaction: str
    causes: str
    how_to_avoid: str
    safety_info: str
    type: str
    recommended_action: str

class SafetyResponse(BaseModel):
    success: bool
    message: str
    processing_time: str
    safety_precautions: List[SafetyInfo]
    safety_information: List[SafetyInfo]

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    processing_time: Optional[str] = None

class ChunkData(BaseModel):
    heading: str
    text: str
    image_paths: List[str] = []  # Original image paths for reference
    embedded_images: List[ImageData] = []  # Actual embedded image data
    tables: List[str] = []

class PDFDeleteResponse(BaseModel):
    success: bool
    message: str
    pdf_name: str