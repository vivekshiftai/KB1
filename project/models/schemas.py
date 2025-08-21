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
    rule_name: str
    threshold: str
    metric: str
    metric_value: str
    description: str
    consequence: str

class RulesResponse(BaseModel):
    success: bool
    pdf_name: str
    rules: List[Rule]
    processing_time: str

class MaintenanceTask(BaseModel):
    task: str
    frequency: str
    category: str
    description: str
    priority: str = "medium"  # high, medium, low
    estimated_duration: str = "5 minutes"
    required_tools: str = ""
    safety_notes: str = ""

class MaintenanceResponse(BaseModel):
    success: bool
    pdf_name: str
    maintenance_tasks: List[MaintenanceTask]
    processing_time: str

class SafetyInfo(BaseModel):
    name: str
    about_reaction: str
    causes: str
    how_to_avoid: str
    safety_info: str

class SafetyResponse(BaseModel):
    success: bool
    pdf_name: str
    safety_information: List[SafetyInfo]
    processing_time: str

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    processing_time: Optional[str] = None

class ChunkData(BaseModel):
    heading: str
    text: str
    images: List[str]  # Image paths from MinerU output
    tables: List[str]

class PDFDeleteResponse(BaseModel):
    success: bool
    message: str
    pdf_name: str