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

class QueryRequest(BaseModel):
    pdf_name: str = Field(..., description="Name of the PDF file to query")
    query: str = Field(..., description="Query text")
    top_k: int = Field(default=5, description="Number of top results to return")

class QueryResponse(BaseModel):
    success: bool
    message: str
    response: str
    chunks_used: List[str]
    images: List[str]
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
    condition: str
    action: str
    category: str
    priority: str

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

class MaintenanceResponse(BaseModel):
    success: bool
    pdf_name: str
    maintenance_tasks: List[MaintenanceTask]
    processing_time: str

class SafetyInfo(BaseModel):
    type: str  # warning, procedure, emergency
    title: str
    description: str
    category: str

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
    images: List[str]
    tables: List[str]