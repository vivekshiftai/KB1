# PDF Intelligence Platform - API Query Schema

This document provides detailed information about all API endpoints, their request/response schemas, and usage examples.

## Table of Contents
- [Core Operations](#core-operations)
- [Generation Endpoints](#generation-endpoints)
- [Utility Endpoints](#utility-endpoints)
- [Error Responses](#error-responses)

---

## Core Operations

### 1. Upload PDF
**Endpoint:** `POST /upload/pdf`

**Description:** Upload and process PDF files with MinerU extraction

**Request:**
```http
POST /upload/pdf
Content-Type: multipart/form-data

file: [PDF File]
```

**Response:**
```json
{
  "success": true,
  "message": "PDF processed and stored successfully",
  "pdf_name": "manual.pdf",
  "chunks_processed": 15,
  "processing_time": "45.23s",
  "collection_name": "pdf_manual"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/upload/pdf" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@manual.pdf"
```

---

### 2. Query PDF Content
**Endpoint:** `POST /query`

**Description:** Query PDF content with intelligent responses and context

**Request Schema:**
```json
{
  "pdf_name": "manual.pdf",
  "query": "How to install the device?",
  "top_k": 5
}
```

**Request Fields:**
- `pdf_name` (string, required): Name of the PDF file to query
- `query` (string, required): The question or query text
- `top_k` (integer, optional, default: 5): Number of top results to return

**Response Schema:**
```json
{
  "success": true,
  "message": "Query processed successfully",
  "response": "To install the device, follow these steps...",
  "chunks_used": ["Installation Guide", "Setup Instructions"],
  "images": ["image1.png", "image2.png"],
  "tables": ["table1.html", "table2.html"],
  "processing_time": "2.45s"
}
```

**Response Fields:**
- `success` (boolean): Whether the query was successful
- `message` (string): Status message
- `response` (string): AI-generated answer based on PDF content
- `chunks_used` (array): List of content sections used for the response
- `images` (array): List of relevant images found in the content
- `tables` (array): List of relevant tables found in the content
- `processing_time` (string): Time taken to process the query

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "pdf_name": "manual.pdf",
       "query": "How to install the device?",
       "top_k": 5
     }'
```

---

### 3. List PDFs
**Endpoint:** `GET /pdfs`

**Description:** List all processed PDFs with pagination

**Query Parameters:**
- `page` (integer, optional, default: 1): Page number
- `limit` (integer, optional, default: 10, max: 100): Items per page

**Response Schema:**
```json
{
  "success": true,
  "pdfs": [
    {
      "collection_name": "pdf_manual",
      "pdf_name": "manual.pdf",
      "created_at": "2024-01-15T10:30:00",
      "chunk_count": 15
    }
  ],
  "total_count": 1
}
```

**Response Fields:**
- `success` (boolean): Whether the request was successful
- `pdfs` (array): List of PDF information objects
  - `collection_name` (string): Internal collection name
  - `pdf_name` (string): Original PDF filename
  - `created_at` (string): ISO timestamp of when PDF was processed
  - `chunk_count` (integer): Number of content chunks
- `total_count` (integer): Total number of PDFs

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/pdfs?page=1&limit=10"
```

---

## Generation Endpoints

### 4. Generate IoT Rules
**Endpoint:** `POST /generate-rules/{pdf_name}`

**Description:** Generate IoT monitoring rules from PDF content

**Path Parameters:**
- `pdf_name` (string, required): Name of the PDF file

**Response Schema:**
```json
{
  "success": true,
  "pdf_name": "manual.pdf",
  "rules": [
    {
      "condition": "Temperature > 30°C",
      "action": "Send notification to operator",
      "category": "temperature_monitoring",
      "priority": "medium"
    }
  ],
  "processing_time": "8.92s"
}
```

**Response Fields:**
- `success` (boolean): Whether generation was successful
- `pdf_name` (string): Name of the PDF processed
- `rules` (array): List of generated rules
  - `condition` (string): The monitoring condition
  - `action` (string): The action to take
  - `category` (string): Rule category
  - `priority` (string): Rule priority (low/medium/high)
- `processing_time` (string): Time taken to generate rules

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/generate-rules/manual.pdf"
```

---

### 5. Generate Maintenance Schedule
**Endpoint:** `POST /generate-maintenance/{pdf_name}`

**Description:** Extract maintenance schedules and tasks from PDF content

**Path Parameters:**
- `pdf_name` (string, required): Name of the PDF file

**Response Schema:**
```json
{
  "success": true,
  "pdf_name": "manual.pdf",
  "maintenance_tasks": [
    {
      "task": "Check oil levels",
      "frequency": "daily",
      "category": "lubrication",
      "description": "Visual inspection of oil levels in main reservoir"
    }
  ],
  "processing_time": "6.34s"
}
```

**Response Fields:**
- `success` (boolean): Whether generation was successful
- `pdf_name` (string): Name of the PDF processed
- `maintenance_tasks` (array): List of maintenance tasks
  - `task` (string): Task description
  - `frequency` (string): How often to perform (daily/weekly/monthly/annually)
  - `category` (string): Task category
  - `description` (string): Detailed task description
- `processing_time` (string): Time taken to generate tasks

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/generate-maintenance/manual.pdf"
```

---

### 6. Generate Safety Information
**Endpoint:** `POST /generate-safety/{pdf_name}`

**Description:** Generate safety information and procedures from PDF content

**Path Parameters:**
- `pdf_name` (string, required): Name of the PDF file

**Response Schema:**
```json
{
  "success": true,
  "pdf_name": "manual.pdf",
  "safety_information": [
    {
      "type": "warning",
      "title": "High Temperature Warning",
      "description": "Equipment surfaces may reach temperatures exceeding 80°C during operation",
      "category": "thermal_hazard"
    }
  ],
  "processing_time": "5.67s"
}
```

**Response Fields:**
- `success` (boolean): Whether generation was successful
- `pdf_name` (string): Name of the PDF processed
- `safety_information` (array): List of safety items
  - `type` (string): Type of safety information (warning/procedure/emergency)
  - `title` (string): Safety item title
  - `description` (string): Detailed safety description
  - `category` (string): Safety category
- `processing_time` (string): Time taken to generate safety information

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/generate-safety/manual.pdf"
```

---

## Utility Endpoints

### 7. Health Check
**Endpoint:** `GET /health`

**Description:** Global health check endpoint

**Response Schema:**
```json
{
  "status": "healthy",
  "service": "PDF Intelligence Platform",
  "version": "1.0.0",
  "components": {
    "upload": "healthy",
    "query": "healthy",
    "pdfs": "healthy",
    "rules": "healthy",
    "maintenance": "healthy",
    "safety": "healthy"
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/health"
```

---

### 8. Service Information
**Endpoint:** `GET /`

**Description:** Service information and available endpoints

**Response Schema:**
```json
{
  "service": "PDF Intelligence Platform",
  "version": "1.0.0",
  "status": "operational",
  "endpoints": {
    "upload": "/upload/pdf",
    "query": "/query",
    "list_pdfs": "/pdfs",
    "generate_rules": "/generate-rules/{pdf_name}",
    "generate_maintenance": "/generate-maintenance/{pdf_name}",
    "generate_safety": "/generate-safety/{pdf_name}",
    "health": "/health",
    "docs": "/docs"
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/"
```

---

## Error Responses

### Standard Error Format
All endpoints return errors in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common HTTP Status Codes
- `400 Bad Request`: Invalid request data or parameters
- `404 Not Found`: PDF not found or endpoint doesn't exist
- `500 Internal Server Error`: Server-side processing error

### Example Error Responses

**PDF Not Found:**
```json
{
  "detail": "PDF 'manual.pdf' not found. Please upload the PDF first."
}
```

**Invalid File Type:**
```json
{
  "detail": "Only PDF files are allowed"
}
```

**File Too Large:**
```json
{
  "detail": "File too large. Maximum size: 52428800 bytes"
}
```

**Processing Error:**
```json
{
  "detail": "PDF processing failed: MinerU processing failed"
}
```

---

## Data Types

### Timestamps
All timestamps are in ISO 8601 format: `YYYY-MM-DDTHH:MM:SS`

### File Sizes
File sizes are specified in bytes

### Processing Times
Processing times are formatted as strings with units: `"45.23s"`, `"2.45s"`

### Collections
PDF collections are automatically named using the pattern: `pdf_<sanitized_filename>`

---

## Rate Limits
Currently, no rate limiting is implemented. Consider implementing rate limiting for production deployments.

## Authentication
Currently, no authentication is required. Consider implementing API key authentication for production deployments.

## CORS
CORS is enabled for all origins (`*`). Configure appropriately for production environments.
