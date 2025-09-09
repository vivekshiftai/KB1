# 🖼️ Image Flow Through the System

## Complete Image Journey from PDF to Query Response

```
┌─────────────────────────────────────────────────────────────────┐
│                        PDF UPLOAD                               │
│  User uploads PDF with images                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MINERU PROCESSING                            │
│                                                                 │
│ 1. MinerU extracts PDF content                                 │
│ 2. Converts to Markdown with image references                  │
│ 3. Saves images as separate files in output directory          │
│ 4. Creates markdown files with ![](image_path) references      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CHUNKING PROCESS                             │
│                                                                 │
│ MarkdownChunker.process_directory()                            │
│                                                                 │
│ For each markdown file:                                        │
│ 1. Parse markdown content                                       │
│ 2. Extract image references: ![](path/to/image.jpg)           │
│ 3. For each image found:                                       │
│    a. Check if should embed (always True)                      │
│    b. Load image file from disk                                │
│    c. Compress if needed (OpenCV)                              │
│    d. Convert to base64                                        │
│    e. Create ImageData object                                  │
│                                                                 │
│ ImageData {                                                     │
│   filename: "image.jpg",                                       │
│   data: "base64_encoded_data...",                              │
│   mime_type: "image/jpeg",                                     │
│   size: 12345                                                  │
│ }                                                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CHUNK CREATION                               │
│                                                                 │
│ ChunkData {                                                     │
│   heading: "Safety Requirements",                              │
│   text: "Markdown content with text...",                       │
│   image_paths: ["path/to/image.jpg"],                          │
│   embedded_images: [ImageData objects],                        │
│   tables: ["<table>...</table>"]                               │
│ }                                                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR DATABASE STORAGE                      │
│                                                                 │
│ VectorDatabase.store_chunks()                                  │
│                                                                 │
│ For each chunk:                                                │
│ 1. Create embedding from text content                          │
│ 2. Store in ChromaDB with metadata:                            │
│                                                                 │
│ ChromaDB Document {                                             │
│   id: "chunk-0",                                               │
│   document: "chunk text content",                              │
│   embedding: [0.1, 0.2, 0.3, ...],                            │
│   metadata: {                                                  │
│     "heading": "Safety Requirements",                          │
│     "image_paths": '["path/to/image.jpg"]',                    │
│     "embedded_images": '[{"filename": "image.jpg",             │
│                          "data": "base64_data...",             │
│                          "mime_type": "image/jpeg",            │
│                          "size": 12345}]',                     │
│     "tables": '["<table>...</table>"]'                         │
│   }                                                             │
│ }                                                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING                             │
│                                                                 │
│ LangGraphQueryProcessor.process_query()                        │
│                                                                 │
│ 1. Retrieve chunks from vector database                        │
│ 2. Chunks come with embedded_images already loaded             │
│ 3. LLM processes chunks (ignores images)                       │
│ 4. Response validation and retry logic                         │
│ 5. Collect images from used chunks                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    IMAGE COLLECTION                             │
│                                                                 │
│ _collect_media_from_chunks()                                   │
│                                                                 │
│ 1. Find chunks used by LLM response                            │
│ 2. Extract embedded_images from those chunks                   │
│ 3. Deduplicate by filename                                     │
│ 4. Return ImageData objects ready for response                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL RESPONSE                               │
│                                                                 │
│ QueryResponse {                                                 │
│   success: true,                                               │
│   message: "Query processed successfully",                     │
│   response: "LLM generated response...",                       │
│   chunks_used: ["Safety Requirements"],                        │
│   images: [ImageData objects with base64 data],                │
│   tables: ["<table>...</table>"],                              │
│   processing_time: "2.34s"                                     │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 🔍 Key Points About Image Flow

### **1. Image Embedding During Chunking**
- Images are **converted to base64** during the chunking process
- **Compression** is applied if images are too large
- Images are **stored directly in chunk metadata**

### **2. No External Image Queries**
- Images travel with chunks through the entire pipeline
- **No separate database queries** needed for images
- Images are **immediately available** when chunks are retrieved

### **3. Image Processing Steps**
```
PDF Image → MinerU Extraction → File System → Chunking → Base64 → Vector DB → Query Response
```

### **4. Image Data Structure**
```python
ImageData {
    filename: "safety_diagram.jpg",
    data: "iVBORw0KGgoAAAANSUhEUgAA...",  # Base64 encoded
    mime_type: "image/jpeg",
    size: 12345
}
```

### **5. Storage in Vector Database**
```python
# Stored in ChromaDB metadata
metadata = {
    "heading": "Safety Requirements",
    "embedded_images": '[{"filename": "image.jpg", "data": "base64...", "mime_type": "image/jpeg", "size": 12345}]'
}
```

## 🚀 Benefits of This Approach

1. **Fast Access**: Images are immediately available with chunks
2. **No Extra Queries**: No need to fetch images separately
3. **Consistent Context**: Images stay with their relevant text
4. **Efficient Storage**: Base64 encoding with compression
5. **Reliable**: No dependency on external file systems during queries

## 📊 Image Flow Summary

- **Upload**: PDF with images
- **Extract**: MinerU saves images as files + markdown references
- **Chunk**: Convert images to base64 and embed in chunks
- **Store**: Save chunks with embedded images in vector database
- **Query**: Retrieve chunks with images already included
- **Response**: Return images directly from chunk data

The images flow seamlessly through the entire pipeline, always staying with their relevant text content! 🎯
