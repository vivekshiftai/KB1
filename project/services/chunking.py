"""
Chunking Service
Handles content chunking for vector storage

Version: 0.1
"""

import re
import json
import logging
import base64
import mimetypes
import io
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from models.schemas import ChunkData, ImageData
from config import settings

logger = logging.getLogger(__name__)

# Try to import OpenCV for image compression
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV available for image compression")
except ImportError as e:
    logger.warning(f"OpenCV not available - image compression will be disabled: {e}")
except Exception as e:
    logger.warning(f"Error importing OpenCV - image compression will be disabled: {e}")

class MarkdownChunker:
    def __init__(self):
        self.heading_pattern = re.compile(r"^#+\s+.*")  # Any heading with # followed by text
        self.image_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
        self.table_pattern = re.compile(r"(<table>.*?</table>)", re.DOTALL | re.IGNORECASE)
        
        # Image compression settings from config
        self.max_image_size = settings.image_max_size
        self.max_dimension = settings.image_max_dimension
        self.quality = settings.image_quality
        self.compression_enabled = settings.image_compression_enabled
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    
    def _load_image_data(self, image_path: str, base_dir: Path) -> ImageData:
        """Load image file, compress it, and convert to base64"""
        try:
            # Resolve relative path from markdown file
            if not Path(image_path).is_absolute():
                full_image_path = base_dir / image_path
            else:
                full_image_path = Path(image_path)
            
            if not full_image_path.exists():
                logger.warning(f"Image file not found: {full_image_path}")
                return None
            
            original_size = full_image_path.stat().st_size
            logger.info(f"Processing image: {image_path} (original size: {original_size:,} bytes)")
            
            # Check if it's a supported image format
            file_extension = full_image_path.suffix.lower()
            if file_extension not in self.supported_formats:
                logger.warning(f"Unsupported image format: {file_extension} for {image_path}")
                return None
            
            # Check if compression is needed based on size
            needs_compression = original_size > self.max_image_size
            
            if needs_compression and CV2_AVAILABLE and self.compression_enabled:
                try:
                    processed_data, mime_type = self._compress_image(full_image_path)
                    processed_size = len(processed_data)
                    
                    # Calculate compression ratio
                    compression_ratio = (original_size - processed_size) / original_size * 100
                    logger.info(f"Image compressed: {image_path} - {original_size:,} → {processed_size:,} bytes ({compression_ratio:.1f}% reduction)")
                    
                except Exception as e:
                    logger.error(f"Compression failed for {image_path}, using original: {str(e)}")
                    # Fallback to original file
                    with open(full_image_path, "rb") as f:
                        processed_data = f.read()
                    mime_type, _ = mimetypes.guess_type(str(full_image_path))
                    if not mime_type:
                        mime_type = "application/octet-stream"
                    processed_size = len(processed_data)
            else:
                # Use original image (no compression needed or available)
                with open(full_image_path, "rb") as f:
                    processed_data = f.read()
                mime_type, _ = mimetypes.guess_type(str(full_image_path))
                if not mime_type:
                    mime_type = "application/octet-stream"
                processed_size = len(processed_data)
                
                if needs_compression and (not CV2_AVAILABLE or not self.compression_enabled):
                    logger.warning(f"Image {image_path} ({original_size:,} bytes) exceeds limit but compression not available")
                else:
                    logger.info(f"Image {image_path} ({processed_size:,} bytes) within limit, using original")
            
            # Convert to base64
            base64_data = base64.b64encode(processed_data).decode('utf-8')
            
            return ImageData(
                filename=full_image_path.name,
                data=base64_data,
                mime_type=mime_type,
                size=processed_size  # Store processed size
            )
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def _compress_image(self, image_path: Path) -> tuple[bytes, str]:
        """Compress image using OpenCV and return (compressed_data, mime_type)"""
        if not CV2_AVAILABLE or not self.compression_enabled:
            # Fallback: read original file without compression
            if not CV2_AVAILABLE:
                logger.warning(f"OpenCV not available, using original image: {image_path}")
            else:
                logger.info(f"Image compression disabled, using original image: {image_path}")
            with open(image_path, "rb") as f:
                original_data = f.read()
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type:
                mime_type = "application/octet-stream"
            return original_data, mime_type
        
        try:
            logger.info(f"Compressing image: {image_path}")
            
            # Read image with OpenCV
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Get image dimensions
            height, width = img.shape[:2]
            logger.info(f"Original image size: {width}x{height}")
            
            # Resize if image is too large (maintain aspect ratio)
            if max(width, height) > self.max_dimension:
                scale = self.max_dimension / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                logger.info(f"Resized image to {new_width}x{new_height}")
            
            # Determine output format based on file extension
            file_extension = image_path.suffix.lower()
            if file_extension in ['.jpg', '.jpeg']:
                # JPEG compression
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                mime_type = 'image/jpeg'
            else:
                # PNG compression (for other formats)
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]  # PNG compression level 0-9
                mime_type = 'image/png'
            
            # Compress image
            success, compressed_data = cv2.imencode(file_extension, img, encode_params)
            if not success:
                raise ValueError("Failed to compress image")
            
            compressed_bytes = compressed_data.tobytes()
            original_size = image_path.stat().st_size
            compression_ratio = (original_size - len(compressed_bytes)) / original_size * 100
            
            logger.info(f"Image compressed: {original_size:,} bytes → {len(compressed_bytes):,} bytes ({compression_ratio:.1f}% reduction)")
            
            return compressed_bytes, mime_type
                
        except Exception as e:
            logger.error(f"Error compressing image {image_path}: {str(e)}")
            raise e
    
    def _should_embed_image(self, image_path: str, base_dir: Path) -> bool:
        """Check if image should be embedded (always True since we handle compression based on size)"""
        try:
            if not Path(image_path).is_absolute():
                full_image_path = base_dir / image_path
            else:
                full_image_path = Path(image_path)
            
            if not full_image_path.exists():
                logger.warning(f"Image file not found for size check: {full_image_path}")
                return False
            
            # Check if it's a supported image format
            file_extension = full_image_path.suffix.lower()
            if file_extension not in self.supported_formats:
                logger.warning(f"Unsupported image format: {file_extension}")
                return False
            
            # Now we can embed all images since we'll compress if needed
            file_size = full_image_path.stat().st_size
            logger.info(f"Image {image_path} ({file_size:,} bytes) will be embedded")
            return True
            
        except Exception as e:
            logger.error(f"Error checking image {image_path}: {str(e)}")
            return False
    
    def chunk_markdown_with_headings(self, md_path: str) -> List[ChunkData]:
        """
        Chunk markdown content based on headings with exact implementation
        """
        logger.info(f"Processing markdown file: {md_path}")
        
        if not Path(md_path).exists():
            logger.error(f"Markdown file not found: {md_path}")
            return []
        
        chunks = []
        heading = None
        content_lines = []
        images = []
        tables = []
        
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Log file content for debugging
            logger.info(f"Markdown file has {len(lines)} lines")
            if lines:
                logger.info(f"First few lines: {lines[:3]}")
                logger.info(f"File size: {len(''.join(lines))} characters")
                logger.info(f"All content preview: {''.join(lines)[:500]}...")
            else:
                logger.warning("Markdown file is empty!")
            
            # If no headings found, create a single chunk with all content
            has_headings = any(line.strip().startswith("#") for line in lines)
            
            if not has_headings:
                logger.info("No headings found, creating single chunk with all content")
                all_content = "".join(lines)
                if all_content.strip():
                    # Extract images from all content
                    all_images = self.image_pattern.findall(all_content)
                    
                    # Process and embed images
                    embedded_images = []
                    base_dir = Path(md_path).parent
                    
                    for image_path in all_images:
                        if self._should_embed_image(image_path, base_dir):
                            image_data = self._load_image_data(image_path, base_dir)
                            if image_data:
                                embedded_images.append(image_data)
                                logger.info(f"Embedded image in single chunk: {image_path}")
                    
                    chunks.append(ChunkData(
                        heading="PDF Content",
                        text=all_content.strip(),
                        image_paths=all_images,
                        embedded_images=embedded_images,
                        tables=[]
                    ))
                logger.info(f"Created {len(chunks)} chunks from {md_path}")
                return chunks
            
            for line in lines:
                stripped = line.strip()
                
                # Detect heading
                if stripped.startswith("#"):
                    # Save previous chunk if exists
                    if heading is not None:
                        chunk_text = "".join(content_lines).strip()
                        if chunk_text or images or tables:
                            # Process and embed images for this chunk
                            embedded_images = []
                            base_dir = Path(md_path).parent
                            
                            for image_path in images:
                                if self._should_embed_image(image_path, base_dir):
                                    image_data = self._load_image_data(image_path, base_dir)
                                    if image_data:
                                        embedded_images.append(image_data)
                                        logger.info(f"Embedded image in chunk '{heading.strip()}': {image_path}")
                            
                            chunks.append(ChunkData(
                                heading=heading.strip(),
                                text=chunk_text,
                                image_paths=images.copy(),
                                embedded_images=embedded_images,
                                tables=tables.copy()
                            ))
                    
                    # Start new chunk
                    heading = line.strip()
                    content_lines.clear()
                    images.clear()
                    tables.clear()
                    continue
                
                # Extract images
                img_matches = self.image_pattern.findall(line)
                if img_matches:
                    images.extend(img_matches)
                
                # Extract tables
                table_matches = self.table_pattern.findall(line)
                if table_matches:
                    tables.extend(table_matches)
                
                # Add to content
                if heading is not None:
                    content_lines.append(line)
            
            # Save final chunk
            if heading is not None:
                chunk_text = "".join(content_lines).strip()
                if chunk_text or images or tables:
                    # Process and embed images for final chunk
                    embedded_images = []
                    base_dir = Path(md_path).parent
                    
                    for image_path in images:
                        if self._should_embed_image(image_path, base_dir):
                            image_data = self._load_image_data(image_path, base_dir)
                            if image_data:
                                embedded_images.append(image_data)
                                logger.info(f"Embedded image in final chunk '{heading.strip()}': {image_path}")
                    
                    chunks.append(ChunkData(
                        heading=heading.strip(),
                        text=chunk_text,
                        image_paths=images.copy(),
                        embedded_images=embedded_images,
                        tables=tables.copy()
                    ))
            
            logger.info(f"Created {len(chunks)} chunks from {md_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking markdown file {md_path}: {str(e)}")
            return []
    
    def process_directory(self, output_dir: str) -> List[ChunkData]:
        """
        Process all markdown files in output directory
        """
        logger.info(f"Processing directory: {output_dir}")
        all_chunks = []
        
        md_files = list(Path(output_dir).glob("**/*.md"))
        logger.info(f"Found {len(md_files)} markdown files")
        
        for md_file in md_files:
            chunks = self.chunk_markdown_with_headings(str(md_file))
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks