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
from pathlib import Path
from typing import List, Dict, Any
from models.schemas import ChunkData, ImageData
from config import settings

# Try to import PIL for image compression
try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
    logger.info("PIL (Pillow) available for image compression")
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL (Pillow) not available - image compression will be disabled")

logger = logging.getLogger(__name__)

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
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
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
            
            # Process the image (compress if PIL available and enabled, otherwise use original)
            if PIL_AVAILABLE and self.compression_enabled:
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
                # No compression available
                with open(full_image_path, "rb") as f:
                    processed_data = f.read()
                mime_type, _ = mimetypes.guess_type(str(full_image_path))
                if not mime_type:
                    mime_type = "application/octet-stream"
                processed_size = len(processed_data)
                logger.info(f"Image processed (no compression): {image_path} - {processed_size:,} bytes")
            
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
        """Compress image and return (compressed_data, mime_type)"""
        if not PIL_AVAILABLE or not self.compression_enabled:
            # Fallback: read original file without compression
            if not PIL_AVAILABLE:
                logger.warning(f"PIL not available, using original image: {image_path}")
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
            
            # Open image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (for JPEG compatibility)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparent images
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if image is too large
                if max(img.size) > self.max_dimension:
                    img = ImageOps.contain(img, (self.max_dimension, self.max_dimension), method=Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to {img.size}")
                
                # Compress image
                output_buffer = io.BytesIO()
                
                # Use JPEG for better compression, PNG for transparency if needed
                if img.mode == 'RGB':
                    img.save(output_buffer, format='JPEG', quality=self.quality, optimize=True)
                    mime_type = 'image/jpeg'
                else:
                    img.save(output_buffer, format='PNG', optimize=True)
                    mime_type = 'image/png'
                
                compressed_data = output_buffer.getvalue()
                original_size = image_path.stat().st_size
                compression_ratio = (original_size - len(compressed_data)) / original_size * 100
                
                logger.info(f"Image compressed: {original_size:,} bytes → {len(compressed_data):,} bytes ({compression_ratio:.1f}% reduction)")
                
                return compressed_data, mime_type
                
        except Exception as e:
            logger.error(f"Error compressing image {image_path}: {str(e)}")
            raise e
    
    def _should_embed_image(self, image_path: str, base_dir: Path) -> bool:
        """Check if image should be embedded (now always True since we compress)"""
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
            
            file_size = full_image_path.stat().st_size
            
            if PIL_AVAILABLE and self.compression_enabled:
                # With compression, we can embed all images
                logger.info(f"Image {image_path} ({file_size:,} bytes) will be compressed and embedded")
                return True
            else:
                # Without compression, check size limit
                if file_size <= self.max_image_size:
                    logger.info(f"Image {image_path} ({file_size:,} bytes) will be embedded (no compression)")
                    return True
                else:
                    logger.info(f"Image {image_path} ({file_size:,} bytes) is too large for embedding without compression")
                    return False
            
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