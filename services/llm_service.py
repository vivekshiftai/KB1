"""
LLM Service - Clean Version
Handles interactions with Azure AI for intelligent responses

Version: 0.2 - Cleaned up for LangGraph integration
"""

import json
import logging
import re
import tiktoken
import threading
import asyncio
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from config import settings
from utils.image_processor import ImageProcessor

logger = logging.getLogger(__name__)

# Try to import PIL for image processing
PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
    logger.info("PIL available for image processing")
except ImportError:
    logger.warning("PIL not available - image processing will be limited")

class LLMService:
    def __init__(self):
        # Azure AI Configuration
        self.endpoint = "https://chgai.services.ai.azure.com/models"
        self.api_version = "2024-05-01-preview"
        
        # Initialize image processor for pre-labeling
        self.image_processor = ImageProcessor()
        
        # Model configurations for different use cases
        self.models = {
            "maintenance": {
                "name": settings.maintenance_model_name,
                "endpoint": settings.maintenance_model_endpoint or self.endpoint
            },
            "rules": {
                "name": settings.rules_model_name,
                "endpoint": settings.rules_model_endpoint or self.endpoint
            },
            "safety": {
                "name": settings.safety_model_name,
                "endpoint": settings.safety_model_endpoint or self.endpoint
            },
            "query": {
                "name": settings.query_model_name,
                "endpoint": settings.query_model_endpoint or self.endpoint
            },
            "analysis": {
                "name": settings.analysis_model_name,
                "endpoint": settings.analysis_model_endpoint or self.endpoint
            }
        }
        
        # Validate Azure AI key
        if not settings.azure_openai_key:
            logger.error("Azure AI key not configured. Please set AZURE_OPENAI_KEY in your environment.")
            raise ValueError("Azure AI key is required but not configured")
        
        try:
            # Initialize Azure AI client
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(settings.azure_openai_key),
                api_version=self.api_version
            )
            logger.info("Azure AI client initialized successfully")
            logger.info(f"Available models: {list(self.models.keys())}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure AI client: {str(e)}")
            raise e
        
        # Token limits
        self.max_tokens = 8192
        self.max_completion_tokens = 1500
        self.max_context_tokens = self.max_tokens - self.max_completion_tokens
        
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception as e:
            logger.warning(f"Failed to load GPT-4 encoding, using default: {str(e)}")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"LLM Service initialized - max_tokens: {self.max_tokens}")
        
        # Thread-safe locks for concurrent operations
        self._request_locks = {}  # Per-model request locks
        self._global_lock = threading.Lock()
        # Semaphore to limit concurrent API requests (prevent rate limiting)
        self._api_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
    
    def _get_request_lock(self, model_name: str) -> threading.Lock:
        """Get or create a lock for a specific model to prevent rate limiting"""
        with self._global_lock:
            if model_name not in self._request_locks:
                self._request_locks[model_name] = threading.Lock()
            return self._request_locks[model_name]
    
    def _get_model_config(self, use_case: str) -> dict:
        """Get model configuration for specific use case"""
        if use_case not in self.models:
            logger.warning(f"Unknown use case '{use_case}', using default query model")
            use_case = "query"
        
        model_config = self.models[use_case]
        logger.info(f"Using {use_case} model: {model_config['name']} at {model_config['endpoint']}")
        return model_config
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def _clean_response_text(self, response_text: str) -> str:
        """Remove unwanted metadata sections from response text"""
        try:
            import re
            
            # Patterns to remove chunks used and visual references sections
            patterns_to_remove = [
                # Bold sections
                r'\n\n\*\*Chunks Used:\*\*.*?(?=\n\n|\Z)',
                r'\n\n\*\*Visual References:\*\*.*?(?=\n\n|\Z)',
                r'\n\n\*\*Suggested Images:\*\*.*?(?=\n\n|\Z)',
                r'\n\n\*\*References:\*\*.*?(?=\n\n|\Z)',
                r'\n\n\*\*Reference Images:\*\*.*?(?=\n\n|\Z)',
                r'\n\n\*\*Sources:\*\*.*?(?=\n\n|\Z)',
                
                # Regular sections
                r'\nChunks Used:.*?(?=\n\n|\Z)',
                r'\nVisual References:.*?(?=\n\n|\Z)',
                r'\nSuggested Images:.*?(?=\n\n|\Z)',
                r'\nReferences:.*?(?=\n\n|\Z)',
                r'\nReference Images:.*?(?=\n\n|\Z)',
                r'\nSources:.*?(?=\n\n|\Z)',
                
                # With lists
                r'Chunks Used:\s*\n.*?(?=\n\n|\Z)',
                r'Visual References:\s*\n.*?(?=\n\n|\Z)',
                r'Suggested Images:\s*\n.*?(?=\n\n|\Z)',
                r'References:\s*\n.*?(?=\n\n|\Z)',
                r'Reference Images:\s*\n.*?(?=\n\n|\Z)',
                r'Sources:\s*\n.*?(?=\n\n|\Z)',
                
                # Standalone at end
                r'\*\*Chunks Used:\*\*\s*$',
                r'\*\*Visual References:\*\*\s*$',
                r'\*\*Suggested Images:\*\*\s*$',
                r'\*\*Reference Images:\*\*\s*$',
                r'\*\*References:\*\*\s*$',
                r'Chunks Used:\s*$',
                r'Visual References:\s*$',
                r'Suggested Images:\s*$',
                r'Reference Images:\s*$',
                r'References:\s*$',
                
                # Image reference patterns that might appear
                r'Image \d+:.*?(?=\n\n|\Z)',
                r'- Image \d+:.*?(?=\n\n|\Z)',
                
                # Incomplete sections at end
                r'\*\*Reference Images:\*\*\s*-\s*$',
                r'Reference Images:\s*-\s*$',
                
                # List-style metadata sections
                r'Suggested Images\s*\n\n-.*?(?=\n\n|\Z)',
                r'Chunks Used\s*\n\n-.*?(?=\n\n|\Z)',
                r'\*\*Suggested Images\*\*\s*\n\n-.*?(?=\n\n|\Z)',
                r'\*\*Chunks Used\*\*\s*\n\n-.*?(?=\n\n|\Z)',
            ]
            
            cleaned_text = response_text
            for pattern in patterns_to_remove:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
            
            # Clean up any extra whitespace
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Replace multiple newlines with double newlines
            cleaned_text = cleaned_text.strip()
            
            if cleaned_text != response_text:
                logger.info("Removed unwanted metadata sections from response text")
            
            return cleaned_text
            
        except Exception as e:
            logger.warning(f"Error cleaning response text: {str(e)}")
            return response_text
    
    def _remove_hashtags(self, response_text: str) -> str:
        """Remove markdown hashtags and convert to bold formatting"""
        try:
            import re
            
            # Convert hashtag headers to bold formatting
            # Handle different levels of hashtags
            patterns = [
                (r'^####\s*(.+)$', r'**\1**'),  # #### Header -> **Header**
                (r'^###\s*(.+)$', r'**\1**'),   # ### Header -> **Header**
                (r'^##\s*(.+)$', r'**\1**'),    # ## Header -> **Header**
                (r'^#\s*(.+)$', r'**\1**'),     # # Header -> **Header**
                (r'\n####\s*(.+)$', r'\n**\1**'),  # Mid-text #### Header
                (r'\n###\s*(.+)$', r'\n**\1**'),   # Mid-text ### Header
                (r'\n##\s*(.+)$', r'\n**\1**'),    # Mid-text ## Header
                (r'\n#\s*(.+)$', r'\n**\1**'),     # Mid-text # Header
            ]
            
            cleaned_text = response_text
            for pattern, replacement in patterns:
                cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.MULTILINE)
            
            if cleaned_text != response_text:
                logger.info("Converted hashtag headers to bold formatting")
            
            return cleaned_text
            
        except Exception as e:
            logger.warning(f"Error removing hashtags: {str(e)}")
            return response_text

    def _extract_images_from_response_text(self, response_text: str, image_reference_mapping: Dict[str, str]) -> List[str]:
        """Extract only images that are actually referenced in the response text"""
        try:
            import re
            
            # Find all image references in the response text (e.g., "image 1", "image 2")
            image_references = re.findall(r'\bimage\s+(\d+)\b', response_text, re.IGNORECASE)
            
            # Convert to the expected format and validate against mapping
            images_actually_used = []
            for img_num in image_references:
                img_reference = f"image {img_num}"
                if img_reference in image_reference_mapping:
                    if img_reference not in images_actually_used:  # Avoid duplicates
                        images_actually_used.append(img_reference)
                        logger.info(f"✅ Found reference to '{img_reference}' in response text")
                else:
                    logger.warning(f"⚠️ Response references '{img_reference}' but it's not in mapping")
            
            logger.info(f"🎯 STRICT FILTERING: Found {len(images_actually_used)} images referenced in response text")
            logger.info(f"Images actually referenced: {images_actually_used}")
            
            return images_actually_used
            
        except Exception as e:
            logger.error(f"Error extracting images from response text: {str(e)}")
            return []
    
    def _validate_suggested_images(self, suggested_images: List[str], available_images: List[str]) -> List[str]:
        """Validate suggested images against available images and log any issues"""
        try:
            validated_images = []
            
            for suggested_image in suggested_images:
                if suggested_image in available_images:
                    validated_images.append(suggested_image)
                    logger.info(f"Valid image suggestion: {suggested_image}")
                else:
                    logger.warning(f"Invalid image suggestion '{suggested_image}' - not found in available images: {available_images}")
                    # Try to find similar image names (case-insensitive partial matching)
                    for available_image in available_images:
                        if (suggested_image.lower() in available_image.lower() or 
                            available_image.lower() in suggested_image.lower()):
                            validated_images.append(available_image)
                            logger.info(f"Found similar image '{available_image}' for suggestion '{suggested_image}'")
                            break
            
            if len(validated_images) != len(suggested_images):
                logger.warning(f"Validated {len(validated_images)} out of {len(suggested_images)} suggested images")
            
            return validated_images
            
        except Exception as e:
            logger.error(f"Error validating suggested images: {str(e)}")
            return []
    
    def _convert_to_image_references(self, suggested_filenames: List[str], image_reference_mapping: Dict[str, str]) -> List[str]:
        """Convert suggested image filenames to numbered references (image 1, image 2, etc.)"""
        try:
            # Create reverse mapping: filename -> "image X"
            filename_to_reference = {filename: reference for reference, filename in image_reference_mapping.items()}
            
            suggested_references = []
            for filename in suggested_filenames:
                if filename in filename_to_reference:
                    suggested_references.append(filename_to_reference[filename])
                    logger.info(f"Converted '{filename}' to '{filename_to_reference[filename]}'")
                else:
                    # Try partial matching for similar filenames
                    found_match = False
                    for mapped_filename, reference in filename_to_reference.items():
                        if (filename.lower() in mapped_filename.lower() or 
                            mapped_filename.lower() in filename.lower()):
                            suggested_references.append(reference)
                            logger.info(f"Fuzzy matched '{filename}' to '{reference}' (actual: '{mapped_filename}')")
                            found_match = True
                            break
                    
                    if not found_match:
                        logger.warning(f"Could not convert filename '{filename}' to image reference")
            
            return suggested_references
            
        except Exception as e:
            logger.error(f"Error converting filenames to image references: {str(e)}")
            return []
    
    def _needs_table_data(self, query: str) -> bool:
        """Determine if the query requires table data - Always return True to include all data"""
        # Always include tables and full data for comprehensive responses
        return True

    def _filter_chunk_content(self, chunk: Dict[str, Any], needs_tables: bool) -> Dict[str, Any]:
        """Extract full chunk content including text, images, and tables for multi-modal processing"""
        try:
            if "metadata" in chunk and "document" in chunk:
                heading = chunk.get("metadata", {}).get("heading", "")
                content = chunk.get("document", "")
                embedded_images = chunk.get("embedded_images", [])
                tables = chunk.get("tables", [])
            else:
                heading = chunk.get("heading", "")
                content = chunk.get("text", chunk.get("content", ""))
                embedded_images = chunk.get("embedded_images", [])
                tables = chunk.get("tables", [])
            
            if not content:
                return {"text": "", "images": [], "tables": []}
            
            # Process images for multi-modal LLM
            processed_images = []
            for i, img in enumerate(embedded_images):
                try:
                    if PIL_AVAILABLE:
                        # Convert base64 to JPG format for LLM
                        # Decode base64 image
                        image_data = base64.b64decode(img.data)
                        
                        # Convert to PIL Image
                        pil_image = Image.open(BytesIO(image_data))
                        
                        # Convert to RGB if necessary (for JPG compatibility)
                        if pil_image.mode in ('RGBA', 'LA', 'P'):
                            pil_image = pil_image.convert('RGB')
                        
                        # Convert to JPG format
                        jpg_buffer = BytesIO()
                        pil_image.save(jpg_buffer, format='JPEG', quality=85)
                        jpg_data = jpg_buffer.getvalue()
                        
                        # Encode back to base64 for LLM
                        jpg_base64 = base64.b64encode(jpg_data).decode('utf-8')
                        
                        processed_images.append({
                            "image_number": i + 1,
                            "filename": img.filename,
                            "data": jpg_base64,
                            "mime_type": "image/jpeg",
                            "description": f"Image {i + 1}: {img.filename}"
                        })
                        
                        logger.info(f"Processed image {i + 1}: {img.filename} -> JPG format for LLM")
                    else:
                        # PIL not available, use original image data
                        processed_images.append({
                            "image_number": i + 1,
                            "filename": img.filename,
                            "data": img.data,  # Keep original base64 data
                            "mime_type": img.mime_type,
                            "description": f"Image {i + 1}: {img.filename}"
                        })
                        
                        logger.info(f"Using original image {i + 1}: {img.filename} (PIL not available)")
                    
                except Exception as e:
                    logger.warning(f"Error processing image {i + 1}: {str(e)}")
                    continue
            
            # Always return full content including tables, specifications, and all data
            # Now includes processed images for multi-modal LLM
            return {
                "text": content,
                "images": processed_images,
                "tables": tables,
                "heading": heading
            }
        except Exception as e:
            logger.warning(f"Error extracting chunk content: {str(e)}")
            return {"text": "", "images": [], "tables": []}

    async def assess_information_sufficiency(self, chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Assess if we have sufficient information to answer the query completely"""
        logger.info(f"Assessing information sufficiency for query: {query}")
        
        if not chunks:
            return {
                "has_sufficient_info": False,
                "missing_information": ["No information available"],
                "additional_queries_needed": [query],
                "confidence_score": 0.0,
                "reasoning": "No chunks available for analysis"
            }
        
        # Prepare context from chunks
        context_parts = []
        for chunk in chunks:
            try:
                if "metadata" in chunk and "document" in chunk:
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                else:
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                
                if content:
                    context_parts.append(f"**{heading}**\n{content[:500]}...")  # Truncate for assessment
            except Exception as e:
                logger.warning(f"Error processing chunk for assessment: {str(e)}")
                continue
        
        context = "\n\n".join(context_parts)
        
        assessment_prompt = f"""Analyze if you have sufficient information to provide a complete, self-contained answer without referencing other sections.

Available Information:
{context}

User Question: {query}

Critical assessment criteria:
- Can you provide ALL necessary steps and details directly in your answer?
- Do you have complete procedures without needing to say "as described in section X"?
- Are there any preparation steps or prerequisites that would require additional information?
- Can you include all specific values, measurements, and technical details needed?
- Would you need to reference other sections or chapters to complete the answer?

If you would need to reference other sections (like "see section 4.1") or if preparation steps lack specific details, mark as insufficient information.

Respond with JSON:
{{
    "has_sufficient_info": true/false,
    "missing_information": ["specific missing details"],
    "additional_queries_needed": ["search terms for missing info"],
    "confidence_score": 0.0-1.0,
    "reasoning": "explanation focusing on completeness"
}}"""
        
        try:
            # Get analysis model configuration
            model_config = self._get_model_config("analysis")
            
            # Use semaphore to limit concurrent requests
            async with self._api_semaphore:
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=assessment_prompt),
                        UserMessage(content="Assess the information sufficiency.")
                    ],
                    max_tokens=1000,
                    temperature=0.1,
                    top_p=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_config["name"]
                )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Assessment response: {raw_response[:200]}...")
            
            # Parse JSON response
            try:
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = raw_response[json_start:json_end]
                    
                    # Clean control characters that can cause JSON parsing errors
                    import re
                    # Remove or replace problematic control characters
                    json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_str)
                    # Fix common JSON issues - preserve structure for nested objects
                    # Only remove carriage returns, preserve newlines and tabs for proper JSON structure
                    json_str = json_str.replace('\r', '')
                    
                    assessment = json.loads(json_str)
                    
                    # Validate required fields
                    required_fields = ["has_sufficient_info", "missing_information", "additional_queries_needed", "confidence_score", "reasoning"]
                    for field in required_fields:
                        if field not in assessment:
                            logger.warning(f"Missing field {field} in assessment response")
                    
                    logger.info(f"Information assessment: sufficient={assessment.get('has_sufficient_info')}, confidence={assessment.get('confidence_score')}")
                    return assessment
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse assessment JSON: {str(e)}")
                logger.error(f"Raw response: {raw_response}")
                # Fallback assessment
                return {
                    "has_sufficient_info": False,
                    "missing_information": ["Unable to assess information"],
                    "additional_queries_needed": [query],
                    "confidence_score": 0.0,
                    "reasoning": f"JSON parsing error: {str(e)}"
                }
            
        except Exception as e:
            logger.error(f"Error in information assessment: {str(e)}")
            return {
                "has_sufficient_info": False,
                "missing_information": ["Assessment failed"],
                "additional_queries_needed": [query],
                "confidence_score": 0.0,
                "reasoning": f"Assessment error: {str(e)}"
            }

    async def query_with_context(self, chunks: List[Dict[str, Any]], query: str, query_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query with context chunks using structured JSON response"""
        logger.info(f"Processing query with {len(chunks)} context chunks")
        
        # If no chunks available, provide a helpful response
        if not chunks:
            logger.warning("No chunks available, providing general response")
            return {
                "response": f"I don't have access to specific documentation about '{query}'. Please ensure the document is properly uploaded and processed.",
                "chunks_used": []
            }
        
        # Determine if query needs table data
        needs_tables = self._needs_table_data(query)
        logger.info(f"Query analysis - Tables needed: {needs_tables}")
        
        # Prepare context from chunks with associated images
        context_parts = []
        chunk_headings = []
        chunk_data_with_images = []
        all_available_images = []  # Collect all image names for LLM suggestions
        
        logger.info(f"Starting to process {len(chunks)} chunks for context preparation")
        
        for chunk in chunks:
            try:
                # Filter content based on table needs - now returns dict with text, images, tables
                content_data = self._filter_chunk_content(chunk, needs_tables)
                
                if content_data and content_data.get("text", "").strip():
                    heading = content_data.get("heading", "")
                    text_content = content_data.get("text", "")
                    images = content_data.get("images", [])
                    tables = content_data.get("tables", [])
                    
                    logger.info(f"Processing chunk '{heading}': {len(text_content)} chars text, {len(images)} images, {len(tables)} tables")
                    
                    # Store chunk data with its associated images
                    chunk_data_with_images.append({
                        "heading": heading,
                        "text": text_content,
                        "images": images,
                        "tables": tables
                    })
                    
                    # Collect all available image names (will be converted to numbered names later)
                    for img in images:
                        if img['filename'] not in all_available_images:
                            all_available_images.append(img['filename'])
                            logger.info(f"Added image '{img['filename']}' to available images list")
                    
                    # Add text content to context
                    context_parts.append(f"**{heading}**\n{text_content}")
                    chunk_headings.append(heading)
                    
                    # Add tables if present
                    if tables:
                        for table in tables:
                            context_parts.append(f"**Table in {heading}:**\n{table}")
                else:
                    logger.warning(f"Skipping chunk with no content: {chunk.get('metadata', {}).get('heading', 'Unknown')}")
            except Exception as e:
                chunk_heading = chunk.get('metadata', {}).get('heading', 'Unknown') if isinstance(chunk, dict) else 'Unknown'
                logger.error(f"Error processing chunk '{chunk_heading}': {str(e)}")
                continue
        
        if not context_parts:
            logger.error("No valid context could be extracted from chunks")
            logger.error(f"Processed {len(chunks)} chunks, found {len(chunk_data_with_images)} valid chunks with content")
            return {
                "response": f"I couldn't extract meaningful content from the provided chunks for '{query}'.",
                "chunks_used": []
            }
        
        context = "\n\n".join(context_parts)
        
        logger.info(f"Context preparation complete: {len(context_parts)} context sections, {len(all_available_images)} total images")
        logger.info(f"All available images: {all_available_images}")
        
        # STEP 1: PRE-LABEL ALL IMAGES BEFORE SENDING TO LLM
        logger.info("🏷️ PRE-LABELING IMAGES BEFORE SENDING TO LLM...")
        
        # Create image reference mapping first
        image_reference_mapping = {}
        image_counter = 1
        for img_filename in all_available_images:
            image_reference_mapping[f"image {image_counter}"] = img_filename
            image_counter += 1
        
        logger.info(f"Created image reference mapping: {image_reference_mapping}")
        
        # Label all images in chunk_data_with_images
        for chunk_data in chunk_data_with_images:
            if chunk_data.get("images"):
                labeled_images = []
                for img in chunk_data["images"]:
                    try:
                        # Find the numbered reference for this image
                        img_filename = img.get('filename', 'unknown.jpg')
                        numbered_reference = None
                        for ref, filename in image_reference_mapping.items():
                            if filename == img_filename:
                                numbered_reference = ref
                                break
                        
                        if numbered_reference:
                            # Convert dict to ImageData for labeling
                            from models.schemas import ImageData
                            
                            # Safely extract image data with defaults
                            img_data = img.get('data', '')
                            img_mime_type = img.get('mime_type', 'image/jpeg')
                            
                            # Calculate size safely
                            img_size = img.get('size')
                            if img_size is None and img_data:
                                try:
                                    img_size = len(base64.b64decode(img_data))
                                except Exception as e:
                                    logger.warning(f"Could not decode image data for size calculation: {e}")
                                    img_size = 0
                            elif img_size is None:
                                img_size = 0
                            
                            image_data = ImageData(
                                filename=img_filename,
                                data=img_data,
                                mime_type=img_mime_type,
                                size=img_size
                            )
                            
                            # Label the image
                            labeled_image_data = self.image_processor.add_label_to_image(image_data, numbered_reference)
                            
                            # Convert back to dict format for chunk storage
                            labeled_img_dict = {
                                'filename': labeled_image_data.filename,
                                'data': labeled_image_data.data,
                                'mime_type': labeled_image_data.mime_type,
                                'size': labeled_image_data.size,
                                'original_filename': img_filename,  # Keep track of original
                                'image_number': image_counter,
                                'description': f"{numbered_reference}: {img_filename}"
                            }
                            labeled_images.append(labeled_img_dict)
                            logger.info(f"✅ Pre-labeled {img_filename} as '{numbered_reference}'")
                        else:
                            # Keep original if no mapping found
                            labeled_images.append(img)
                            logger.warning(f"No reference mapping found for {img_filename}")
                    
                    except Exception as e:
                        logger.error(f"Error pre-labeling image {img.get('filename', 'unknown')}: {str(e)}")
                        # Keep original image if labeling fails
                        labeled_images.append(img)
                
                # Replace original images with labeled ones
                chunk_data["images"] = labeled_images
        
        logger.info(f"🎯 PRE-LABELING COMPLETE: All {total_images} images now have labels")
        
        # Enhanced prompt for structured JSON response
        table_info = "Full documentation data including tables, specifications, and all technical details has been included in the context for comprehensive analysis."
        
        # Create numbered image list for LLM context (now with labeled images)
        image_context = ""
        total_images = sum(len(chunk["images"]) for chunk in chunk_data_with_images)
        logger.info(f"Total labeled images across all chunks: {total_images}")
        
        if total_images > 0:
            # Create numbered list of available images for LLM
            numbered_image_list = []
            image_counter = 1
            for img_filename in all_available_images:
                numbered_entry = f"image {image_counter}: {img_filename}"
                numbered_image_list.append(numbered_entry)
                logger.info(f"Created numbered reference: {numbered_entry}")
                image_counter += 1
            
            image_list_text = ", ".join(numbered_image_list)
            logger.info(f"Complete numbered image list for LLM: {image_list_text}")
            image_context = f"""

VISUAL CONTENT: {total_images} images are provided with this documentation for visual analysis. Each image contains important visual information that complements the text.

Available images: {image_list_text}

Please examine these images carefully and use them to:
- Identify specific controls, buttons, screens, or equipment shown
- Provide visual context for procedures and instructions
- Reference specific visual elements when explaining steps
- Enhance your answer with details visible in the images
- Use natural references like "as shown in image 1", "refer to image 2", "see image 3"

Image selection guidelines:
- Suggest images that show actual procedures, controls, or equipment relevant to the question
- Avoid suggesting error code charts, warning symbols, emergency indicators, safety signs, or alert symbols unless specifically asked about errors or safety
- Do not suggest images of prohibited activity signs, hazard symbols, or emergency stop indicators for operational questions
- Focus on procedural images that help users complete the specific task (control panels, buttons, screens, equipment)
- Be highly selective - only suggest images that directly support completing the user's specific task
- For safety information or error codes, provide the details in text rather than suggesting warning symbol images"""
        
        # Add query analysis information if available
        analysis_info = ""
        if query_analysis:
            complexity = query_analysis.get("complexity", "unknown")
            question_count = query_analysis.get("question_count", 1)
            is_single = query_analysis.get("is_single_question", True)
            reasoning = query_analysis.get("reasoning", "")
            
            analysis_info = f"""
Query Analysis:
- Complexity: {complexity}
- Question Count: {question_count}
- Single Question: {is_single}
- Reasoning: {reasoning}
"""
        
        system_prompt = f"""You are a technical documentation assistant with visual analysis capabilities. Provide complete, self-contained answers using the provided documentation and images.

{table_info}{image_context}{analysis_info}

Critical response requirements:
- Provide all necessary details directly in your answer - never say "as described in section X" or "refer to chapter Y"
- Include the actual steps, procedures, and information rather than references to other sections
- If you need to mention preparation steps or prerequisites, include the specific details of what needs to be done
- Analyze both text content and images to provide comprehensive guidance
- Use images to enhance understanding and reference them naturally (e.g., "as shown in image 1")
- Suggest only images that are directly relevant to answering the user's specific question

Response quality standards:
- Every step must be actionable and complete
- Include specific details, values, and procedures from the documentation
- Combine text instructions with visual references for clarity
- Focus on providing practical, implementable guidance

Response formatting:
- Use **bold** for main headings (not hashtags like # or ##)
- Use Roman numerals for high-level sections (I., II., III., etc.)
- Use numbered steps for procedures (1., 2., 3., etc.)
- Use bullet points for sub-items (• or -)
- Add line breaks between sections
- Reference images when they support the explanation
- Avoid markdown hashtags (#, ##, ###, ####) in your response
- CRITICAL: Never include "Suggested Images", "Chunks Used", "Images Used", "Visual References", or any metadata sections in your response text
- Your response should contain ONLY the actual answer content, no metadata or reference sections

Respond in JSON format:
{{
    "response": "**Main Topic**\\n\\nI. **High Level Section**\\n1. First step with details\\n2. Second step with details\\n\\nII. **Another High Level Section**\\n1. First step\\n• Sub-detail\\n• Another sub-detail",
    "chunks_used": ["section headings you referenced"],
    "suggested_images": ["image 1", "image 2"]
}}

Important: Use **bold** for headings with Roman numerals for major sections, numbered steps for procedures. Never include "Suggested Images" or "Chunks Used" sections in your response text."""
        
        # Add query analysis guidance if available
        analysis_guidance = ""
        if query_analysis and not query_analysis.get("is_single_question", True):
            individual_questions = query_analysis.get("individual_questions", [])
            analysis_guidance = f"""

IMPORTANT: This query has been analyzed as containing multiple questions: {individual_questions}
Please ensure your response addresses all aspects of the original query comprehensively."""
        
        # Prepare image information for the prompt
        image_info = ""
        if total_images > 0:
            # Create the same numbered list for user prompt
            numbered_image_list = []
            image_counter = 1
            for img_filename in all_available_images:
                numbered_image_list.append(f"image {image_counter}: {img_filename}")
                image_counter += 1
            
            image_info = f"""

VISUAL ANALYSIS REQUIRED: {total_images} images are provided for you to analyze and incorporate into your answer.

Available images: {", ".join(numbered_image_list)}

Important: These images contain crucial visual information. Please:
- Examine each image to understand what it shows (controls, screens, equipment, procedures)
- Incorporate visual details into your answer where relevant
- Reference images naturally (e.g., "as shown in image 1", "refer to image 2", "see the control panel in image 3")
- Use the images to provide more specific and actionable guidance

Image suggestion rules:
- In the suggested_images field, include ONLY images that directly help answer this specific question
- Avoid suggesting error code tables, warning symbols, or emergency indicators unless the query is about errors or safety
- Focus on images showing procedures, controls, or equipment relevant to the user's task
- Be highly selective - suggest only images that provide practical value for this specific query
- If error codes are mentioned, include the code details in text rather than suggesting error code images"""
        
        user_prompt = f"""Documentation:
{context}

Question: {query}{analysis_guidance}{image_info}

Provide a complete, self-contained answer using the documentation and images. Include all necessary details directly in your response.

Visual Analysis Instructions:
 instructions:
- Include the actual steps and procedures in your answer, not references to sections
- If you mention preparation steps or prerequisites, provide the specific details of what needs to be done
- Analyze the images and use them to enhance your answer with visual context
- Suggest only images that are directly relevant to answering this specific question
- IMPORTANT: Do not suggest emergency symbols, alert icons, warning signs, or prohibited activity indicators for operational questions

Image analysis and suggestions:
- Examine each image to identify relevant visual elements for this specific query
- Reference images naturally when they support your explanation (e.g., "as shown in image 1")
- In the suggested_images field, include ONLY images that directly help answer the user's question
- Avoid suggesting error code images, warning symbols, emergency indicators, safety signs, or alert symbols unless the query specifically asks about errors or safety
- Do not suggest images of prohibited activity signs, hazard symbols, or emergency stop indicators for procedural questions
- Focus on images showing actual procedures, controls, equipment, buttons, screens, or operational steps mentioned in your answer
- Be highly selective - suggest only images that directly help users complete the specific task or procedure
- For error codes or safety information, include the specific codes and details directly in your text response rather than suggesting symbol or chart images
- If you need to mention error codes, write them out (e.g., "Error Code 101: Motor Overload") instead of suggesting error code chart images

Quality and formatting requirements:
- Every step must be complete and actionable
- Include specific details from the documentation
- Combine text instructions with relevant visual references
- Provide practical, implementable guidance without section references
- Use **bold** formatting for headings, not hashtags (#, ##, ###)
- Use Roman numerals (I., II., III.) for high-level sections
- Use numbered steps (1., 2., 3.) for procedures
- Use bullet points (•) for sub-items
- CRITICAL: Never include "Suggested Images", "Chunks Used", "Images Used", "Visual References", or any metadata sections in your response text
- Your response should contain ONLY the actual answer content, no metadata or reference sections
- Your response should end with actual content, not metadata sections"""
        
        try:
            # Get query model configuration
            model_config = self._get_model_config("query")
            
            # Use semaphore to limit concurrent requests
            async with self._api_semaphore:
                # Prepare multi-modal content with chunks and their associated images
                user_content = [{"type": "text", "text": user_prompt}]
                
                logger.info("Starting to prepare multi-modal content for LLM request")
                
                # Add images from each chunk with their associated text
                image_counter = 1
                images_used_for_response = []  # Track images used in response generation (as "image 1", "image 2", etc.)
                image_reference_mapping = {}  # Map "image 1" -> actual filename
                
                logger.info(f"Processing {len(chunk_data_with_images)} chunks with images for LLM request")
                
                for chunk_data in chunk_data_with_images:
                    if chunk_data["images"]:
                        logger.info(f"Adding {len(chunk_data['images'])} images from chunk '{chunk_data['heading']}'")
                        
                        # Add a separator for this chunk's images with analysis instructions
                        user_content.append({
                            "type": "text", 
                            "text": f"\n\n--- Visual Content for '{chunk_data['heading']}' ---\nPlease analyze the following images carefully. Look for controls, buttons, screens, indicators, and any visual elements that support the text content."
                        })
                        
                        # Add each image from this chunk
                        for img in chunk_data["images"]:
                            user_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{img['mime_type']};base64,{img['data']}",
                                    "detail": "high"
                                }
                            })
                            # Create numbered reference for this image
                            image_reference = f"image {image_counter}"
                            images_used_for_response.append(image_reference)
                            image_reference_mapping[image_reference] = img['filename']
                            
                            logger.info(f"Added {image_reference} to LLM request: {img['filename']} (from chunk: {chunk_data['heading']})")
                            logger.info(f"Image mapping created: {image_reference} -> {img['filename']}")
                            image_counter += 1
                
                logger.info(f"Multi-modal content preparation complete:")
                logger.info(f"- Total content items: {len(user_content)}")
                logger.info(f"- Images sent to LLM: {len(images_used_for_response)}")
                logger.info(f"- Image reference mapping: {image_reference_mapping}")
                logger.info(f"- Images used for response: {images_used_for_response}")
                
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=system_prompt),
                        UserMessage(content=user_content)
                    ],
                    max_tokens=self.max_completion_tokens,
                    temperature=0.1,
                    top_p=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_config["name"]
                )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response using {model_config['name']}: {raw_response[:200]}...")
            logger.info(f"Full raw response length: {len(raw_response)} characters")
            
            # Try to parse JSON response
            try:
                # Clean the response to extract JSON
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                
                logger.info(f"JSON extraction: start={json_start}, end={json_end}")
                
                if json_start != -1 and json_end > json_start:
                    json_str = raw_response[json_start:json_end]
                    
                    # Clean control characters that can cause JSON parsing errors
                    import re
                    # Remove or replace problematic control characters
                    json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_str)
                    # Fix common JSON issues - preserve structure for nested objects
                    # Only remove carriage returns, preserve newlines and tabs for proper JSON structure
                    json_str = json_str.replace('\r', '')
                    
                    parsed_response = json.loads(json_str)
                    logger.info(f"Successfully parsed JSON response with keys: {list(parsed_response.keys())}")
                    
                    # Handle nested response structure (LLM sometimes returns {"response": {"question": "answer"}})
                    if "response" in parsed_response and isinstance(parsed_response["response"], dict):
                        logger.warning("Detected nested response structure, converting to flat format")
                        nested_response = parsed_response["response"]
                        
                        # Convert nested Q&A format to flat response
                        if len(nested_response) == 1:
                            # Single question-answer pair
                            question, answer = next(iter(nested_response.items()))
                            if isinstance(answer, list):
                                # Answer is a list, join it
                                response_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(answer)])
                            else:
                                response_text = str(answer)
                        else:
                            # Multiple question-answer pairs
                            response_parts = []
                            for question, answer in nested_response.items():
                                response_parts.append(f"**{question}**")
                                if isinstance(answer, list):
                                    response_parts.extend([f"{i+1}. {item}" for i, item in enumerate(answer)])
                                else:
                                    response_parts.append(str(answer))
                                response_parts.append("")  # Add spacing
                            response_text = "\n".join(response_parts)
                        
                        # Create proper response structure
                        parsed_response = {
                            "response": response_text,
                            "chunks_used": parsed_response.get("chunks_used", [])
                        }
                        logger.info("Successfully converted nested response to flat format")
                    
                    # Additional validation for response field type
                    if "response" in parsed_response and not isinstance(parsed_response["response"], str):
                        logger.warning(f"Response field is not a string, type: {type(parsed_response['response'])}")
                        if isinstance(parsed_response["response"], list):
                            # Convert list to string
                            parsed_response["response"] = "\n".join([str(item) for item in parsed_response["response"]])
                            logger.info("Converted list response to string")
                        else:
                            # Convert any other type to string
                            parsed_response["response"] = str(parsed_response["response"])
                            logger.info("Converted non-string response to string")
                    
                    # Validate required fields
                    if "response" in parsed_response and "chunks_used" in parsed_response:
                        # Post-process to remove any remaining generic references
                        response_text = parsed_response["response"]
                        
                        # Extract suggested images if provided by LLM (already in numbered format)
                        suggested_images = parsed_response.get("suggested_images", [])
                        
                        # LLM should already be suggesting numbered names like "image 1", "image 2"
                        # No need to convert since LLM knows images by their numbered names
                        
                        # Ensure re module is available
                        import re
                        
                        # Check for and fix generic references
                        generic_patterns = [
                            r"as described in chapter [\d\.]+[^.]*",
                            r"refer to chapter [\d\.]+[^.]*",
                            r"see chapter [\d\.]+[^.]*",
                            r"check chapter [\d\.]+[^.]*",
                            r"as described in section [\d\.]+[^.]*",
                            r"refer to section [\d\.]+[^.]*",
                            r"see section [\d\.]+[^.]*",
                            r"check section [\d\.]+[^.]*",
                            r"described in section [\d\.]+[^.]*",
                            r"in section [\d\.]+[^.]*of the documentation",
                            r"section [\d\.]+[^.]*of the documentation"
                        ]
                        
                        for pattern in generic_patterns:
                            if re.search(pattern, response_text, re.IGNORECASE):
                                logger.warning(f"Found and removing generic reference: {pattern}")
                                # Remove the generic reference entirely
                                response_text = re.sub(pattern, "", response_text, flags=re.IGNORECASE)
                                logger.info("Generic reference removed from response")
                        
                        # Post-process to remove any unwanted metadata sections and hashtags
                        response_text = self._clean_response_text(response_text)
                        response_text = self._remove_hashtags(response_text)
                        parsed_response["response"] = response_text
                        
                        # STRICT FILTERING: Only return images actually referenced in the response text
                        images_actually_used = self._extract_images_from_response_text(response_text, image_reference_mapping)
                        
                        # Override LLM suggestions with only those actually used in response
                        parsed_response["suggested_images"] = images_actually_used
                        parsed_response["images_used_for_response"] = images_actually_used
                        parsed_response["image_reference_mapping"] = image_reference_mapping
                        parsed_response["chunk_data_with_images"] = chunk_data_with_images  # Include pre-labeled images
                        
                        logger.info(f"Successfully parsed JSON response with {len(parsed_response.get('chunks_used', []))} referenced chunks")
                        logger.info(f"Available images: {all_available_images}")
                        logger.info(f"LLM suggested images: {suggested_images}")
                        logger.info(f"Images used for response: {images_used_for_response}")
                        logger.info(f"Image reference mapping: {image_reference_mapping}")
                        logger.info(f"Chunk data with pre-labeled images: {len(chunk_data_with_images)} chunks")
                        return parsed_response
                    else:
                        logger.warning("JSON response missing required fields, using fallback")
                        raise ValueError("Missing required fields")
                else:
                    logger.warning("No valid JSON found in response, using fallback")
                    raise ValueError("No JSON found")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response: {str(e)}, using fallback extraction")
                logger.warning(f"Problematic JSON string: {json_str[:200]}...")
                
                # Try one more time with more aggressive cleaning
                try:
                    # More aggressive cleaning for stubborn control characters
                    json_str_clean = re.sub(r'[^\x20-\x7E\n\r\t]', '', json_str)
                    json_str_clean = json_str_clean.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                    # Remove extra spaces
                    json_str_clean = re.sub(r'\s+', ' ', json_str_clean)
                    
                    parsed_response = json.loads(json_str_clean)
                    if "response" in parsed_response and "chunks_used" in parsed_response:
                        # STRICT FILTERING: Only return images actually referenced in the response text
                        response_text = parsed_response.get("response", "")
                        images_actually_used = self._extract_images_from_response_text(response_text, image_reference_mapping)
                        
                        parsed_response["suggested_images"] = images_actually_used
                        parsed_response["images_used_for_response"] = images_actually_used
                        parsed_response["image_reference_mapping"] = image_reference_mapping
                        parsed_response["chunk_data_with_images"] = chunk_data_with_images  # Include pre-labeled images
                        logger.info("Successfully parsed JSON after aggressive cleaning")
                        return parsed_response
                except Exception as e2:
                    logger.warning(f"Aggressive cleaning also failed: {str(e2)}")
                
                # Final fallback: extract response and chunks using old method
                chunks_used = self._extract_referenced_sections(raw_response, chunks)
                cleaned_response = self._clean_response_text(raw_response)
                cleaned_response = self._remove_hashtags(cleaned_response)
                
                # STRICT FILTERING: Only return images actually referenced in the response text
                images_actually_used = self._extract_images_from_response_text(cleaned_response, image_reference_mapping)
                
                return {
                    "response": cleaned_response,
                    "chunks_used": chunks_used,
                    "suggested_images": images_actually_used,
                    "images_used_for_response": images_actually_used,
                    "image_reference_mapping": image_reference_mapping,
                    "chunk_data_with_images": chunk_data_with_images  # Include pre-labeled images
                }
            
        except Exception as e:
            logger.error(f"Error in LLM query: {str(e)}")
            raise e
    
    async def _get_analysis_response(self, prompt: str) -> str:
        """Get analysis response from LLM for query analysis"""
        try:
            system_prompt = """Analyze user queries and break them down into individual questions.

Respond with JSON:
{
  "is_single_question": true/false,
  "question_count": number,
  "individual_questions": ["question1", "question2"],
  "complexity": "simple/moderate/complex",
  "reasoning": "brief explanation"
}"""
            
            # Get analysis model configuration
            model_config = self._get_model_config("analysis")
            
            # Use semaphore to limit concurrent requests
            async with self._api_semaphore:
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=system_prompt),
                        UserMessage(content=prompt)
                    ],
                    max_tokens=1000,
                    temperature=0.1,
                    top_p=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_config["name"]
                )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw analysis response using {model_config['name']}: {raw_response[:200]}...")
            
            # Try to parse JSON response
            try:
                # Clean the response to extract JSON
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = raw_response[json_start:json_end]
                    
                    # Clean control characters that can cause JSON parsing errors
                    import re
                    # Remove or replace problematic control characters
                    json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_str)
                    # Fix common JSON issues - preserve structure for nested objects
                    # Only remove carriage returns, preserve newlines and tabs for proper JSON structure
                    json_str = json_str.replace('\r', '')
                    
                    parsed_response = json.loads(json_str)
                    
                    # Validate required fields
                    required_fields = ["is_single_question", "question_count", "individual_questions", "complexity", "reasoning"]
                    if all(field in parsed_response for field in required_fields):
                        logger.info("Successfully parsed analysis JSON response")
                        return json_str
                    else:
                        logger.warning("Analysis JSON response missing required fields, using fallback")
                        raise ValueError("Missing required fields")
                else:
                    logger.warning("No valid JSON found in analysis response, using fallback")
                    raise ValueError("No JSON found")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse analysis JSON response: {str(e)}, using fallback")
                # Return fallback JSON
                return '{"is_single_question": true, "question_count": 1, "individual_questions": ["Original query"], "complexity": "simple", "reasoning": "Error in analysis"}'
            
        except Exception as e:
            logger.error(f"Error getting analysis response: {str(e)}")
            # Return fallback JSON
            return '{"is_single_question": true, "question_count": 1, "individual_questions": ["Original query"], "complexity": "simple", "reasoning": "Error in analysis"}'
    
    def _extract_referenced_sections(self, answer: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract referenced sections from LLM response"""
        referenced_sections = []
        
        try:
            logger.info(f"Extracting references from answer: {answer[:200]}...")
            
            # Look for REFERENCES section in the answer
            if "REFERENCES:" in answer:
                # Extract the REFERENCES section
                ref_start = answer.find("REFERENCES:")
                ref_section = answer[ref_start:]
                logger.info(f"Found REFERENCES section: {ref_section}")
                
                # Extract section headings from the references
                lines = ref_section.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('-') or line.startswith('•'):
                        # Remove bullet points and extract heading
                        heading = line[1:].strip()
                        if heading:
                            referenced_sections.append(heading)
                            logger.info(f"Extracted bullet reference: {heading}")
                    elif ':' in line and not line.startswith('REFERENCES:'):
                        # Handle format like "Section 1: Introduction"
                        heading = line.strip()
                        if heading:
                            referenced_sections.append(heading)
                            logger.info(f"Extracted colon reference: {heading}")
            
            # If no explicit REFERENCES section, try to match headings in the answer
            if not referenced_sections:
                logger.warning("No explicit REFERENCES section found, trying text matching")
                for chunk in chunks:
                    try:
                        # Handle both possible chunk structures
                        if "metadata" in chunk and "document" in chunk:
                            # Vector DB format
                            heading = chunk.get("metadata", {}).get("heading", "")
                        else:
                            # Fallback format
                            heading = chunk.get("heading", "")
                        
                        # More flexible matching - check if heading or parts of it appear in answer
                        if heading:
                            heading_words = heading.lower().split()
                            answer_lower = answer.lower()
                            
                            # Check if any significant words from heading appear in answer
                            matches = sum(1 for word in heading_words if len(word) > 3 and word in answer_lower)
                            if matches >= min(2, len(heading_words)):  # At least 2 words or most of heading
                                referenced_sections.append(heading)
                                logger.info(f"Matched heading in text: {heading} (matches: {matches})")
                    except Exception as e:
                        logger.warning(f"Error extracting heading from chunk: {str(e)}")
                        continue
            
            # If still no references found, use all chunks (fallback)
            if not referenced_sections:
                logger.warning("No references found, using all chunks as fallback")
                for chunk in chunks:
                    try:
                        if "metadata" in chunk and "document" in chunk:
                            heading = chunk.get("metadata", {}).get("heading", "")
                        else:
                            heading = chunk.get("heading", "")
                        
                        if heading:
                            referenced_sections.append(heading)
                    except Exception as e:
                        logger.warning(f"Error getting heading from chunk: {str(e)}")
                        continue
            
            # If still no references, use chunk indices as fallback
            if not referenced_sections:
                logger.warning("No headings found, using chunk indices as fallback")
                for i, chunk in enumerate(chunks):
                    referenced_sections.append(f"Chunk {i+1}")
            
            logger.info(f"Final extracted referenced sections: {referenced_sections}")
            return referenced_sections
            
        except Exception as e:
            logger.error(f"Error extracting referenced sections: {str(e)}")
            # Fallback: return chunk indices
            fallback_sections = []
            for i, chunk in enumerate(chunks):
                fallback_sections.append(f"Chunk {i+1}")
            
            logger.info(f"Using fallback sections: {fallback_sections}")
            return fallback_sections

    async def generate_rules(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate IoT monitoring rules from chunks"""
        logger.info(f"Generating rules from {len(chunks)} chunks")
        
        if not chunks:
            return "No content available to generate rules from."
        
        # Prepare context from chunks (include full data with tables)
        context_parts = []
        for chunk in chunks:
            try:
                if "metadata" in chunk and "document" in chunk:
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                    # Extract tables from metadata if available
                    tables = chunk.get("metadata", {}).get("tables", [])
                else:
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                    tables = chunk.get("tables", [])
                
                if content:
                    # Include full content including tables, specifications, and all data
                    context_text = f"**{heading}**\n{content}"
                    
                    # Add tables to context if they exist
                    if tables:
                        context_text += f"\n\n**Tables in this section:**\n"
                        for i, table in enumerate(tables, 1):
                            context_text += f"\nTable {i}:\n{table}\n"
                    
                    context_parts.append(context_text)
            except Exception as e:
                logger.warning(f"Error processing chunk for rules: {str(e)}")
                continue
        
        if not context_parts:
            return "No valid content could be extracted to generate rules."
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are an expert in IoT monitoring and industrial automation. Generate IoT monitoring rules from technical documentation.

Please respond with a JSON object using this structure:
{
  "rules": [
    {
      "name": "Rule Name",
      "description": "Detailed description of the rule",
      "metric": "sensor_metric_name",
      "metric_value": "numerical_value_with_unit",
      "threshold": "numerical_threshold_condition",
      "consequence": "What happens when threshold is exceeded",
      "condition": "IF condition statement",
      "action": "Actions to take (SEND_ALERT, LOG_EVENT, etc.)",
      "priority": "HIGH/MEDIUM/LOW"
    }
  ]
}

IoT MONITORING REQUIREMENTS:
- metric must be a SENSOR-MEASURABLE parameter (temperature, pressure, voltage, current, speed, vibration, etc.)
- metric_value must be a specific numerical value with unit (e.g., "75°C", "1500 N", "85%", "220 V", "50 Hz")
- threshold must be numerical conditions (e.g., "> 75°C", "< 20%", "> 1500 N", "= 220 V")
- Avoid descriptive states like "hinged down", "pushed in", "equal distance" - focus on measurable parameters
- Focus on measurable physical parameters that sensors can detect
- Use actual numerical values from the documentation specifications

VALID IoT METRICS EXAMPLES:
- temperature, pressure, voltage, current, speed, vibration, frequency, power, flow_rate, level, humidity, etc.

INVALID METRICS (DO NOT USE):
- bolt_status, alignment_status, installation_status, position_status, etc.

Provide the JSON response."""
        
        user_prompt = f"""Based on the following technical documentation, generate comprehensive IoT monitoring rules:

Documentation:
{context}

Generate a comprehensive set of IoT monitoring rules that cover:
1. Equipment monitoring parameters (temperature, pressure, voltage, current, speed, vibration)
2. Safety thresholds and alerts (temperature limits, pressure limits, electrical limits)
3. Performance metrics (power consumption, efficiency, output levels)
4. Maintenance indicators (vibration levels, wear indicators, performance degradation)
5. Operational conditions (operating ranges, normal vs abnormal conditions)

CRITICAL: The documentation includes tables with specific IoT monitoring data. Use the table information extensively for:
- Exact threshold values from specification tables (temperature, pressure, voltage, current limits)
- Precise monitoring parameters from technical data tables
- Specific sensor ranges and operating conditions from equipment tables
- Performance metrics and efficiency data from measurement tables
- Safety limits and alert thresholds from safety specification tables

Pay special attention to any tables containing technical specifications, operating parameters, or monitoring data as these contain the most relevant IoT monitoring information.

IoT MONITORING REQUIREMENTS:
- Create rules for sensor-measurable parameters that IoT devices can monitor
- Extract actual numerical values from tables, specifications, and technical data
- Use specific numbers with units (e.g., "75°C", "1500 N", "85%", "220 V", "50 Hz")
- Create precise threshold conditions with real numbers from the documentation
- Focus on physical parameters that can be measured by sensors
- Focus on sensor-measurable parameters rather than mechanical states, positions, or installation status

VALID SENSOR METRICS TO USE:
- Temperature, pressure, voltage, current, speed, vibration, frequency, power, flow rate, level, humidity, torque, force, etc.

Provide the JSON object with the rules array."""
        
        try:
            # Get rules model configuration
            model_config = self._get_model_config("rules")
            
            # Use semaphore to limit concurrent requests
            async with self._api_semaphore:
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=system_prompt),
                        UserMessage(content=user_prompt)
                    ],
                    max_tokens=self.max_completion_tokens,
                    temperature=0.2,
                    top_p=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_config["name"]
                )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response for rules using {model_config['name']}: {raw_response[:200]}...")
            
            # Parse JSON response
            try:
                # Ensure re module is available
                import re
                
                # Remove markdown code blocks if present
                cleaned_response = raw_response
                if "```json" in cleaned_response:
                    cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
                if "```" in cleaned_response:
                    cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
                
                parsed_response = json.loads(cleaned_response.strip())
                return parsed_response.get("rules", [])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse rules JSON: {str(e)}")
                logger.error(f"Raw response: {raw_response}")
                return []
            
        except Exception as e:
            logger.error(f"Error generating rules: {str(e)}")
            return []

    async def generate_safety_information(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate safety information from chunks"""
        logger.info(f"Generating safety information from {len(chunks)} chunks")
        
        if not chunks:
            return "No content available to generate safety information from."
        
        # Prepare context from chunks (include full data with tables)
        context_parts = []
        for chunk in chunks:
            try:
                if "metadata" in chunk and "document" in chunk:
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                    # Extract tables from metadata if available
                    tables = chunk.get("metadata", {}).get("tables", [])
                else:
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                    tables = chunk.get("tables", [])
                
                if content:
                    # Include full content including tables, specifications, and all data
                    context_text = f"**{heading}**\n{content}"
                    
                    # Add tables to context if they exist
                    if tables:
                        context_text += f"\n\n**Tables in this section:**\n"
                        for i, table in enumerate(tables, 1):
                            context_text += f"\nTable {i}:\n{table}\n"
                    
                    context_parts.append(context_text)
            except Exception as e:
                logger.warning(f"Error processing chunk for safety: {str(e)}")
                continue
        
        if not context_parts:
            return "No valid content could be extracted to generate safety information."
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a safety expert specializing in industrial equipment and machinery safety. Generate safety information from technical documentation.

Please respond with a JSON object using this structure:
{
  "safety_precautions": [
    {
      "title": "Safety Title",
      "description": "Description of the safety concern",
      "category": "category_type",
      "severity": "HIGH/MEDIUM/LOW",
      "mitigation": "How to mitigate the risk",
      "about_reaction": "What happens if not addressed",
      "causes": "What causes this safety issue",
      "how_to_avoid": "How to avoid the issue",
      "safety_info": "Additional safety information",
      "type": "warning/caution/procedure",
      "recommended_action": "Recommended action to take"
    }
  ],
  "safety_information": [
    {
      "title": "Information Title",
      "description": "Description of safety information",
      "category": "category_type",
      "severity": "HIGH/MEDIUM/LOW",
      "mitigation": "How to mitigate the risk",
      "about_reaction": "What happens if not addressed",
      "causes": "What causes this safety issue",
      "how_to_avoid": "How to avoid the issue",
      "safety_info": "Additional safety information",
      "type": "warning/caution/procedure",
      "recommended_action": "Recommended action to take"
    }
  ]
}

Provide the JSON response."""
        
        user_prompt = f"""Based on the following technical documentation, generate comprehensive safety information:

Documentation:
{context}

Generate detailed safety information covering:
1. Safety procedures and protocols
2. Hazard identification and mitigation
3. Personal protective equipment (PPE) requirements
4. Emergency procedures
5. Safety warnings and precautions
6. Risk assessments
7. Error codes and their meanings
8. Alert conditions and responses

CRITICAL: The documentation includes tables with specific safety data. Use the table information extensively for:
- Exact error codes and their descriptions from error code tables
- Precise safety limits and thresholds (temperature, pressure, voltage, current)
- Specific precautionary measures from safety tables
- Alert conditions and their severity levels from alert tables
- PPE requirements and specifications from safety tables
- Emergency response procedures and contact information

Pay special attention to any tables labeled as "error codes", "precaution", "alert", or "safety" as these contain the most relevant safety data.

IMPORTANT: Use specific numerical values from the documentation for:
- Temperature limits (e.g., "60°C", "100°F")
- Pressure limits (e.g., "150 psi", "10 bar")
- Voltage/current limits (e.g., "220 V", "5 A")
- Distance requirements (e.g., "2 meters", "6 feet")
- Time limits and durations from safety procedures

Provide the JSON object with safety_precautions and safety_information arrays."""
        
        try:
            # Get safety model configuration
            model_config = self._get_model_config("safety")
            
            # Use semaphore to limit concurrent requests
            async with self._api_semaphore:
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=system_prompt),
                        UserMessage(content=user_prompt)
                    ],
                    max_tokens=self.max_completion_tokens,
                    temperature=0.2,
                    top_p=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_config["name"]
                )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response for safety using {model_config['name']}: {raw_response[:200]}...")
            
            # Parse JSON response
            try:
                # Ensure re module is available
                import re
                
                # Remove markdown code blocks if present
                cleaned_response = raw_response
                if "```json" in cleaned_response:
                    cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
                if "```" in cleaned_response:
                    cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
                
                parsed_response = json.loads(cleaned_response.strip())
                return {
                    "safety_precautions": parsed_response.get("safety_precautions", []),
                    "safety_information": parsed_response.get("safety_information", [])
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse safety JSON: {str(e)}")
                logger.error(f"Raw response: {raw_response}")
                return {"safety_precautions": [], "safety_information": []}
            
        except Exception as e:
            logger.error(f"Error generating safety information: {str(e)}")
            return {"safety_precautions": [], "safety_information": []}

    async def dynamic_query_processing(self, vector_db, collection_name: str, query: str, query_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dynamic information gathering flow with multi-stage processing"""
        logger.info(f"Starting dynamic query processing for: {query}")
        logger.info(f"Using collection: {collection_name}")
        
        try:
            # Stage 1: Initial query to get base chunks
            logger.info(f"Stage 1: Getting initial chunks from collection: {collection_name}")
            logger.info(f"Initial query: '{query}' with top_k=5")
            
            initial_chunks = await vector_db.query_chunks(
                collection_name=collection_name,
                query=query,
                top_k=5
            )
            
            logger.info(f"Stage 1 complete: Retrieved {len(initial_chunks)} initial chunks")
            
            if not initial_chunks:
                logger.warning("No initial chunks found")
                return {
                    "response": f"I don't have access to specific documentation about '{query}'. Please ensure the document is properly uploaded and processed.",
                    "chunks_used": [],
                    "processing_stages": ["initial_query"],
                    "confidence_score": 0.0,
                    "suggested_images": [],
                    "images_used_for_response": [],
                    "image_reference_mapping": {}
                }
            
            # Stage 2: Assess information sufficiency
            logger.info("Stage 2: Assessing information sufficiency")
            assessment = await self.assess_information_sufficiency(initial_chunks, query)
            
            all_chunks = initial_chunks.copy()
            processing_stages = ["initial_query", "assessment"]
            
            # Stage 3: Get additional information if needed
            if not assessment.get("has_sufficient_info", False):
                logger.info("Stage 3: Gathering additional information")
                logger.info(f"Using same collection for additional queries: {collection_name}")
                additional_queries = assessment.get("additional_queries_needed", [])
                
                for additional_query in additional_queries[:3]:  # Limit to 3 additional queries
                    try:
                        logger.info(f"Querying for additional info: {additional_query} in collection: {collection_name}")
                        additional_chunks = await vector_db.query_chunks(
                            collection_name=collection_name,
                            query=additional_query,
                            top_k=3
                        )
                        
                        # Add new chunks (avoid duplicates)
                        for chunk in additional_chunks:
                            chunk_id = chunk.get("metadata", {}).get("chunk_index", "")
                            if not any(existing.get("metadata", {}).get("chunk_index", "") == chunk_id for existing in all_chunks):
                                all_chunks.append(chunk)
                        
                        logger.info(f"Added {len(additional_chunks)} additional chunks for: {additional_query}")
                        
                    except Exception as e:
                        logger.warning(f"Error getting additional chunks for '{additional_query}': {str(e)}")
                        continue
                
                processing_stages.append("additional_queries")
            
            # Stage 4: Generate comprehensive response
            logger.info("Stage 4: Generating comprehensive response")
            final_response = await self.query_with_context(all_chunks, query, query_analysis)
            
            # Stage 4.5: Check if response indicates need for more information
            logger.info("Stage 4.5: Checking response completeness")
            response_text = final_response.get("response", "")
            
            # Simple check for incomplete responses that might need more information
            needs_more_info = any(phrase in response_text.lower() for phrase in [
                "need more specific information",
                "additional information is needed", 
                "more details are required",
                "insufficient information"
            ])
            
            if needs_more_info:
                logger.info("Response indicates need for more information, attempting additional queries")
                
                # Try to get more relevant chunks based on the original query
                additional_queries = [
                    f"{query} detailed steps procedures",
                    f"{query} complete instructions",
                    f"{query} specific requirements"
                ]
                
                for additional_query in additional_queries[:2]:  # Limit to 2 additional queries
                    try:
                        logger.info(f"Querying for additional info: {additional_query}")
                        additional_chunks = await vector_db.query_chunks(
                            collection_name=collection_name,
                            query=additional_query,
                            top_k=3
                        )
                        
                        # Add new chunks (avoid duplicates)
                        for chunk in additional_chunks:
                            chunk_id = chunk.get("metadata", {}).get("chunk_index", "")
                            if not any(existing.get("metadata", {}).get("chunk_index", "") == chunk_id for existing in all_chunks):
                                all_chunks.append(chunk)
                        
                        logger.info(f"Added {len(additional_chunks)} additional chunks for: {additional_query}")
                        
                    except Exception as e:
                        logger.warning(f"Error getting additional chunks for '{additional_query}': {str(e)}")
                        continue
                
                # Regenerate response with additional chunks
                logger.info("Regenerating response with additional information")
                final_response = await self.query_with_context(all_chunks, query, query_analysis)
                processing_stages.append("additional_information_gathering")
            
            # Stage 5: Evaluate response quality
            logger.info("Stage 5: Evaluating response quality")
            evaluation = await self.evaluate_response_quality(final_response, query, assessment)
            
            # Combine all information
            result = {
                **final_response,
                "processing_stages": processing_stages,
                "initial_chunks_count": len(initial_chunks),
                "total_chunks_count": len(all_chunks),
                "assessment": assessment,
                "evaluation": evaluation,
                "confidence_score": evaluation.get("confidence_score", 0.0),
                "collection_used": collection_name,  # Track which collection was used throughout
                "processed_chunks": all_chunks  # Store all chunks that were processed by LLM
            }
            
            logger.info(f"Dynamic processing completed: {len(processing_stages)} stages, {len(all_chunks)} total chunks, confidence: {result['confidence_score']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in dynamic query processing: {str(e)}")
            return {
                "response": f"Error processing query: {str(e)}",
                "chunks_used": [],
                "processing_stages": ["error"],
                "confidence_score": 0.0,
                "error": str(e),
                "suggested_images": [],
                "images_used_for_response": [],
                "image_reference_mapping": {}
            }

    async def evaluate_response_quality(self, response: Dict[str, Any], query: str, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality and completeness of the response"""
        logger.info("Evaluating response quality")
        
        try:
            response_text = response.get("response", "")
            chunks_used = response.get("chunks_used", [])
            
            # Quality metrics
            quality_metrics = {
                "response_length": len(response_text),
                "has_content": len(response_text.strip()) > 50,
                "uses_chunks": len(chunks_used) > 0,
                "addresses_query": any(word.lower() in response_text.lower() for word in query.split()),
                "has_specific_details": any(char.isdigit() for char in response_text),  # Has numbers/measurements
                "has_procedural_steps": "1." in response_text or "step" in response_text.lower(),
                "no_generic_references": not any(phrase in response_text.lower() for phrase in [
                    "refer to", "see section", "check the", "please refer", "see documentation"
                ])
            }
            
            # Calculate confidence score
            confidence_score = 0.0
            if quality_metrics["has_content"]:
                confidence_score += 0.2
            if quality_metrics["uses_chunks"]:
                confidence_score += 0.2
            if quality_metrics["addresses_query"]:
                confidence_score += 0.2
            if quality_metrics["has_specific_details"]:
                confidence_score += 0.15
            if quality_metrics["has_procedural_steps"]:
                confidence_score += 0.15
            if quality_metrics["no_generic_references"]:
                confidence_score += 0.1
            
            # Bonus for addressing assessment concerns
            if assessment.get("has_sufficient_info", False):
                confidence_score += 0.1
            
            # Cap at 1.0
            confidence_score = min(confidence_score, 1.0)
            
            evaluation = {
                "confidence_score": confidence_score,
                "quality_metrics": quality_metrics,
                "is_comprehensive": confidence_score >= 0.7,
                "needs_improvement": confidence_score < 0.5,
                "assessment_met": assessment.get("has_sufficient_info", False)
            }
            
            logger.info(f"Response evaluation: confidence={confidence_score:.2f}, comprehensive={evaluation['is_comprehensive']}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating response quality: {str(e)}")
            return {
                "confidence_score": 0.0,
                "quality_metrics": {},
                "is_comprehensive": False,
                "needs_improvement": True,
                "assessment_met": False,
                "error": str(e)
            }

    async def generate_maintenance_schedule(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate maintenance schedule from chunks"""
        logger.info(f"Generating maintenance schedule from {len(chunks)} chunks")
        
        if not chunks:
            return "No content available to generate maintenance schedule from."
        
        # Prepare context from chunks (always include tables for maintenance)
        context_parts = []
        for chunk in chunks:
            try:
                if "metadata" in chunk and "document" in chunk:
                    heading = chunk.get("metadata", {}).get("heading", "")
                    content = chunk.get("document", "")
                    # Extract tables from metadata if available
                    tables = chunk.get("metadata", {}).get("tables", [])
                else:
                    heading = chunk.get("heading", "")
                    content = chunk.get("text", chunk.get("content", ""))
                    tables = chunk.get("tables", [])
                
                if content:
                    # For maintenance, always include all content including tables
                    context_text = f"**{heading}**\n{content}"
                    
                    # Add tables to context if they exist
                    if tables:
                        context_text += f"\n\n**Tables in this section:**\n"
                        for i, table in enumerate(tables, 1):
                            context_text += f"\nTable {i}:\n{table}\n"
                    
                    context_parts.append(context_text)
            except Exception as e:
                logger.warning(f"Error processing chunk for maintenance: {str(e)}")
                continue
        
        if not context_parts:
            return "No valid content could be extracted to generate maintenance schedule."
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a maintenance expert specializing in industrial equipment and machinery maintenance. Generate maintenance schedules from technical documentation.

Please respond with a JSON object using this structure:
{
  "maintenance_tasks": [
    {
      "task": "Task description",
      "task_name": "Specific task name",
      "description": "Detailed description of the task",
      "frequency": "frequency_interval",
      "priority": "HIGH/MEDIUM/LOW",
      "estimated_duration": "duration_in_time",
      "required_tools": "List of required tools",
      "category": "PREVENTIVE/CORRECTIVE/PREDICTIVE",
      "safety_notes": "Safety considerations and notes"
    }
  ]
}

Provide the JSON response."""
        
        user_prompt = f"""Based on the following technical documentation, generate a comprehensive maintenance schedule:

Documentation:
{context}

Generate a detailed maintenance schedule covering:
1. Preventive maintenance tasks
2. Maintenance intervals and frequencies
3. Required tools and materials
4. Step-by-step maintenance procedures
5. Inspection checklists
6. Maintenance priorities

CRITICAL: The documentation includes tables with specific maintenance data. Use the table information extensively for:
- Exact frequencies from maintenance schedules (e.g., "every 30 days", "monthly", "every 1000 hours")
- Precise durations from maintenance procedures (e.g., "2 hours", "30 minutes", "4 hours")
- Specific quantities, measurements, and specifications from tables
- Tool requirements and material lists from maintenance tables
- Priority levels and categories from maintenance schedules

Pay special attention to any tables labeled as "maintenance list", "maintenance tasks", "maintenance schedules", or "maintenance procedure" as these contain the most relevant data.

Provide the JSON object with the maintenance_tasks array."""
        
        try:
            # Get maintenance model configuration
            model_config = self._get_model_config("maintenance")
            
            # Use semaphore to limit concurrent requests
            async with self._api_semaphore:
                response = self.client.complete(
                    messages=[
                        SystemMessage(content=system_prompt),
                        UserMessage(content=user_prompt)
                    ],
                    max_tokens=self.max_completion_tokens,
                    temperature=0.2,
                    top_p=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    model=model_config["name"]
                )
            
            raw_response = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM response for maintenance using {model_config['name']}: {raw_response[:200]}...")
            
            # Parse JSON response
            try:
                # Ensure re module is available
                import re
                
                # Remove markdown code blocks if present
                cleaned_response = raw_response
                if "```json" in cleaned_response:
                    cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
                if "```" in cleaned_response:
                    cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
                
                parsed_response = json.loads(cleaned_response.strip())
                return parsed_response.get("maintenance_tasks", [])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse maintenance JSON: {str(e)}")
                logger.error(f"Raw response: {raw_response}")
                return []
            
        except Exception as e:
            logger.error(f"Error generating maintenance schedule: {str(e)}")
            return []