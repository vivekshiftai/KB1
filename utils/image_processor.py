"""
Image Processing Utilities
Handles image processing operations including adding labels to images

Version: 0.1
"""

import base64
import logging
from io import BytesIO
from typing import Optional, Tuple
from models.schemas import ImageData

logger = logging.getLogger(__name__)

# Try to import PIL for image processing
PIL_AVAILABLE = False
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
    logger.info("PIL available for image processing")
except ImportError:
    logger.warning("PIL not available - image processing will be limited")

class ImageProcessor:
    """Utility class for image processing operations"""
    
    def __init__(self):
        self.default_font_size = 32  # Increased from 20 to 32 for better visibility
        self.label_padding = 15  # Increased padding
        self.label_height = 60  # Increased height to accommodate larger text
        self.background_color = (255, 255, 255)  # White background
        self.text_color = (0, 0, 0)  # Black text
        
    def add_label_to_image(self, image_data: ImageData, label_text: str) -> ImageData:
        """
        Add a label at the bottom of an image by extending it with white background
        
        Args:
            image_data: ImageData object containing base64 image
            label_text: Text to add as label (e.g., "Image 1")
            
        Returns:
            ImageData object with labeled image
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available - returning original image without label")
            return image_data
            
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.data)
            original_image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if original_image.mode in ('RGBA', 'LA', 'P'):
                original_image = original_image.convert('RGB')
            
            # Get original dimensions
            original_width, original_height = original_image.size
            
            # Create new image with extended height for label
            new_height = original_height + self.label_height
            labeled_image = Image.new('RGB', (original_width, new_height), self.background_color)
            
            # Paste original image at the top
            labeled_image.paste(original_image, (0, 0))
            
            # Add text label at the bottom
            draw = ImageDraw.Draw(labeled_image)
            
            # Try to use a system font, fall back to default if not available
            font = None
            font_names = [
                "arialbd.ttf",  # Arial Bold - best choice
                "arial.ttf",    # Arial Regular
                "calibrib.ttf", # Calibri Bold
                "calibri.ttf",  # Calibri Regular
                "Arial.ttf",    # Alternative Arial name
                "tahoma.ttf",   # Tahoma
                "verdana.ttf"   # Verdana
            ]
            
            for font_name in font_names:
                try:
                    font = ImageFont.truetype(font_name, self.default_font_size)
                    logger.info(f"Using font: {font_name} at size {self.default_font_size}")
                    break
                except (OSError, IOError):
                    continue
            
            if font is None:
                try:
                    # Try system default with larger size
                    font = ImageFont.load_default()
                    logger.info("Using default system font")
                except:
                    font = None
                    logger.warning("No font available, using basic text rendering")
            
            # Calculate text position (centered horizontally)
            if font:
                # Get text bounding box for centering
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # Fallback for default font - larger estimates for bigger text
                text_width = len(label_text) * 16  # Increased estimate for larger text
                text_height = 24  # Increased height estimate
            
            text_x = (original_width - text_width) // 2
            text_y = original_height + (self.label_height - text_height) // 2
            
            # Draw the text
            if font:
                draw.text((text_x, text_y), label_text, fill=self.text_color, font=font)
            else:
                draw.text((text_x, text_y), label_text, fill=self.text_color)
            
            # Convert back to base64
            output_buffer = BytesIO()
            labeled_image.save(output_buffer, format='JPEG', quality=90)
            labeled_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            
            # Create new ImageData with labeled image
            labeled_image_data = ImageData(
                filename=f"labeled_{image_data.filename}",
                data=labeled_base64,
                mime_type="image/jpeg",
                size=len(output_buffer.getvalue())
            )
            
            logger.info(f"Successfully added label '{label_text}' to image {image_data.filename}")
            logger.info(f"Original size: {original_width}x{original_height}, New size: {original_width}x{new_height}")
            
            return labeled_image_data
            
        except Exception as e:
            logger.error(f"Error adding label to image {image_data.filename}: {str(e)}")
            return image_data  # Return original image if labeling fails
    
    def add_labels_to_images(self, images: list[ImageData], image_reference_mapping: dict[str, str]) -> list[ImageData]:
        """
        Add labels to multiple images based on their reference mapping
        
        Args:
            images: List of ImageData objects
            image_reference_mapping: Mapping from "image 1" to actual filename
            
        Returns:
            List of ImageData objects with labels added
        """
        if not images or not image_reference_mapping:
            return images
            
        labeled_images = []
        
        # Create reverse mapping: filename -> "image X"
        filename_to_reference = {}
        for reference, filename in image_reference_mapping.items():
            # Handle both direct filename matches and numbered filename matches
            filename_to_reference[filename] = reference
            # Also handle the case where the filename might be "image_1.jpg" format
            numbered_filename = f"{reference.replace(' ', '_')}.jpg"
            filename_to_reference[numbered_filename] = reference
        
        logger.info(f"Processing {len(images)} images for labeling")
        logger.info(f"Filename to reference mapping: {filename_to_reference}")
        
        for image in images:
            # Find the reference name for this image
            reference_name = None
            
            # Try exact filename match first
            if image.filename in filename_to_reference:
                reference_name = filename_to_reference[image.filename]
            else:
                # Try to find partial matches
                for filename, ref in filename_to_reference.items():
                    if (image.filename in filename or 
                        filename in image.filename or
                        image.filename.replace('.jpg', '') == filename.replace('.jpg', '')):
                        reference_name = ref
                        break
            
            if reference_name:
                # Add label to the image
                labeled_image = self.add_label_to_image(image, reference_name)
                labeled_images.append(labeled_image)
                logger.info(f"Added label '{reference_name}' to {image.filename}")
            else:
                # No reference found, keep original image
                labeled_images.append(image)
                logger.warning(f"No reference found for image {image.filename}, keeping original")
        
        logger.info(f"Labeled {len([img for img in labeled_images if img.filename.startswith('labeled_')])} out of {len(images)} images")
        
        return labeled_images
    
    def resize_image_if_needed(self, image_data: ImageData, max_width: int = 800, max_height: int = 600) -> ImageData:
        """
        Resize image if it's too large, maintaining aspect ratio
        
        Args:
            image_data: ImageData object
            max_width: Maximum width in pixels
            max_height: Maximum height in pixels
            
        Returns:
            ImageData object with resized image if needed
        """
        if not PIL_AVAILABLE:
            return image_data
            
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.data)
            image = Image.open(BytesIO(image_bytes))
            
            # Get current dimensions
            width, height = image.size
            
            # Check if resizing is needed
            if width <= max_width and height <= max_height:
                return image_data  # No resizing needed
            
            # Calculate new dimensions maintaining aspect ratio
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Resize image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert back to base64
            output_buffer = BytesIO()
            resized_image.save(output_buffer, format='JPEG', quality=90)
            resized_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            
            # Create new ImageData
            resized_image_data = ImageData(
                filename=image_data.filename,
                data=resized_base64,
                mime_type="image/jpeg",
                size=len(output_buffer.getvalue())
            )
            
            logger.info(f"Resized image {image_data.filename} from {width}x{height} to {new_width}x{new_height}")
            
            return resized_image_data
            
        except Exception as e:
            logger.error(f"Error resizing image {image_data.filename}: {str(e)}")
            return image_data  # Return original if resizing fails
