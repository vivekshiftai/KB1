import os
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any
from huggingface_hub import snapshot_download
from config import settings
from models.schemas import ChunkData
from services.chunking import MarkdownChunker

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.models_dir = settings.models_dir
        self.chunker = MarkdownChunker()
        
    async def download_models_with_retry(self, retries: int = 3) -> str:
        """Download PDF-Extract-Kit models with retry logic"""
        logger.info("Downloading PDF-Extract-Kit models...")
        
        for attempt in range(retries):
            try:
                local_dir = snapshot_download(
                    repo_id="opendatalab/PDF-Extract-Kit-1.0",
                    local_dir=self.models_dir,
                    local_dir_use_symlinks=False
                )
                logger.info(f"Models downloaded successfully to: {local_dir}")
                return local_dir
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt == retries - 1:
                    logger.error("All download attempts failed")
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return self.models_dir
    
    def setup_mineru_config(self) -> str:
        """Setup MinerU configuration"""
        config_path = "mineru-config.json"
        
        config = {
            "device_mode": settings.device_mode,
            "models_dir": self.models_dir,
            "formula_enable": settings.formula_enable,
            "table_enable": settings.table_enable,
            "method": "auto"
        }
        
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"MinerU config created: {config_path}")
            return config_path
            
        except Exception as e:
            logger.error(f"Failed to create MinerU config: {str(e)}")
            raise e
    
    async def process_pdf_with_mineru(self, pdf_path: str, output_base: str) -> str:
        """Process PDF using MinerU CLI"""
        logger.info(f"Processing PDF with MinerU: {pdf_path}")
        
        # Ensure models are downloaded
        if not Path(self.models_dir).exists() or not list(Path(self.models_dir).iterdir()):
            await self.download_models_with_retry()
        
        # Setup configuration
        config_path = self.setup_mineru_config()
        
        # Prepare output directory
        output_dir = Path(output_base)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run MinerU command - try different module paths with verbose output
            possible_commands = [
                ["mineru", "-p", pdf_path, "-o", str(output_dir), "-m", "auto", "-c", config_path, "--verbose"],
                ["mineru", "-p", pdf_path, "-o", str(output_dir), "-m", "auto", "--verbose"],
                ["mineru", "-p", pdf_path, "-o", str(output_dir), "-m", "auto", "-v"],
                ["mineru", "-p", pdf_path, "-o", str(output_dir), "-m", "auto"],
                ["python", "-m", "mineru", "-p", pdf_path, "-o", str(output_dir), "-m", "auto"],
                ["python", "-m", "mineru.cli", "-p", pdf_path, "-o", str(output_dir), "-m", "auto"]
            ]
            
            success = False
            last_error = None
            
            for cmd in possible_commands:
                try:
                    logger.info(f"Trying MinerU command: {' '.join(cmd)}")
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=1800  # 30 minutes timeout
                    )
                    
                    if result.returncode == 0:
                        logger.info("MinerU processing completed successfully")
                        success = True
                        break
                    else:
                        logger.warning(f"Command failed with return code {result.returncode}: {' '.join(cmd)}")
                        logger.warning(f"STDERR: {result.stderr}")
                        last_error = result.stderr
                        
                except FileNotFoundError:
                    logger.warning(f"Command not found: {' '.join(cmd)}")
                    last_error = f"Command not found: {' '.join(cmd)}"
                    continue
                except subprocess.TimeoutExpired:
                    logger.error("MinerU processing timed out")
                    raise Exception("PDF processing timed out")
                except Exception as e:
                    logger.warning(f"Command failed: {' '.join(cmd)} - {str(e)}")
                    last_error = str(e)
                    continue
            
            if not success:
                logger.warning("MinerU failed, falling back to PyMuPDF for basic text extraction")
                # Fallback to PyMuPDF
                return await self.process_pdf_with_pymupdf(pdf_path, output_dir)
            
            return str(output_dir)
        finally:
            # Cleanup config file
            if Path(config_path).exists():
                Path(config_path).unlink()
    
    async def process_pdf_with_pymupdf(self, pdf_path: str, output_dir: Path) -> str:
        """Process PDF using PyMuPDF for basic text extraction"""
        logger.info(f"Processing PDF with PyMuPDF: {pdf_path}")
        
        try:
            import fitz  # PyMuPDF
            
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Extract text from all pages
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
            
            doc.close()
            
            # Save as markdown
            output_file = output_dir / "extracted_text.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# PDF Text Extraction\n\n")
                f.write(f"Source: {Path(pdf_path).name}\n\n")
                f.write(full_text)
            
            logger.info(f"PyMuPDF processing completed. Text saved to: {output_file}")
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"PyMuPDF processing failed: {str(e)}")
            raise Exception(f"PyMuPDF processing failed: {str(e)}")
    
    async def process_pdf(self, pdf_path: str, output_base: str) -> List[ChunkData]:
        """
        Complete PDF processing pipeline
        """
        logger.info(f"Starting complete PDF processing for: {pdf_path}")
        
        try:
            # Process with MinerU
            output_dir = await self.process_pdf_with_mineru(pdf_path, output_base)
            
            # Wait for processing to complete and files to be written
            time.sleep(2)
            
            # Check if output files exist
            output_path = Path(output_dir)
            md_files = list(output_path.glob("**/*.md"))
            
            if not md_files:
                logger.warning(f"No markdown files found in {output_dir}")
                # Try alternative output structure
                pdf_name = Path(pdf_path).stem
                alt_output_dir = output_path / pdf_name
                md_files = list(alt_output_dir.glob("**/*.md")) if alt_output_dir.exists() else []
            
            if not md_files:
                raise Exception("No markdown files generated by MinerU")
            
            logger.info(f"Found {len(md_files)} markdown files")
            
            # Chunk the processed content
            chunks = self.chunker.process_directory(str(output_dir))
            
            if not chunks:
                logger.warning("No chunks generated from markdown files")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in complete PDF processing: {str(e)}")
            raise e