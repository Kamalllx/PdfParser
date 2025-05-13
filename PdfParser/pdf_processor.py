"""
PDF Processor Module
Handles extraction of text and images from PDFs and prepares them for vector storage
"""

import os
import fitz  # PyMuPDF
import hashlib
import tempfile
import logging
from PIL import Image
import io
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, db_handler):
        """
        Initialize the PDF processor
        
        Args:
            db_handler: MongoDB handler for storing processed documents
        """
        self.db_handler = db_handler
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len
        )
    
    def process_pdf(self, pdf_path: str, force_reindex: bool = False) -> bool:
        """
        Process a PDF file, extract text and images, and store in database
        
        Args:
            pdf_path: Path to the PDF file
            force_reindex: Whether to reindex even if already indexed
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            # Check if PDF already indexed and not forcing reindex
            pdf_hash = self._calculate_file_hash(pdf_path)
            if not force_reindex and self.db_handler.is_pdf_indexed(pdf_hash):
                logger.info(f"PDF already indexed: {pdf_path}")
                return True
            
            logger.info(f"Processing PDF: {pdf_path}")
            pdf_title = os.path.basename(pdf_path).replace('.pdf', '')
            
            # Process PDF
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = {
                "title": pdf_title,
                "file_path": pdf_path,
                "file_hash": pdf_hash,
                "page_count": len(doc)
            }
            
            # Create PDF collection in database
            self.db_handler.create_pdf_collection(pdf_hash, metadata)
            
            # Process each page
            total_extracted_text = ""
            
            for page_num, page in enumerate(doc):
                # Extract text
                page_text = page.get_text()
                total_extracted_text += f"\nPage {page_num + 1}:\n{page_text}"
                
                # Extract images
                self._extract_and_store_images(page, page_num, pdf_hash)
            
            # Split text into chunks and store
            chunks = self._split_text(total_extracted_text, metadata, pdf_path)
            self.db_handler.add_pdf_chunks(pdf_hash, chunks)
            
            logger.info(f"Successfully processed PDF: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return False
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        hash_obj = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def _extract_and_store_images(self, page, page_num: int, pdf_hash: str) -> None:
        """
        Extract images from a page and store them in the database
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            pdf_hash: Hash of the PDF file
        """
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Get image format
                image_ext = base_image["ext"]
                
                # Create a PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert image to base64 for storage
                buffered = io.BytesIO()
                image.save(buffered, format=image_ext.upper())
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Create a caption/description for the image
                img_caption = f"Image {img_index+1} on page {page_num+1}"
                
                # Get image location on page
                for img_rect in page.get_image_rects(xref):
                    location = {
                        "page": page_num,
                        "rect": [img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1]
                    }
                    break
                else:
                    location = {"page": page_num}
                
                # Store image in database
                image_data = {
                    "base64": img_base64,
                    "format": image_ext,
                    "caption": img_caption,
                    "location": location,
                    "width": image.width,
                    "height": image.height
                }
                
                self.db_handler.add_pdf_image(pdf_hash, image_data)
                
            except Exception as e:
                logger.warning(f"Error processing image {img_index} on page {page_num}: {str(e)}")
    
    def _split_text(self, text: str, metadata: Dict[str, Any], pdf_path: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks for storage and retrieval
        
        Args:
            text: Full text to split
            metadata: PDF metadata
            pdf_path: Path to the PDF file
            
        Returns:
            List of text chunks with metadata
        """
        docs = [Document(page_content=text, metadata=metadata)]
        chunks = self.text_splitter.split_documents(docs)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "chunk_id": i
            }
            processed_chunks.append(chunk_data)
        
        return processed_chunks