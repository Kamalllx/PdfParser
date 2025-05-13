"""
PDF Processor for NCERT chatbot

This module extracts text and images from PDF files, 
processes them using Groq vision model, and stores them in Pinecone.
"""

import os
import logging
import tempfile
import base64
from typing import List, Dict, Any, Optional, Tuple
import uuid
import fitz  # PyMuPDF
from PIL import Image
import io
import hashlib
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# In pdf_processor.py
from vector_store import PineconeVectorStore

logger = logging.getLogger(__name__)

class NCERTPDFProcessor:
    """Processes PDF files for the NCERT chatbot."""
    
    def __init__(
        self,
        vector_store: PineconeVectorStore,
        groq_api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vision_model: str = "llava-13b"
    ):
        """
        Initialize the PDF processor.
        
        Args:
            vector_store: The vector store to use
            groq_api_key: The Groq API key
            chunk_size: The size of text chunks
            chunk_overlap: The overlap between chunks
            vision_model: The vision model to use for image understanding
        """
        self.vector_store = vector_store
        self.groq_api_key = groq_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vision_model = vision_model
        # Use HuggingFace embeddings instead of GroqEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create a text splitter for chunking the PDF content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    def process_pdf(self, pdf_path: str) -> str:
        """
        Process a PDF file and store it in the vector database.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            collection_name: Name of the collection created in Pinecone
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Create a unique collection name based on the PDF filename
        pdf_name = os.path.basename(pdf_path)
        collection_name = f"ncert-{Path(pdf_name).stem.lower().replace(' ', '-')}"
        
        # Extract text and images from the PDF
        documents, images = self._extract_pdf_content(pdf_path)
        
        if not documents and not images:
            logger.warning(f"No content extracted from {pdf_path}")
            return ""
        
        # Process images with vision model if any
        if images:
            image_docs = self._process_images_with_vision(images, pdf_path)
            documents.extend(image_docs)
        
        # Split the documents into chunks
        chunks = self._split_documents(documents)
        
        # Store the chunks in the vector database
        self.vector_store.add_documents(chunks, collection_name=collection_name)
        
        logger.info(f"Successfully processed {pdf_path} into collection {collection_name}")
        return collection_name
    
    def _extract_pdf_content(self, pdf_path: str) -> Tuple[List[Document], List[Dict]]:
        """
        Extract text and images from a PDF.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Tuple containing a list of Document objects and a list of image dictionaries
        """
        documents = []
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            pdf_name = os.path.basename(pdf_path)
            
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path,
                                "pdf_name": pdf_name,
                                "page": page_num + 1,
                                "type": "text"
                            }
                        )
                    )
                
                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert image bytes to base64 for storage
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Generate a unique image ID using the content hash
                    image_hash = hashlib.md5(image_bytes).hexdigest()
                    image_id = f"img-{image_hash}"
                    
                    # Store image information
                    images.append({
                        "image_id": image_id,
                        "page_num": page_num + 1,
                        "img_index": img_index,
                        "source": pdf_path,
                        "pdf_name": pdf_name,
                        "image_base64": image_base64
                    })
            
            return documents, images
        
        except Exception as e:
            logger.error(f"Error extracting content from PDF {pdf_path}: {str(e)}")
            return [], []
    
    def _process_images_with_vision(self, images: List[Dict], pdf_path: str) -> List[Document]:
        """
        Process images with the Groq vision model.
        
        Args:
            images: List of image dictionaries
            pdf_path: Path to the PDF file
        
        Returns:
            List of Document objects with image descriptions
        """
        import requests
        
        image_documents = []
        
        for img_data in images:
            try:
                image_base64 = img_data["image_base64"]
                page_num = img_data["page_num"]
                
                # Create a temporary file for the image
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                    img_bytes = base64.b64decode(image_base64)
                    temp_img.write(img_bytes)
                    temp_img_path = temp_img.name
                
                # Call Groq API for image understanding
                # Note: Using requests directly here as langchain_groq doesn't have vision API integration yet
                headers = {
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                }
                
                # Convert image to base64 for API request
                with open(temp_img_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                
                payload = {
                    "model": self.vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image in detail, focusing on any scientific diagrams, chemical structures, physical phenomena, mathematical equations or educational content shown."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ]
                        }
                    ],
                    "max_tokens": 1000
                }
                
                # Make API call to Groq Vision model
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                # Extract the image description
                image_description = response.json()["choices"][0]["message"]["content"]
                
                # Create a document with the image description
                image_documents.append(
                    Document(
                        page_content=f"[IMAGE DESCRIPTION: {image_description}]",
                        metadata={
                            "source": pdf_path,
                            "pdf_name": img_data["pdf_name"],
                            "page": page_num,
                            "type": "image",
                            "image_id": img_data["image_id"]
                        }
                    )
                )
                
                # Clean up the temporary file
                os.unlink(temp_img_path)
                
            except Exception as e:
                logger.error(f"Error processing image on page {page_num}: {str(e)}")
        
        return image_documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
        
        Returns:
            List of split Document objects
        """
        all_chunks = []
        
        for doc in documents:
            # Only split text documents, not image descriptions
            if doc.metadata.get("type") == "text":
                chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            else:
                # Keep image documents as is
                all_chunks.append(doc)
        
        return all_chunks