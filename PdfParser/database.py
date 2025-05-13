"""
MongoDB Database Handler
Manages interactions with MongoDB for storing and retrieving PDF content, images, and conversations
"""

import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_groq import GroqEmbeddings
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MongoDBHandler:
    def __init__(self, mongodb_uri: str):
        """
        Initialize MongoDB handler
        
        Args:
            mongodb_uri: MongoDB connection URI
        """
        self.mongodb_uri = mongodb_uri
        self.client = None
        self.db = None
        self.connect()
    
    def connect(self) -> bool:
        """
        Connect to MongoDB
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client['ncert_chatbot']
            logger.info("Connected to MongoDB")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
    
    def create_pdf_collection(self, pdf_hash: str, metadata: Dict[str, Any]) -> None:
        """
        Create collections for a PDF
        
        Args:
            pdf_hash: Hash of the PDF file
            metadata: PDF metadata
        """
        # Store PDF metadata
        self.db.pdf_metadata.update_one(
            {"file_hash": pdf_hash},
            {"$set": metadata},
            upsert=True
        )
        
        # Create chunks collection if it doesn't exist
        collection_name = f"pdf_chunks_{pdf_hash}"
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)
        
        # Create images collection if it doesn't exist
        images_collection_name = f"pdf_images_{pdf_hash}"
        if images_collection_name not in self.db.list_collection_names():
            self.db.create_collection(images_collection_name)
    
    def add_pdf_chunks(self, pdf_hash: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Add PDF text chunks to database
        
        Args:
            pdf_hash: Hash of the PDF file
            chunks: List of text chunks with content and metadata
        """
        collection_name = f"pdf_chunks_{pdf_hash}"
        collection = self.db[collection_name]
        
        # Clear existing chunks
        collection.delete_many({})
        
        # Insert new chunks
        if chunks:
            collection.insert_many(chunks)
    
    def add_pdf_image(self, pdf_hash: str, image_data: Dict[str, Any]) -> None:
        """
        Add PDF image to database
        
        Args:
            pdf_hash: Hash of the PDF file
            image_data: Image data including base64, format, caption, etc.
        """
        collection_name = f"pdf_images_{pdf_hash}"
        collection = self.db[collection_name]
        collection.insert_one(image_data)
    
    def is_pdf_indexed(self, pdf_hash: str) -> bool:
        """
        Check if a PDF is already indexed
        
        Args:
            pdf_hash: Hash of the PDF file
            
        Returns:
            bool: True if indexed, False otherwise
        """
        return self.db.pdf_metadata.find_one({"file_hash": pdf_hash}) is not None
    
    def list_indexed_pdfs(self) -> List[Dict[str, Any]]:
        """
        Get list of all indexed PDFs
        
        Returns:
            List of PDF metadata
        """
        return list(self.db.pdf_metadata.find({}, {"_id": 0}))
    
    def retrieve_relevant_chunks(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve chunks relevant to a query using vector search
        
        Args:
            query: Query string
            limit: Maximum number of chunks to retrieve
            
        Returns:
            List of relevant chunks with content and metadata
        """
        try:
            # Use Groq embeddings for the query
            embeddings = GroqEmbeddings(model_name="Llama3-8b-8192")
            
            # Search across all PDF collections
            all_results = []
            
            for metadata in self.list_indexed_pdfs():
                pdf_hash = metadata["file_hash"]
                collection_name = f"pdf_chunks_{pdf_hash}"
                
                # Create vector store for this collection
                vector_store = MongoDBAtlasVectorSearch(
                    collection=self.db[collection_name],
                    embedding=embeddings,
                    index_name=f"vector_index_{pdf_hash}",
                    text_key="content"
                )
                
                # Search for relevant chunks
                results = vector_store.similarity_search(query, k=limit)
                all_results.extend([{
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "pdf_hash": pdf_hash
                } for doc in results])
            
            # Sort results by relevance (if score available) or return as is
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {str(e)}")
            return []
    
    def retrieve_relevant_images(self, pdf_hash: str, page_numbers: List[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve images from specific pages of a PDF
        
        Args:
            pdf_hash: Hash of the PDF file
            page_numbers: List of page numbers to retrieve images from (None for all pages)
            
        Returns:
            List of image data
        """
        collection_name = f"pdf_images_{pdf_hash}"
        collection = self.db[collection_name]
        
        query = {}
        if page_numbers:
            query = {"location.page": {"$in": page_numbers}}
        
        return list(collection.find(query, {"_id": 0}))
    
    def create_conversation(self, metadata: Dict[str, Any] = None) -> str:
        """
        Create a new conversation
        
        Args:
            metadata: Conversation metadata
            
        Returns:
            str: Conversation ID
        """
        result = self.db.conversations.insert_one({
            "messages": [],
            "metadata": metadata or {}
        })
        return str(result.inserted_id)
    
    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """
        Add a message to a conversation
        
        Args:
            conversation_id: Conversation ID
            message: Message data
        """
        self.db.conversations.update_one(
            {"_id": conversation_id},
            {"$push": {"messages": message}}
        )
    
    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a conversation by ID
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation data
        """
        return self.db.conversations.find_one({"_id": conversation_id})
    
    def create_vector_index(self, pdf_hash: str) -> bool:
        """
        Create vector index for a PDF collection
        
        Args:
            pdf_hash: Hash of the PDF file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            collection_name = f"pdf_chunks_{pdf_hash}"
            
            # Create a vector index
            self.db.command({
                "createIndexes": collection_name,
                "indexes": [{
                    "name": f"vector_index_{pdf_hash}",
                    "key": {"vector": "vector"},
                    "weights": {"vector": 1},
                    "defaultLanguage": "english"
                }]
            })
            
            return True
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}")
            return False