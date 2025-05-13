"""
Vector store for NCERT chatbot using Pinecone.

This module manages the interaction with Pinecone vector database,
including creating indexes, adding documents, and searching.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union
import uuid

from langchain_core.documents import Document
# Fix the circular import by importing directly from the vectorstores module
from langchain_pinecone.vectorstores import PineconeVectorStore as LangchainPineconeVS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pinecone

logger = logging.getLogger(__name__)

class PineconeVectorStore:
    """Interface to Pinecone vector database for storing and retrieving document embeddings."""
    
    def __init__(
        self,
        api_key: str,
        index_name: str = "ncert-assistant",
        dimension: int = 1536,
        environment: str = "gcp-starter",
        metric: str = "cosine"
    ):
        """
        Initialize the Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            dimension: Dimension of the embeddings
            environment: Pinecone environment
            metric: Distance metric for vectors
        """
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.environment = environment
        self.metric = metric
        
        # Initialize Pinecone client
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create index if it doesn't exist
        self._create_index_if_not_exists()
        
        # Get the Pinecone index
        self.index = pinecone.Index(self.index_name)
        
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    def _create_index_if_not_exists(self) -> None:
        """Create a Pinecone index if it doesn't already exist."""
        existing_indexes = pinecone.list_indexes()
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric
            )
            
            # Wait for the index to be ready
            while not self.index_name in pinecone.list_indexes():
                logger.info(f"Waiting for index {self.index_name} to be ready...")
                time.sleep(1)
                
            logger.info(f"Created new Pinecone index: {self.index_name}")
        else:
            logger.info(f"Using existing Pinecone index: {self.index_name}")
    
    def add_documents(
        self,
        documents: List[Document],
        collection_name: str,
        embeddings_provider: Optional[Any] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            collection_name: Name of the collection/namespace
            embeddings_provider: Optional embeddings provider (defaults to GroqEmbeddings)
        """
        if not documents:
            logger.warning("No documents to add to vector store")
            return
        
        if embeddings_provider is None:
            # Use GroqEmbeddings as the default embeddings provider
            embeddings_provider = GroqEmbeddings()
        
        # Create a Langchain PineconeVectorStore instance
        langchain_vs = LangchainPineconeVS.from_existing_index(
            index_name=self.index_name,
            embedding=embeddings_provider,
            namespace=collection_name
        )
        
        # Add documents to the vector store
        logger.info(f"Adding {len(documents)} documents to collection '{collection_name}'")
        
        # Add document-specific IDs to track them better
        for i, doc in enumerate(documents):
            if 'id' not in doc.metadata:
                doc.metadata['id'] = str(uuid.uuid4())
        
        # Add the documents to the vector store
        langchain_vs.add_documents(documents)
        
        logger.info(f"Successfully added {len(documents)} documents to collection '{collection_name}'")
    
    def search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        embeddings_provider: Optional[Any] = None,
        top_k: int = 5
    ) -> List[Document]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query string
            collection_name: Optional name of the collection/namespace to search in
            embeddings_provider: Optional embeddings provider (defaults to GroqEmbeddings)
            top_k: Number of results to return
        
        Returns:
            List of matching Document objects
        """
        if embeddings_provider is None:
            # Use GroqEmbeddings as the default embeddings provider
            embeddings_provider = GroqEmbeddings()
        
        # Create a Langchain PineconeVectorStore instance
        langchain_vs = LangchainPineconeVS.from_existing_index(
            index_name=self.index_name,
            embedding=embeddings_provider,
            namespace=collection_name
        )
        
        # Perform the similarity search
        logger.info(f"Searching for query: '{query}' in {collection_name or 'all collections'}")
        results = langchain_vs.similarity_search(query, k=top_k)
        
        logger.info(f"Found {len(results)} matching documents")
        return results
    
    def search_across_collections(
        self,
        query: str,
        embeddings_provider: Optional[Any] = None,
        top_k: int = 5
    ) -> List[Document]:
        """
        Search across all collections.
        
        Args:
            query: Query string
            embeddings_provider: Optional embeddings provider (defaults to GroqEmbeddings)
            top_k: Number of results to return per collection
        
        Returns:
            List of matching Document objects from all collections
        """
        if embeddings_provider is None:
            # Use GroqEmbeddings as the default embeddings provider
            embeddings_provider = GroqEmbeddings()
        
        # Get list of all collections (namespaces)
        collections = self.list_collections()
        
        all_results = []
        for collection in collections:
            # Search in each collection
            results = self.search(
                query=query,
                collection_name=collection,
                embeddings_provider=embeddings_provider,
                top_k=top_k
            )
            all_results.extend(results)
        
        # Sort by relevance (score would be ideal but Langchain doesn't expose it directly)
        # For now, we'll return all results
        return all_results[:top_k]
    
    def list_collections(self) -> List[str]:
        """
        List all collections (namespaces) in the index.
        
        Returns:
            List of collection names
        """
        stats = self.index.describe_index_stats()
        collections = list(stats.get("namespaces", {}).keys())
        return collections
    
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection from the index.
        
        Args:
            collection_name: Name of the collection to delete
        """
        self.index.delete(namespace=collection_name, delete_all=True)
        logger.info(f"Deleted collection: {collection_name}")
    
    def has_documents(self) -> bool:
        """
        Check if the index has any documents.
        
        Returns:
            True if the index has documents, False otherwise
        """
        stats = self.index.describe_index_stats()
        total_vector_count = stats.get("total_vector_count", 0)
        return total_vector_count > 0