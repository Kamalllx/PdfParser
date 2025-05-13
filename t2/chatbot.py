"""
Chatbot for NCERT Study Assistant

This module provides a chatbot interface using Groq's LLM
to answer questions about NCERT textbooks using vector retrieval.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from vector_store import PineconeVectorStore
from conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

# System prompt template
SYSTEM_PROMPT = """You are the NCERT Study Assistant, a specialized educational AI helper focused on helping students understand concepts from NCERT (National Council of Educational Research and Training) textbooks for classes 11 and 12, particularly in Physics and Chemistry.

Your purpose is to explain complex scientific concepts clearly and accurately, helping students with their studies. You have access to information from NCERT textbooks, including both text and diagrams.

When answering questions:
1. Focus on explaining the core concepts in a way that's easy to understand
2. When relevant, refer to diagrams or illustrations from the book to aid understanding
3. Use examples to illustrate complex ideas
4. Break down complex processes into simpler steps
5. Highlight key formulas and equations when appropriate
6. Connect concepts to real-world applications
7. Use simple language while maintaining scientific accuracy

If you don't know the answer or if the information isn't in the provided context, say so clearly. Don't make up information. Always provide explanations that are scientifically accurate and aligned with NCERT textbooks.

The following context information is from NCERT textbooks:

{context}

Remember to address the student's question directly, focusing on the specific topic they are asking about.
"""

class GroqChatbot:
    """Chatbot interface using Groq LLM."""
    
    def __init__(
        self,
        vector_store: PineconeVectorStore,
        conversation_manager: ConversationManager,
        groq_api_key: str,
        model: str = "llama3-70b-8192",
        temperature: float = 0.2,
        top_k: int = 5
    ):
        """
        Initialize the Groq chatbot.
        
        Args:
            vector_store: Vector store for retrieving context
            conversation_manager: Manager for conversation history
            groq_api_key: Groq API key
            model: Groq model to use
            temperature: Temperature for generation
            top_k: Number of documents to retrieve for context
        """
        self.vector_store = vector_store
        self.conversation_manager = conversation_manager
        self.groq_api_key = groq_api_key
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        
        # Initialize embeddings using HuggingFace
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize the LLM
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model,
            temperature=temperature
        )
        
        logger.info(f"Initialized Groq chatbot with model: {model}")
    
    def _retrieve_context(self, query: str) -> str:
        """
        Retrieve relevant context for the query.
        
        Args:
            query: User query
        
        Returns:
            String containing relevant context
        """
        # Get conversation history for better context
        history_context = self.conversation_manager.get_recent_history()
        enhanced_query = query
        
        # If we have history, enhance the query with it
        if history_context:
            # Combine the most recent exchanges with the current query for better context
            enhanced_query = f"{history_context}\n\nCurrent question: {query}"
        
        # Search across all collections
        logger.info(f"Retrieving context for query: '{query}'")
        documents = self.vector_store.search_across_collections(
            query=enhanced_query,
            embeddings_provider=self.embeddings,
            top_k=self.top_k
        )
        
        if not documents:
            logger.warning(f"No relevant documents found for query: '{query}'")
            return "No relevant information found in the NCERT textbooks for this query."
        
        # Format the documents into a context string
        context_parts = []
        for i, doc in enumerate(documents):
            # Format the metadata for better context
            source_info = f"From: {doc.metadata.get('pdf_name', 'Unknown source')}"
            page_info = f"Page: {doc.metadata.get('page', 'Unknown page')}"
            content_type = doc.metadata.get('type', 'text')
            
            # Add the document content with its metadata
            context_parts.append(
                f"--- Document {i+1} ({content_type}) ---\n"
                f"{source_info} | {page_info}\n\n"
                f"{doc.page_content}\n"
            )
        
        return "\n".join(context_parts)
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            user_message: User's message
        
        Returns:
            Chatbot's response
        """
        try:
            # Retrieve relevant context
            context = self._retrieve_context(user_message)
            
            # Create the system message with context
            system_message = SystemMessage(content=SYSTEM_PROMPT.format(context=context))
            
            # Get conversation history
            conversation_history = self.conversation_manager.get_messages()
            
            # Add the system message at the beginning
            all_messages = [system_message] + conversation_history
            
            # Add the new user message
            all_messages.append(HumanMessage(content=user_message))
            
            # Generate the response
            response = self.llm.invoke(all_messages)
            
            # Update conversation history
            self.conversation_manager.add_user_message(user_message)
            self.conversation_manager.add_assistant_message(response.content)
            
            return response.content
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your request: {str(e)}"