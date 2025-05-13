"""
Chatbot Module
Handles interaction with Groq LLM API and manages conversation context
"""

import os
import logging
import base64
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self, groq_api_key: str, db_handler, conversation_manager):
        """
        Initialize chatbot with Groq LLM API
        
        Args:
            groq_api_key: Groq API key
            db_handler: MongoDB handler for retrieval
            conversation_manager: Conversation manager for context
        """
        self.groq_api_key = groq_api_key
        os.environ["GROQ_API_KEY"] = groq_api_key
        self.db_handler = db_handler
        self.conversation_manager = conversation_manager
        
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama3-8b-8192",
            max_tokens=1024,
            temperature=0.7
        )
        
        # Initialize vision model for image understanding
        self.vision_llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama3-70b-8192-vision",
            max_tokens=1024,
            temperature=0.5
        )
        
        logger.info("Chatbot initialized with Groq LLM API")
    
    def process_query(self, query: str, conversation_id: str) -> str:
        """
        Process user query with context from conversation history
        
        Args:
            query: User query
            conversation_id: Conversation ID
            
        Returns:
            str: Chatbot response
        """
        try:
            # Add user message to conversation
            self.conversation_manager.add_message(conversation_id, "user", query)
            
            # Retrieve relevant chunks from database
            relevant_chunks = self.db_handler.retrieve_relevant_chunks(query)
            
            # Extract relevant PDF hashes and page numbers for image retrieval
            pdf_hashes = set()
            page_numbers = set()
            
            for chunk in relevant_chunks:
                if "pdf_hash" in chunk:
                    pdf_hashes.add(chunk["pdf_hash"])
                if "metadata" in chunk and "page" in chunk["metadata"]:
                    page_numbers.add(chunk["metadata"]["page"])
            
            # Prepare context from text chunks
            context_text = ""
            
            for i, chunk in enumerate(relevant_chunks):
                content = chunk.get("content", "")
                context_text += f"\nRelevant content {i+1}:\n{content}\n"
            
            # Retrieve relevant images if available
            relevant_images = []
            for pdf_hash in pdf_hashes:
                images = self.db_handler.retrieve_relevant_images(
                    pdf_hash, 
                    list(page_numbers) if page_numbers else None
                )
                relevant_images.extend(images)
            
            # Process images if available
            vision_insights = ""
            if relevant_images:
                vision_insights = self._process_images_with_vision_model(query, relevant_images)
            
            # Get conversation history for context
            conversation_history = self.conversation_manager.get_messages_for_context(conversation_id)
            
            # Format conversation history for the model
            formatted_messages = []
            for msg in conversation_history:
                if msg["role"] == "user":
                    formatted_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_messages.append(AIMessage(content=msg["content"]))
            
            # Create system prompt
            system_prompt = f"""You are a helpful assistant that answers questions about NCERT textbooks for classes 11 and 12 in physics, chemistry, and other subjects.
            
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know.
Be precise, accurate, and educational in your responses.

CONTEXT:
{context_text}

VISION MODEL INSIGHTS (from diagrams and images in the textbook):
{vision_insights}

Answer the question with proper formatting, equations (when needed), and clear explanations.
"""
            
            # Generate response with context and history
            messages = [
                SystemMessage(content=system_prompt),
                *formatted_messages,
                HumanMessage(content=query)
            ]
            
            # Generate response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Add assistant message to conversation
            self.conversation_manager.add_message(conversation_id, "assistant", response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_message = f"I'm sorry, I encountered an error while processing your query: {str(e)}"
            self.conversation_manager.add_message(conversation_id, "assistant", error_message)
            return error_message
    
    def _process_images_with_vision_model(self, query: str, images: List[Dict[str, Any]]) -> str:
        """
        Process images with Groq vision model
        
        Args:
            query: User query
            images: List of image data
            
        Returns:
            str: Vision model insights
        """
        if not images:
            return ""
        
        try:
            # Prepare prompt for vision model
            vision_prompt = f"""Analyze the following images from NCERT textbooks and provide insights related to the query: "{query}"
            
Focus on:
1. Identifying what the diagrams/images represent
2. Explaining key concepts visible in the images
3. Relating the images to physical or chemical principles
4. Any mathematical formulas or equations visible in the images
5. Any experimental setups or procedures shown

Provide a detailed analysis that would help answer the query."""
            
            # Process each image with vision model (up to 3 to avoid overloading)
            insights = []
            
            for idx, image_data in enumerate(images[:3]):
                try:
                    # Get image base64 data
                    img_base64 = image_data.get("base64", "")
                    if not img_base64:
                        continue
                    
                    # Create image message for vision model
                    image_format = image_data.get("format", "png")
                    image_caption = image_data.get("caption", f"Image {idx+1}")
                    
                    image_message = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{img_base64}",
                            "detail": "high"
                        }
                    }
                    
                    # Create messages for vision model
                    vision_messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"{vision_prompt}\n\nImage caption: {image_caption}"},
                                image_message
                            ]
                        }
                    ]
                    
                    # Call vision model
                    vision_response = self.vision_llm.invoke(vision_messages)
                    
                    # Add insight
                    insights.append(f"Image {idx+1} ({image_caption}):\n{vision_response.content}")
                    
                except Exception as e:
                    logger.warning(f"Error processing image {idx}: {str(e)}")
                    continue
            
            return "\n\n".join(insights)
            
        except Exception as e:
            logger.error(f"Error processing images with vision model: {str(e)}")
            return "Error processing images: " + str(e)