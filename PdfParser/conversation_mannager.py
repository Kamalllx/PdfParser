"""
Conversation Manager
Handles conversation history and context management
"""

import logging
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, db_handler):
        """
        Initialize conversation manager
        
        Args:
            db_handler: MongoDB handler for storing conversations
        """
        self.db_handler = db_handler
        self.active_conversations = {}
    
    def create_conversation(self) -> str:
        """
        Create a new conversation
        
        Returns:
            str: Conversation ID
        """
        metadata = {
            "created_at": time.time(),
            "last_active": time.time()
        }
        conversation_id = self.db_handler.create_conversation(metadata)
        self.active_conversations[conversation_id] = {
            "messages": [],
            "metadata": metadata
        }
        logger.info(f"Created new conversation with ID: {conversation_id}")
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, 
                   metadata: Dict[str, Any] = None) -> None:
        """
        Add a message to a conversation
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user/assistant)
            content: Message content
            metadata: Additional message metadata
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Update database
        self.db_handler.add_message(conversation_id, message)
        
        # Update local cache
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["messages"].append(message)
            self.active_conversations[conversation_id]["metadata"]["last_active"] = time.time()
        else:
            # Load conversation from database if not in local cache
            conversation = self.db_handler.get_conversation(conversation_id)
            if conversation:
                self.active_conversations[conversation_id] = conversation
    
    def get_conversation_history(self, conversation_id: str, 
                                limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of recent messages
        """
        if conversation_id in self.active_conversations:
            messages = self.active_conversations[conversation_id]["messages"]
            return messages[-limit:] if len(messages) > limit else messages
        else:
            # Load conversation from database
            conversation = self.db_handler.get_conversation(conversation_id)
            if conversation:
                self.active_conversations[conversation_id] = conversation
                messages = conversation["messages"]
                return messages[-limit:] if len(messages) > limit else messages
            return []
    
    def get_conversation_as_string(self, conversation_id: str, 
                                  limit: int = 10) -> str:
        """
        Get conversation history as a formatted string
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to include
            
        Returns:
            Formatted conversation history
        """
        messages = self.get_conversation_history(conversation_id, limit)
        formatted_messages = []
        
        for message in messages:
            role = message["role"].capitalize()
            content = message["content"]
            formatted_messages.append(f"{role}: {content}")
        
        return "\n\n".join(formatted_messages)
    
    def get_messages_for_context(self, conversation_id: str, 
                                limit: int = 5) -> List[Dict[str, str]]:
        """
        Get recent messages in a format suitable for LLM context
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to include
            
        Returns:
            List of message dicts with role and content
        """
        messages = self.get_conversation_history(conversation_id, limit)
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]