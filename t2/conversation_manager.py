"""
Conversation Manager for NCERT Study Assistant

This module manages the conversation history between the user and chatbot.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation history and context for the chatbot."""
    
    def __init__(self, max_history: int = 5):
        """
        Initialize the conversation manager.
        
        Args:
            max_history: Maximum number of message pairs to keep in history
        """
        self.max_history = max_history
        self.messages: List[BaseMessage] = []
        
        logger.info(f"Initialized ConversationManager with max_history={max_history}")
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            message: User message content
        """
        self.messages.append(HumanMessage(content=message))
        self._trim_history()
    
    def add_assistant_message(self, message: str) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            message: Assistant message content
        """
        self.messages.append(AIMessage(content=message))
        self._trim_history()
    
    def get_messages(self) -> List[BaseMessage]:
        """
        Get the conversation history.
        
        Returns:
            List of message objects
        """
        return self.messages
    
    def get_recent_history(self, num_pairs: Optional[int] = None) -> str:
        """
        Get a string representation of recent conversation history.
        
        Args:
            num_pairs: Number of message pairs to include (default: max_history)
        
        Returns:
            String containing recent conversation history
        """
        if num_pairs is None:
            num_pairs = self.max_history
        
        # Get the most recent messages
        recent_messages = self.messages[-(num_pairs * 2):]
        
        # Format the messages
        history_parts = []
        for i in range(0, len(recent_messages), 2):
            if i + 1 < len(recent_messages):
                user_msg = recent_messages[i].content
                assistant_msg = recent_messages[i + 1].content
                history_parts.append(f"User: {user_msg}\nAssistant: {assistant_msg}")
        
        return "\n\n".join(history_parts)
    
    def _trim_history(self) -> None:
        """Trim the conversation history to the maximum length."""
        # Keep an even number of messages (pairs of user and assistant messages)
        max_messages = self.max_history * 2
        
        if len(self.messages) > max_messages:
            # Keep only the most recent messages
            self.messages = self.messages[-max_messages:]
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []
        logger.info("Cleared conversation history")