"""
Simplified Base Agent - Base class for agents without AutoGen dependencies.
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from utils.llm_provider import get_llm_response
from config.settings import MAX_RETRIES, TIMEOUT


class SimpleBaseAgent(ABC):
    """
    Simplified base class for all agents without AutoGen dependencies.
    """
    
    def __init__(self, name: str, description: str, tools: List[Callable] = None):
        self.name = name
        self.description = description
        self.tools = tools or []
        self.logger = logging.getLogger(f"agent.{name}")
        self.activity_log = []
        
        # Using centralized LLM provider
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processing results
        """
        pass
    
    def log_activity(self, activity: str, data: Dict[str, Any] = None):
        """Log agent activity."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "activity": activity,
            "data": data or {}
        }
        self.activity_log.append(log_entry)
        self.logger.info(f"{activity}: {json.dumps(data or {}, indent=2)}")
    
    def create_error_response(self, error_message: str, error_type: str = "processing_error") -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": {
                "message": error_message,
                "type": error_type,
                "agent": self.name,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def create_success_response(self, data: Dict[str, Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create standardized success response."""
        return {
            "success": True,
            "data": data,
            "metadata": {
                "agent": self.name,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
        }
    
    async def call_llm(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Call LLM API with error handling using centralized provider."""
        try:
            response_text = get_llm_response(messages, **kwargs)
            
            return {
                "success": True,
                "response": response_text,
                "metadata": {"provider": "centralized"}
            }
            
        except Exception as e:
            return self.create_error_response(f"LLM API call failed: {str(e)}")
    
    def get_activity_log(self) -> List[Dict[str, Any]]:
        """Get agent activity log."""
        return self.activity_log
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": self.name,
            "description": self.description,
            "tools_count": len(self.tools),
            "activities_logged": len(self.activity_log),
            "status": "active"
        }

