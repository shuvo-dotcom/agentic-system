"""
AutoGen-based base agent class for the agentic system.
"""
import logging
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

import autogen_agentchat as autogen
from autogen_agentchat import ConversableAgent, UserProxyAgent

from config.settings import OPENAI_API_KEY, OPENAI_MODEL, MAX_RETRIES, TIMEOUT


class AutoGenBaseAgent(ABC):
    """
    Base class for all AutoGen-based agents in the system.
    """
    
    def __init__(self, name: str, description: str, system_message: str = None, tools: Optional[List[Callable]] = None):
        self.name = name
        self.description = description
        self.tools = tools or []
        self.logger = self._setup_logger()
        
        # AutoGen configuration
        self.llm_config = {
            "model": OPENAI_MODEL,
            "api_key": OPENAI_API_KEY,
            "temperature": 0,
            "timeout": TIMEOUT,
            "max_tokens": 4000,
            "functions": [self._tool_to_function_schema(tool) for tool in self.tools] if self.tools else None
        }
        
        # Create the AutoGen agent
        self.agent = self._create_autogen_agent(system_message or self._get_default_system_message())
        
        # Register tools if any
        if self.tools:
            self._register_tools()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the agent."""
        logger = logging.getLogger(f"autogen_agent.{self.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _create_autogen_agent(self, system_message: str) -> ConversableAgent:
        """Create the AutoGen agent instance."""
        return ConversableAgent(
            name=self.name,
            system_message=system_message,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False
        )
    
    def _get_default_system_message(self) -> str:
        """Get default system message for the agent."""
        return f"""
        You are {self.name}, a specialized AI agent in an agentic system.
        
        Description: {self.description}
        
        Your responsibilities:
        1. Process input data according to your specialization
        2. Use available tools when necessary
        3. Return structured responses in JSON format
        4. Log your activities and decisions
        5. Handle errors gracefully
        
        Always respond with valid JSON containing:
        - "success": boolean indicating if the task was completed successfully
        - "data": the main result data (if successful)
        - "error": error information (if unsuccessful)
        - "metadata": additional information about the processing
        """
    
    def _tool_to_function_schema(self, tool: Callable) -> Dict:
        """Convert a tool function to OpenAI function schema."""
        # This is a simplified version - in practice, you'd want more sophisticated schema generation
        return {
            "name": tool.__name__,
            "description": tool.__doc__ or f"Tool function {tool.__name__}",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    
    def _register_tools(self):
        """Register tools with the AutoGen agent."""
        for tool in self.tools:
            self.agent.register_function(
                function_map={tool.__name__: tool}
            )
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Dictionary containing processing results
        """
        pass
    
    async def _execute_with_autogen(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a task using AutoGen conversation.
        """
        try:
            # Create a user proxy for the conversation
            user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False
            )
            
            # Prepare the message with context
            full_message = message
            if context:
                full_message = f"Context: {json.dumps(context)}\n\nTask: {message}"
            
            # Initiate conversation
            response = user_proxy.initiate_chat(
                self.agent,
                message=full_message,
                max_turns=1
            )
            
            # Extract the response
            if response and hasattr(response, 'chat_history') and response.chat_history:
                last_message = response.chat_history[-1]
                if isinstance(last_message, dict) and 'content' in last_message:
                    response_content = last_message['content']
                else:
                    response_content = str(last_message)
            else:
                response_content = "No response received"
            
            # Try to parse as JSON
            try:
                result = json.loads(response_content)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
            
            # If not JSON, wrap in success response
            return self.create_success_response({"response": response_content})
            
        except Exception as e:
            self.logger.error(f"Error in AutoGen execution: {str(e)}")
            return self.create_error_response(f"AutoGen execution failed: {str(e)}")
    
    def log_activity(self, activity: str, data: Optional[Dict] = None):
        """Log agent activity."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "activity": activity,
            "data": data or {}
        }
        self.logger.info(json.dumps(log_entry))
    
    def validate_input(self, input_data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate input data contains required fields."""
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            return False
        return True
    
    def create_error_response(self, error_message: str, error_code: str = "PROCESSING_ERROR") -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": {
                "code": error_code,
                "message": error_message,
                "agent": self.name,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def create_success_response(self, data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
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
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "name": self.name,
            "description": self.description,
            "tools": [tool.__name__ for tool in self.tools],
            "model": OPENAI_MODEL
        }

