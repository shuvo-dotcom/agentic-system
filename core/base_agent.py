"""
Base agent class for the agentic system.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage

from config.settings import OPENAI_API_KEY, OPENAI_MODEL, MAX_RETRIES, TIMEOUT


class BaseAgent(ABC):
    """
    Base class for all agents in the system.
    """
    
    def __init__(self, name: str, description: str, tools: Optional[List] = None):
        self.name = name
        self.description = description
        self.tools = tools or []
        self.logger = self._setup_logger()
        self.llm = self._setup_llm()
        self.agent_executor = self._setup_agent() if self.tools else None
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the agent."""
        logger = logging.getLogger(f"agent.{self.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _setup_llm(self) -> ChatOpenAI:
        """Set up the language model."""
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0,
            max_retries=MAX_RETRIES,
            request_timeout=TIMEOUT
        )
    
    def _setup_agent(self) -> Optional[AgentExecutor]:
        """Set up the agent executor with tools."""
        if not self.tools:
            return None
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are {self.name}. {self.description}"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        llm_with_tools = self.llm.bind(functions=[tool.to_dict() for tool in self.tools])
        
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
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

