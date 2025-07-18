"""
AutoGen-based Task Manager - Central orchestrator for the agentic system.
"""
import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

import autogen_agentchat as autogen
from autogen_agentchat import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager

from core.autogen_base_agent import AutoGenBaseAgent
from config.settings import OPENAI_API_KEY, OPENAI_MODEL


class AutoGenTaskManager(AutoGenBaseAgent):
    """
    Central task manager using AutoGen that orchestrates the entire agentic workflow.
    Parses user queries, decomposes them into atomic tasks, and coordinates specialist agents.
    """
    
    def __init__(self):
        super().__init__(
            name="TaskManager",
            description="Central orchestrator that parses user queries, decomposes them into atomic tasks, and coordinates specialist agents using AutoGen framework."
        )
        self.specialist_agents = {}
        self.task_history = []
        self.group_chat = None
        self.group_chat_manager = None
        
    def register_agent(self, agent_name: str, agent_instance: AutoGenBaseAgent):
        """Register a specialist agent with the task manager."""
        self.specialist_agents[agent_name] = agent_instance
        self.logger.info(f"Registered AutoGen agent: {agent_name}")
        
        # Recreate group chat with new agent
        self._setup_group_chat()
    
    def _setup_group_chat(self):
        """Set up AutoGen group chat with all registered agents."""
        if not self.specialist_agents:
            return
        
        # Create list of all agents for group chat
        agents = [self.agent]  # Task manager agent
        for agent_instance in self.specialist_agents.values():
            agents.append(agent_instance.agent)
        
        # Create group chat
        self.group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=20,
            speaker_selection_method="auto"
        )
        
        # Create group chat manager
        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config
        )
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method that handles user queries using AutoGen.
        
        Args:
            input_data: Dictionary containing 'query' and optional 'context'
            
        Returns:
            Final processed result
        """
        try:
            if not self.validate_input(input_data, ["query"]):
                return self.create_error_response("Missing required field: query")
            
            user_query = input_data["query"]
            context = input_data.get("context", {})
            
            self.log_activity("Processing user query with AutoGen", {"query": user_query})
            
            # Step 1: Use AutoGen to decompose and execute the query
            result = await self._process_with_autogen(user_query, context)
            
            # Store in task history
            self.task_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "context": context,
                "result": result
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return self.create_error_response(f"Processing failed: {str(e)}")
    
    async def _process_with_autogen(self, query: str, context: Dict) -> Dict[str, Any]:
        """
        Process the query using AutoGen's multi-agent conversation.
        """
        try:
            # Create a comprehensive system message for task decomposition
            task_decomposition_message = f"""
            You are the Task Manager in an agentic system. Your job is to:
            
            1. Analyze the user query: "{query}"
            2. Context: {json.dumps(context)}
            3. Break down the query into atomic tasks
            4. Coordinate with available specialist agents to execute these tasks
            5. Synthesize the results into a comprehensive response
            
            Available specialist agents and their capabilities:
            {self._get_agent_capabilities()}
            
            Please start by analyzing the query and determining which agents need to be involved.
            Then coordinate their work to provide a complete answer.
            
            Respond with a structured plan and then execute it step by step.
            """
            
            # If we have a group chat set up, use it
            if self.group_chat_manager and len(self.specialist_agents) > 0:
                return await self._execute_group_chat(task_decomposition_message)
            else:
                # Fallback to single agent processing
                return await self._execute_with_autogen(task_decomposition_message, context)
            
        except Exception as e:
            self.logger.error(f"Error in AutoGen processing: {str(e)}")
            return self.create_error_response(f"AutoGen processing failed: {str(e)}")
    
    async def _execute_group_chat(self, message: str) -> Dict[str, Any]:
        """
        Execute the task using AutoGen group chat.
        """
        try:
            # Create a user proxy to initiate the conversation
            user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
                system_message="You are a user proxy that initiates conversations and collects results."
            )
            
            # Start the group chat
            self.logger.info("Starting AutoGen group chat")
            
            chat_result = user_proxy.initiate_chat(
                self.group_chat_manager,
                message=message,
                max_turns=10
            )
            
            # Extract and process the results
            if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                # Get the last meaningful response
                final_response = None
                for msg in reversed(chat_result.chat_history):
                    if isinstance(msg, dict) and msg.get('content'):
                        content = msg['content']
                        # Try to find a JSON response
                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, dict) and 'success' in parsed:
                                final_response = parsed
                                break
                        except json.JSONDecodeError:
                            continue
                
                if final_response:
                    return final_response
                else:
                    # If no structured response found, create one from the conversation
                    conversation_summary = self._summarize_conversation(chat_result.chat_history)
                    return self.create_success_response({
                        "answer": conversation_summary,
                        "conversation_history": chat_result.chat_history[-5:]  # Last 5 messages
                    })
            else:
                return self.create_error_response("No response from group chat")
            
        except Exception as e:
            self.logger.error(f"Error in group chat execution: {str(e)}")
            return self.create_error_response(f"Group chat execution failed: {str(e)}")
    
    def _get_agent_capabilities(self) -> str:
        """Get a formatted string of agent capabilities."""
        if not self.specialist_agents:
            return "No specialist agents registered."
        
        capabilities = []
        for name, agent in self.specialist_agents.items():
            info = agent.get_agent_info()
            capabilities.append(f"- {name}: {info['description']}")
            if info['tools']:
                capabilities.append(f"  Tools: {', '.join(info['tools'])}")
        
        return "\n".join(capabilities)
    
    def _summarize_conversation(self, chat_history: List) -> str:
        """Summarize the conversation history into a coherent response."""
        try:
            # Extract meaningful content from the conversation
            meaningful_messages = []
            for msg in chat_history:
                if isinstance(msg, dict) and msg.get('content'):
                    content = msg['content']
                    if len(content.strip()) > 10:  # Filter out very short messages
                        meaningful_messages.append(content)
            
            if not meaningful_messages:
                return "No meaningful conversation found."
            
            # Join the messages into a summary
            summary = "Based on the multi-agent conversation:\n\n"
            summary += "\n\n".join(meaningful_messages[-3:])  # Last 3 meaningful messages
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing conversation: {str(e)}")
            return "Error summarizing the conversation results."
    
    async def process_direct_agent_call(self, agent_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Directly call a specific agent (useful for testing or specific workflows).
        """
        try:
            if agent_name not in self.specialist_agents:
                return self.create_error_response(f"Agent {agent_name} not found")
            
            agent = self.specialist_agents[agent_name]
            self.log_activity(f"Direct call to agent {agent_name}", input_data)
            
            result = await agent.process(input_data)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in direct agent call: {str(e)}")
            return self.create_error_response(f"Direct agent call failed: {str(e)}")
    
    def get_task_history(self) -> List[Dict]:
        """Get the history of processed tasks."""
        return self.task_history
    
    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent names."""
        return list(self.specialist_agents.keys())
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "task_manager": "active",
            "registered_agents": len(self.specialist_agents),
            "agent_list": list(self.specialist_agents.keys()),
            "group_chat_enabled": self.group_chat is not None,
            "tasks_processed": len(self.task_history),
            "last_activity": self.task_history[-1]["timestamp"] if self.task_history else None
        }

