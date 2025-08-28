import json
import argparse
import os
from config.settings import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_PROJECT_ID
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents._base_chat_agent import TextMessage as AutoGenTextMessage
from autogen_core import CancellationToken
import asyncio
import sys
from agents.messages import TextMessage, Reset

class AutoGenParserAgent:
    def __init__(self):
        try:
            from autogen_ext.models.openai import OpenAIChatCompletionClient
        except ImportError:
            print("\nERROR: The package 'autogen-ext' is required for OpenAIChatCompletionClient.\nInstall it with: pip install autogen-ext\nOr: pip install 'pyautogen[ext]'\n")
            sys.exit(1)
            
        # Get project ID from settings
        project_id = OPENAI_PROJECT_ID
        
        # Check if we're using a project-scoped API key
        is_project_key = OPENAI_API_KEY.startswith("sk-proj-")
        
        # Setup default headers if using project-scoped key
        default_headers = {}
        if is_project_key and project_id:
            default_headers = {"OpenAI-Project": project_id}
            print(f"🔑 Using Nohm project ID: {project_id}")
        
        # Create the OpenAI client with proper configuration
        model_client = OpenAIChatCompletionClient(
            model=OPENAI_MODEL, 
            api_key=OPENAI_API_KEY,
            default_headers=default_headers
        )
        
        self.agent = AssistantAgent(
            name="AutoGenParserAgent",
            model_client=model_client,
            description="A world-class information extraction agent. Given a user query, extract all relevant parameters (names, values, units, and a short context/description) and return them as a JSON object. For each parameter, include: name, value (as a number, if possible), unit (if any), description (short context). If a value is not explicitly given, set it to null. If a value contains a currency symbol or other non-numeric characters (e.g., '$2000/kW'), extract the numeric value (e.g., 2000) and provide the unit (e.g., '/kW'). Always provide the value as a number if possible. Only return the JSON object, no extra text.",
            system_message="""
You are a world-class information extraction agent. Given a user query, extract all relevant parameters (names, values, units, and a short context/description) and return them as a JSON object. For each parameter, include:
- name
- value (as a number, if possible)
- unit (if any)
- description (short context)
If a value is not explicitly given, set it to null.
If a value contains a currency symbol or other non-numeric characters (e.g., '$2000/kW'), extract the numeric value (e.g., 2000) and provide the unit (e.g., '/kW'). Always provide the value as a number if possible.
Only return the JSON object, no extra text.
"""
        )
        
        # Update LLM config with project header if needed
        llm_config = {
            "model": OPENAI_MODEL,
            "api_key": OPENAI_API_KEY,
            "temperature": 0,
            "timeout": 60,
            "max_tokens": 2000
        }
        
        # Add default headers to the config if using a project-scoped key
        if is_project_key and project_id:
            llm_config["default_headers"] = default_headers
            
        self.agent.llm_config = llm_config

    async def on_message(self, message: TextMessage):
        query = message.content
        prompt = f"Extract all relevant parameters from the following query and return as JSON. If a value contains a currency symbol or other non-numeric characters (e.g., '$2000/kW'), extract the numeric value (e.g., 2000) and provide the unit (e.g., '/kW'). Always provide the value as a number if possible.\nQuery: \"{query}\""
        user_message = AutoGenTextMessage(content=prompt, source=message.source)
        response = await self.agent.on_messages([user_message], CancellationToken())
        # Extract the final message content
        try:
            if hasattr(response, 'chat_message') and hasattr(response.chat_message, 'content'):
                content = response.chat_message.content
            elif isinstance(response, dict) and 'content' in response:
                content = response['content']
            else:
                content = str(response)
            # Remove code block formatting if present
            if content.strip().startswith('```'):
                content = content.strip().lstrip('`').lstrip('json').strip()
                content = content.split('```')[0].strip()
            parsed = json.loads(content)
            parameters = parsed.get("parameters", parsed) if isinstance(parsed, dict) else parsed
            return TextMessage(content=json.dumps(parameters, indent=2), source="AutoGenParserAgent")
        except json.JSONDecodeError:
            error_msg = f"Error parsing JSON from LLM response: {content}"
            return TextMessage(content=error_msg, source="AutoGenParserAgent")
        except Exception as e:
            error_msg = f"Error processing LLM response: {str(e)}"
            return TextMessage(content=error_msg, source="AutoGenParserAgent")

    def on_reset(self, message: Reset = None):
        return TextMessage(content="AutoGenParserAgent reset", source="AutoGenParserAgent")
