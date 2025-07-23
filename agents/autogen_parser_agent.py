import json
import argparse
from config.settings import OPENAI_API_KEY, OPENAI_MODEL
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
        model_client = OpenAIChatCompletionClient(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
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
        self.agent.llm_config = {
            "model": OPENAI_MODEL,
            "api_key": OPENAI_API_KEY,
            "temperature": 0,
            "timeout": 60,
            "max_tokens": 2000
        }

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
            # --- Post-processing: clean all parameter values ---
            def clean_value(val):
                if isinstance(val, str):
                    import re
                    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", val.replace(",", ""))
                    if match:
                        return float(match.group())
                    return val
                return val
            if isinstance(parameters, dict):
                for param in parameters.values():
                    if isinstance(param, dict) and "value" in param:
                        param["value"] = clean_value(param["value"])
            elif isinstance(parameters, list):
                for param in parameters:
                    if isinstance(param, dict) and "value" in param:
                        param["value"] = clean_value(param["value"])
            # --- Fallback: extract numbers from query for null values (no regex) ---
            def fallback_extract(query, param_name):
                import re
                # Regex: parameter name, optional 'of', ':', or '=', then a number (optionally with $)
                pattern = rf'{re.escape(param_name)}\s*(?:of|:|=)?\s*[^\d\$]*\$?(\d+[\.,]?\d*)'
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    num_str = match.group(1).replace(',', '')
                    print(f"[DEBUG] Regex match for {param_name}: '{match.group(0)}', value: '{num_str}'")
                    try:
                        val = float(num_str)
                        print(f"[DEBUG] Fallback extracted for {param_name}: {val}")
                        return val
                    except Exception:
                        pass
                print(f"[DEBUG] Fallback found nothing for {param_name}")
                return None
            # Apply fallback for any null values
            if isinstance(parameters, dict):
                param_iter = parameters.values()
            elif isinstance(parameters, list):
                param_iter = parameters
            else:
                param_iter = []
            for param in param_iter:
                if isinstance(param, dict) and (param.get("value") is None or param.get("value") == "null"):
                    fallback = fallback_extract(query, param.get("name") or param.get("Name") or "")
                    if fallback is not None:
                        param["value"] = fallback
            return {"success": True, "parameters": parameters, "raw": content}
        except Exception as e:
            return {"success": False, "error": str(e), "raw": str(response)}

    async def on_reset(self, message: Reset):
        # This is a placeholder for reset logic if needed
        return {"success": True, "message": "Agent memory reset."}

    async def parse(self, query: str) -> dict:
        # For backward compatibility: wrap query in TextMessage
        return await self.on_message(TextMessage(content=query, source="user"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoGenParserAgent independently.")
    parser.add_argument('--query', type=str, help='Natural language query to parse')
    args = parser.parse_args()

    if not args.query:
        print("You must provide a query string via --query.")
        exit(1)

    agent = AutoGenParserAgent()
    result = asyncio.run(agent.parse(args.query))
    print(json.dumps(result, indent=2, default=str)) 