
import re
from typing import Dict, Any, List
from core.simple_base_agent import SimpleBaseAgent
from utils.llm_provider import get_llm_response
import json

class QueryParser(SimpleBaseAgent):
    """
    A specialist agent for parsing natural language queries and extracting
    relevant parameters, including numerical values, units, and entities.
    Now uses an LLM for dynamic extraction.
    """
    def __init__(self):
        super().__init__("QueryParser", "Parses natural language queries to extract parameters using LLM.")
        # Using centralized LLM provider

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input query to extract parameters using an LLM.

        Args:
            input_data: A dictionary containing the 'query' string.

        Returns:
            A dictionary with extracted 'parameters' and a 'success' status.
        """
        query = input_data.get("query", "")
        self.logger.info(f"QueryParser processing query: {query[:100]}...")

        try:
            prompt = f"""You are an expert data extraction agent. Given the following user query, extract all relevant parameters (names, values, units, and context) as a JSON object. For each parameter, include its name, value, unit (if any), and a short context or description if possible. If a value contains a currency symbol or other non-numeric characters (e.g., '$2000/kW'), extract the numeric value (e.g., 2000) and provide the unit (e.g., '/kW'). Always provide the value as a number if possible. Only return the JSON object, do not include any other text.\n\nQuery: \"{query}\"\n\nExample output:\n{{\n  \"parameters\": {{\n    \"CAPEX\": {{\"value\": 2000, \"unit\": \"$/kW\", \"description\": \"Capital expenditure\"}},\n    \"OPEX_t\": {{\"value\": 50, \"unit\": \"$/kW/year\", \"description\": \"Operating expenditure per year\"}},\n    \"capacity_factor\": {{\"value\": 0.35, \"unit\": \"fraction\", \"description\": \"Capacity factor\"}},\n    \"discount_rate\": {{\"value\": 0.08, \"unit\": \"fraction\", \"description\": \"Discount rate\"}},\n    \"n\": {{\"value\": 20, \"unit\": \"years\", \"description\": \"Project lifetime\"}}\n  }}\n}}\n"""
            
            messages = [{"role": "user", "content": prompt}]
            response_text = get_llm_response(messages, response_format={"type": "json_object"})
            
            parsed_content = json.loads(response_text)
            parameters = parsed_content.get("parameters", {})
            # --- Post-processing: clean all parameter values ---
            def clean_value(val):
                if isinstance(val, str):
                    # Remove $ and commas, extract first float or int
                    import re
                    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", val.replace(",", ""))
                    if match:
                        return float(match.group())
                    return val
                return val
            for param in parameters.values():
                if isinstance(param, dict) and "value" in param:
                    param["value"] = clean_value(param["value"])
            return {"success": True, "parameters": parameters, "error": None}
        except Exception as e:
            self.logger.error(f"Error during query parsing: {str(e)}")
            return {"success": False, "parameters": {}, "error": str(e)}


if __name__ == "__main__":
    import argparse
    import json
    import asyncio
    parser = argparse.ArgumentParser(description="Run QueryParser independently.")
    parser.add_argument('--query', type=str, help='Natural language query to parse')
    args = parser.parse_args()

    if not args.query:
        print("You must provide a query string via --query.")
        exit(1)

    agent = QueryParser()
    result = asyncio.run(agent.process({"query": args.query}))
    print(json.dumps(result, indent=2, default=str))



