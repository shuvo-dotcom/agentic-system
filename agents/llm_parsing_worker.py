import json
from core.simple_base_agent import SimpleBaseAgent
from utils.llm_provider import get_llm_response, get_openai_client
from config.settings import OPENAI_API_KEY

class ParsingWorkerAgent(SimpleBaseAgent):
    """
    LLM-powered agent that extracts the value for a specific parameter from a query.
    """
    def __init__(self):
        super().__init__("ParsingWorkerAgent", "Extracts a specific parameter value from a query using LLM.")
        self.client = get_openai_client()

    async def extract(self, query: str, parameter: str, model: str = None) -> dict:
        # Try gpt-4.1-mini first, then gpt-4o, then gpt-4 as fallback if value is null
        models_to_try = [model] if model else ["gpt-4.1-mini", "gpt-4o", "gpt-4"]
        for m in models_to_try:
            prompt = f"""
You are an expert information extraction agent. Extract the numeric value for the parameter '{parameter}' from the following query. The value may be written as a currency (e.g., $2000/kW), with or without units or symbols. If there are multiple numbers, choose the one most likely to be the value for '{parameter}' based on context. Return only the number, or null if not found, as a JSON object: {{\"value\": <number or null>}}. Do not return any text except the JSON object.
Query: \"{query}\"
"""
            try:
                if m in ["gpt-4.1-mini", "gpt-4o"]:
                    response = self.client.chat.completions.create(
                        model=m,
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": prompt}]
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=m,
                        messages=[{"role": "user", "content": prompt}]
                    )
                content = response.choices[0].message.content
                print(f"[DEBUG] LLM ({m}) raw output for {parameter}: {content}")
                # Try to parse as a number or null
                try:
                    parsed = json.loads(content)
                    value = parsed.get("value", parsed)
                except Exception:
                    # Fallback: try to extract a number from the string
                    import re
                    match = re.search(r"[-+]?\d*[\.,]?\d+", content)
                    if match:
                        try:
                            value = float(match.group().replace(",", ""))
                        except Exception:
                            value = None
                    else:
                        value = None
                if value is not None:
                    return {"success": True, "parameter": parameter, "value": value, "raw": content, "model": m}
            except Exception as e:
                print(f"[DEBUG] LLM ({m}) error for {parameter}: {e}")
                continue
        return {"success": True, "parameter": parameter, "value": None, "raw": None, "model": models_to_try[-1]}

    @staticmethod
    async def extract_param_names(query: str, model: str = "gpt-4.1-mini") -> list:
        client = get_openai_client()
        prompt = f"""
List all parameter names (as a JSON array of strings) that are relevant for calculation in the following query. Only return the JSON array, no extra text.
Query: \"{query}\"
"""
        try:
            if model in ["gpt-4.1-mini", "gpt-4o"]:
                response = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}]
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
            content = response.choices[0].message.content
            print(f"[DEBUG] LLM ({model}) param name extraction raw output: {content}")
            # Try to parse as a JSON array
            import json
            try:
                arr = json.loads(content)
                if isinstance(arr, dict) and "parameters" in arr:
                    arr = arr["parameters"]
                if isinstance(arr, list):
                    return [str(x) for x in arr]
            except Exception:
                pass
            # Fallback: extract quoted strings
            import re
            return re.findall(r'"([^"]+)"', content)
        except Exception as e:
            print(f"[DEBUG] LLM param name extraction error: {e}")
            return []

    async def process(self, input_data):
        # Dummy implementation to satisfy abstract base class
        return {"success": False, "error": "Not implemented. Use extract() instead."} 