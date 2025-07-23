import re
import asyncio
from agents.llm_formula_resolver import LLMFormulaResolver
from agents.param_workers import SimpleNumberWorker

class PinpointValueAgent:
    """
    Agent that handles single-value calculations.
    It uses an LLM to find a formula, extracts parameters, and calculates the result.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.formula_resolver = LLMFormulaResolver()
        self.param_worker = SimpleNumberWorker()

    async def analyze_query(self, prompt: str, session_id: str, agent_name: str, call_tree: dict, get_node_id, extracted_params: dict = None) -> dict:
        """
        Analyzes a prompt to calculate a single value.
        """
        print(f"PinpointValueAgent received prompt: {prompt}")
        
        # 1. Resolve formula
        formula_input = {"operation": "resolve_and_generate_code", "query": prompt}
        formula_details_response = await self.formula_resolver.process(formula_input)
        
        if not formula_details_response.get('success', False):
            return {"error": "Failed to resolve formula.", "details": formula_details_response.get('error')}
            
        formula_details = formula_details_response.get('data', {})
        formula = formula_details.get('formula')
        required_params_info = formula_details.get('parameters', {})
        
        # 2. Extract parameters from prompt or use provided extracted_params
        params = dict(extracted_params) if extracted_params else {}
        for param_name, param_info in required_params_info.items():
            if param_name not in params or params[param_name] is None:
                value = await self.param_worker.extract(prompt, param_name)
                if value is not None:
                    params[param_name] = value

        # Identify missing required parameters dynamically based on LLM parameter info
        missing_params = []
        for param_name, param_info in required_params_info.items():
            is_required = param_info.get('required', False)
            if is_required and (param_name not in params or params[param_name] is None):
                missing_params.append(param_name)

        if missing_params:
            # Return a response indicating missing parameters to ask the user
            return {
                "prompt": prompt,
                "formula": formula,
                "parameters": params,
                "missing_parameters": missing_params,
                "result": None,
                "message": f"Missing required parameters: {', '.join(missing_params)}. Please provide these values."
            }

        # Dynamic type conversion for all parameters before formula evaluation
        def dynamic_type_convert(value, expected_type):
            if value is None:
                return None
            # Remove any leading/trailing quotes and whitespace
            if isinstance(value, str):
                value = value.strip().strip('"').strip("'")
            if expected_type == "int":
                try:
                    # Extract first integer from string if present
                    import re
                    match = re.search(r'-?\d+', str(value))
                    if match:
                        return int(match.group(0))
                    return int(float(value))
                except Exception:
                    return None
            elif expected_type == "float":
                try:
                    # Extract first float from string if present
                    import re
                    match = re.search(r'-?\d+(\.\d+)?', str(value))
                    if match:
                        return float(match.group(0))
                    return float(value)
                except Exception:
                    return None
            elif expected_type == "yes/no":
                val = str(value).strip().lower()
                if val in ("yes", "no"):
                    return val
                return None
            else:
                return value

        for param_name, param_info in required_params_info.items():
            if param_name in params and params[param_name] is not None:
                expected_type = param_info.get("type", "string")
                params[param_name] = dynamic_type_convert(params[param_name], expected_type)

        # 3. Execute formula
        result = self._execute_formula(formula, params)

        return {
            "prompt": prompt,
            "formula": formula,
            "parameters": params,
            "result": result
        }

    def _execute_formula(self, formula: str, params: dict) -> float:
        # In a real scenario, use a safe execution environment like asteval
        # This is a simplified and insecure placeholder.
        for param, value in params.items():
            # Use word boundaries to avoid replacing parts of words
            formula = re.sub(r'\b' + re.escape(param) + r'\b', str(value), formula)
        
        try:
            # Basic math evaluation, highly insecure, for demo only
            return eval(formula, {"__builtins__": None}, {})
        except Exception as e:
            print(f"Error evaluating formula '{formula}': {e}")
            return float('nan')
