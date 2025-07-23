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

    async def analyze_query(self, prompt: str, session_id: str, agent_name: str, call_tree: dict, get_node_id) -> dict:
        """
        Analyzes a prompt to calculate a single value.
        """
        print(f"PinpointValueAgent received prompt: {prompt}")
        
        # 1. Resolve formula
        formula_input = {"operation": "resolve_and_generate_code", "query": prompt}
        formula_details_response = await self.formula_resolver.process(formula_input)
        
        if formula_details_response['status'] != 'success':
            return {"error": "Failed to resolve formula."}
            
        formula_details = formula_details_response['data']
        formula = formula_details.get('formula')
        required_params_info = formula_details.get('parameters', {})
        
        # 2. Extract parameters from prompt
        extracted_params = {}
        for param_name, param_info in required_params_info.items():
            value = await self.param_worker.extract(prompt, param_name)
            if value is not None:
                extracted_params[param_name] = value
        
        # 3. Execute formula
        result = self._execute_formula(formula, extracted_params)

        return {
            "prompt": prompt,
            "formula": formula,
            "parameters": extracted_params,
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
