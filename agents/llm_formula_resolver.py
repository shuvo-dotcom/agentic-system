
import json
from typing import Any, Dict, List

from core.simple_base_agent import SimpleBaseAgent
from openai import OpenAI

class LLMFormulaResolver(SimpleBaseAgent):
    """
    Agent responsible for dynamically identifying formulas, parameters, and generating
    executable Python code for calculations using an LLM.
    """

    def __init__(self):
        super().__init__(
            name="LLMFormulaResolver",
            description="Dynamically identifies formulas, extracts parameters, and generates executable code for calculations using an LLM."
        )
        self.client = OpenAI()

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        operation = input_data.get("operation")

        if operation == "resolve_and_generate_code":
            query = input_data.get("query")
            return await self._resolve_and_generate_code(query)
        else:
            return self.create_error_response(f"Unsupported operation: {operation}")

    async def _resolve_and_generate_code(self, query: str) -> Dict[str, Any]:
        """
        Uses LLM to identify the metric, formula, parameters, and generate executable Python code.
        """
        try:
            prompt = f"""You are an expert in energy economics and financial modeling. Your task is to analyze a user\"s query, identify the core metric they want to calculate, determine the relevant formula, extract all necessary parameters from the query (or suggest missing ones), and then generate executable Python code to perform the calculation.

Here\"s the user\"s query: \"{query}\"

Follow these steps:
1. Identify the \"metric_name\" (e.g., \"Levelized Cost of Energy\", \"Net Present Value\", \"Capacity Factor\").
2. Provide the \"formula\" in a mathematical expression. Use standard operators (+, -, *, /, **, sum(), etc.). For LCOE, use the form: (CAPEX + sum(OPEX_t[i] / (1 + discount_rate)**(i+1) for i in range(n))) / sum(energy_output_t[i] / (1 + discount_rate)**(i+1) for i in range(n)). For NPV, use: sum(cash_flow_t[i] / (1 + discount_rate)**i for i in range(n+1)).
3. List all \"parameters\" required for the formula. For each parameter, provide its \"description\", \"unit\", \"required\" (boolean), and \"value\" (extracted from the query, or null if missing). If a parameter is a time series, indicate it with \"type\": \"list\".
4. Generate \"executable_code\" in Python. This code should:
   - Define the formula as a Python function or direct calculation.
   - Use the extracted parameters as variables.
   - Handle time-series parameters (e.g., OPEX_t, energy_output_t, cash_flow_t) as lists.
   - Include any necessary imports (e.g., math, numpy).
   - Output the final calculated result by assigning it to a variable named `result`.
   - Ensure the code is robust and handles potential edge cases or missing parameters by suggesting defaults or asking for user input.

Provide the output as a JSON object ONLY, with the following keys: \"metric_name\", \"formula\", \"parameters\", \"executable_code\". Do NOT include any other text or markdown outside the JSON object.

Example for LCOE:
{{
  "metric_name": "Levelized Cost of Energy",
  "formula": "(CAPEX + sum(OPEX_t[i] / (1 + discount_rate)**(i+1) for i in range(n))) / sum(energy_output_t[i] / (1 + discount_rate)**(i+1) for i in range(n))",
  "parameters": {{
    "CAPEX": {{
      "description": "Capital expenditure",
      "unit": "$/kW",
      "required": true,
      "value": 2000.0
    }},
    "OPEX_t": {{
      "description": "Operating expenditure in year t",
      "unit": "$/kW/year",
      "required": true,
      "type": "list",
      "value": [50.0] * 20
    }},
    "energy_output_t": {{
      "description": "Energy output in year t",
      "unit": "MWh/year",
      "required": true,
      "type": "list",
      "value": null
    }},
    "discount_rate": {{
      "description": "Discount rate",
      "unit": "percentage",
      "required": true,
      "value": 0.08
    }},
    "n": {{
      "description": "Project lifetime",
      "unit": "years",
      "required": true,
      "value": 20
    }}
  }},
  "executable_code": "import math\n\nCAPEX = 2000.0\nOPEX_t = [50.0] * 20\ndiscount_rate = 0.08\nn = 20\n\n# If energy_output_t is not provided, estimate it\n# For LCOE, if energy_output_t is missing, we can estimate it using rated_capacity, capacity_factor, and hours_in_period\n# Assuming a rated_capacity of 1 kW for per kW basis calculation\nrated_capacity = 1.0 # kW\nhours_in_period = 8760 # hours/year\ncapacity_factor = 0.35 # from query\n\nannual_energy_output = rated_capacity * capacity_factor * hours_in_period # kWh/year\nenergy_output_t = [annual_energy_output / 1000] * n # Convert to MWh/year\n\n# Calculate LCOE\nsum_opex_discounted = sum(OPEX_t[i] / (1 + discount_rate)**(i+1) for i in range(n))\nsum_energy_discounted = sum(energy_output_t[i] / (1 + discount_rate)**(i+1) for i in range(n))\n\nLCOE = (CAPEX + sum_opex_discounted) / sum_energy_discounted\n\nresult = LCOE"
}}

Now, process the following query:
Query: \"{query}\"
Output:
"""
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini", # Changed model here
                response_format={"type": "json_object"},
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.choices[0].message.content
            parsed_content = json.loads(content)
            
            return self.create_success_response(parsed_content)

        except Exception as e:
            self.logger.error(f"Error in LLM formula resolution: {str(e)}")
            return self.create_error_response(f"LLM formula resolution failed: {str(e)}")



