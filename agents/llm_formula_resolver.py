
import json
from typing import Any, Dict, List
from core.simple_base_agent import SimpleBaseAgent
from utils.llm_provider import get_llm_response

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
        # Using centralized LLM provider

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        operation = input_data.get("operation")

        if operation == "resolve_and_generate_code":
            query = input_data.get("query")
            data_context = input_data.get("data_context")
            accuracy_mode = input_data.get("accuracy_mode", False)
            return await self._resolve_and_generate_code(query, data_context, accuracy_mode)
        else:
            return self.create_error_response(f"Unsupported operation: {operation}")

    async def _resolve_and_generate_code(self, query: str, data_context: Dict = None, accuracy_mode: bool = False) -> Dict[str, Any]:
        """
        Uses LLM to identify the metric, formula, parameters, and generate executable Python code.
        Enhanced to work with real data context from CSV files.
        """
        try:
            # Build enhanced prompt with data context
            base_prompt = f"""You are an expert in energy economics and financial modeling. Your task is to analyze a user's query, identify the core metric they want to calculate, determine the relevant formula, extract all necessary parameters from the query (or suggest missing ones), and then generate executable Python code to perform the calculation.

            Here's the user's query: "{query}"
            """
            
            # Add data context if available
            if data_context and accuracy_mode:
                data_info = self._format_data_context(data_context)
                base_prompt += f"""

            IMPORTANT: You have access to real data from a CSV file. Use this data instead of making assumptions:
            
            {data_info}
            
            CRITICAL INSTRUCTIONS FOR ACCURACY MODE:
            - Use the actual data provided above instead of default assumptions
            - Extract real values from the data where possible
            - Reference specific data points and time periods from the dataset
            - If the data doesn't contain what's needed, clearly state what's missing
            - Generate code that works with the provided dataset structure
            """
            
            prompt = base_prompt + """

            Follow these steps:
            1. Identify the "metric_name" (e.g., "Levelized Cost of Energy", "Net Present Value", "Capacity Factor").
            2. Provide the "formula" in a mathematical expression. Use standard operators (+, -, *, /, **, sum(), etc.). For LCOE, use the form: (CAPEX + sum(OPEX_t[i] / (1 + discount_rate)**(i+1) for i in range(n))) / sum(energy_output_t[i] / (1 + discount_rate)**(i+1) for i in range(n)). For NPV, use: sum(cash_flow_t[i] / (1 + discount_rate)**i for i in range(n+1)).
            3. List all "parameters" required for the formula. For each parameter, provide its "description", "unit", "required" (boolean), and "value" (extracted from the query, or a reasonable default if missing). If a parameter is a time series, indicate it with "type": "list".
            4. Generate "executable_code" in Python. This code should:
              - Start by defining all parameters as variables with their values
              - Define the formula as a Python function or direct calculation
              - Use only the defined variables in calculations
              - Handle time-series parameters (e.g., OPEX_t, energy_output_t, cash_flow_t) as lists
              - Include any necessary imports (e.g., math, numpy) at the top
              - Output the final calculated result by assigning it to a variable named `result`
              - Be self-contained and executable without external input
              - Use reasonable default values for any missing parameters"""
            
            if data_context and accuracy_mode:
                prompt += """
              - USE THE REAL DATA PROVIDED - incorporate actual values from the dataset
              - Process the data_context variable that contains the extracted CSV data
              - Reference actual time periods and values from the dataset"""
            
            prompt += """

            CRITICAL: The executable_code must be completely self-contained. All variables used in calculations must be defined first. Do not use input() or print() statements.

            Provide the output as a JSON object ONLY, with the following keys: "metric_name", "formula", "parameters", "executable_code". Do NOT include any other text or markdown outside the JSON object.

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
              "executable_code": "import math\\n\\nCAPEX = 2000.0\\nOPEX_t = [50.0] * 20\\ndiscount_rate = 0.08\\nn = 20\\n\\n# If energy_output_t is not provided, estimate it\\n# For LCOE, if energy_output_t is missing, we can estimate it using rated_capacity, capacity_factor, and hours_in_period\\n# Assuming a rated_capacity of 1 kW for per kW basis calculation\\nrated_capacity = 1.0 # kW\\nhours_in_period = 8760 # hours/year\\ncapacity_factor = 0.35 # from query\\n\\nannual_energy_output = rated_capacity * capacity_factor * hours_in_period # kWh/year\\nenergy_output_t = [annual_energy_output / 1000] * n # Convert to MWh/year\\n\\n# Calculate LCOE\\nsum_opex_discounted = sum(OPEX_t[i] / (1 + discount_rate)**(i+1) for i in range(n))\\nsum_energy_discounted = sum(energy_output_t[i] / (1 + discount_rate)**(i+1) for i in range(n))\\n\\nLCOE = (CAPEX + sum_opex_discounted) / sum_energy_discounted\\n\\nresult = LCOE"
            }}

            Now, process the following query:
            Query: \"{query}\"
            Output:
            """
            
            messages = [{"role": "user", "content": prompt}]
            response_text = get_llm_response(messages, response_format={"type": "json_object"})
            
            parsed_content = json.loads(response_text)
            
            # Enhance the response with data context information
            if data_context:
                parsed_content["data_context_used"] = True
                parsed_content["data_source"] = data_context.get("file_info", {}).get("filename", "Unknown")
                parsed_content["extraction_strategy"] = data_context.get("extraction_strategy", {}).get("strategy_type", "Unknown")
            else:
                parsed_content["data_context_used"] = False
            
            return self.create_success_response(parsed_content)

        except Exception as e:
            self.logger.error(f"Error in LLM formula resolution: {str(e)}")
            return self.create_error_response(f"LLM formula resolution failed: {str(e)}")
    
    def _format_data_context(self, data_context: Dict) -> str:
        """Format the data context for inclusion in the LLM prompt."""
        
        context_parts = []
        
        # File information
        if "file_info" in data_context:
            file_info = data_context["file_info"]
            context_parts.append(f"Data Source: {file_info.get('filename', 'Unknown')}")
            context_parts.append(f"Total Records: {file_info.get('total_rows', 'Unknown')}")
            context_parts.append(f"Columns: {', '.join(file_info.get('columns', []))}")
        
        # Extracted data summary
        if "data_summary" in data_context:
            summary = data_context["data_summary"]
            context_parts.append(f"Extracted Records: {summary.get('total_records', 'Unknown')}")
            
            if "key_statistics" in summary:
                context_parts.append("Key Statistics:")
                for col, stats in summary["key_statistics"].items():
                    context_parts.append(f"  {col}: min={stats.get('min', 'N/A')}, max={stats.get('max', 'N/A')}, avg={stats.get('average', 'N/A')}")
        
        # Sample data
        if "extracted_data" in data_context:
            extracted = data_context["extracted_data"]
            if "data" in extracted and extracted["data"]:
                context_parts.append("Sample Data (first 3 records):")
                sample_data = extracted["data"][:3]
                for i, record in enumerate(sample_data, 1):
                    context_parts.append(f"  Record {i}: {record}")
        
        # Extraction strategy
        if "extraction_strategy" in data_context:
            strategy = data_context["extraction_strategy"]
            context_parts.append(f"Extraction Strategy: {strategy.get('strategy_type', 'Unknown')}")
            if "time_period" in strategy:
                time_info = strategy["time_period"]
                context_parts.append(f"Time Period: {time_info.get('description', 'Unknown')}")
        
        return "\n".join(context_parts)


if __name__ == "__main__":
    import argparse
    import json
    import asyncio
    parser = argparse.ArgumentParser(description="Run LLMFormulaResolver independently.")
    parser.add_argument('--query', type=str, help='Natural language prompt to generate code for')
    parser.add_argument('--json', type=str, help='Path to a JSON file with input_data (advanced)')
    args = parser.parse_args()

    if args.json:
        with open(args.json, 'r') as f:
            input_data = json.load(f)
    elif args.query:
        input_data = {"operation": "resolve_and_generate_code", "query": args.query}
    else:
        print("You must provide either --query or --json.")
        exit(1)

    agent = LLMFormulaResolver()
    # Run the async process method synchronously for CLI use
    result = asyncio.run(agent.process(input_data))
    print(json.dumps(result, indent=2, default=str))



