
import re
from typing import Dict, Any, List
from core.simple_base_agent import SimpleBaseAgent

class QueryParser(SimpleBaseAgent):
    """
    A specialist agent for parsing natural language queries and extracting
    relevant parameters, including numerical values, units, and entities.
    """
    def __init__(self):
        super().__init__("QueryParser", "Parses natural language queries to extract parameters.")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input query to extract parameters.

        Args:
            input_data: A dictionary containing the 'query' string.

        Returns:
            A dictionary with extracted 'parameters' and a 'success' status.
        """
        query = input_data.get("query", "").lower()
        self.logger.info(f"QueryParser processing query: {query[:100]}...")

        extracted_params = {}
        success = True
        error_message = None

        try:
            # General numerical extraction (e.g., $100, 20%, 5 years)
            numbers = re.findall(r'\b(\d+\.?\d*)\s*(?:%|/kw|/kwh|/year|years|mw|mwh|gwh|twh|tonnes|dollars|\$)?\b', query)
            # This is a very basic extraction. More sophisticated NLP would be needed for complex cases.
            # For now, we'll focus on specific patterns for known metrics.

            # Specific parameter extraction for LCOE-related terms
            if "capex" in query:
                match = re.search(r"capex(?: of)?(?: \$)?([\d\.,]+)", query)
                if match: extracted_params["CAPEX"] = float(match.group(1).replace(",", ""))

            if "opex" in query:
                # OPEX can be per year or total, assuming per year for now
                match = re.search(r"opex(?: of)?(?: \$)?([\d\.,]+)(?:/kw/year)?", query)
                if match: extracted_params["OPEX_t"] = float(match.group(1).replace(",", ""))

            if "capacity factor" in query:
                match = re.search(r"capacity factor(?: of)?(?: is)?\s*([\d\.]+)(?:%|percent)?", query)
                if match: extracted_params["capacity_factor"] = float(match.group(1)) / 100.0

            if "discount rate" in query:
                match = re.search(r"discount rate(?: of)?(?: is)?\s*([\d\.]+)(?:%|percent)?", query)
                if match: extracted_params["discount_rate"] = float(match.group(1)) / 100.0

            if "lifetime" in query:
                match = re.search(r"([\d\.]+)\s*(?:year|years)\s*lifetime", query)
                if match: extracted_params["n"] = int(float(match.group(1)))

            # For energy_output_t, it's usually derived or looked up, not directly in query
            # This will be handled by DataHarvester or integration with Plexos data

            self.logger.info(f"Extracted parameters: {extracted_params}")

        except Exception as e:
            self.logger.error(f"Error during query parsing: {str(e)}")
            success = False
            error_message = str(e)

        return {
            "success": success,
            "parameters": extracted_params,
            "error": error_message
        }



