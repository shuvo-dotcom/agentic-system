import re
import asyncio
from agents.llm_parsing_worker import ParsingWorkerAgent
from agents.param_workers import CurrencyUnitWorker, ListSeriesWorker, SimpleNumberWorker

def regex_fallback(query, param_name):
    import re
    # 1. Look for 'PARAM of $NUMBER', 'PARAM: $NUMBER', 'PARAM = $NUMBER', or PARAM ... $NUMBER (up to 3 words, possibly embedded in a token)
    patterns = [
        rf'{re.escape(param_name)}\s*(?:of|:|=)?\s*[^\d\$]*\$?(\d+[\.,]?\d*)',
        rf'{re.escape(param_name)}(?:\s+\w+){{0,3}}?[^\d\$]*\$?(\d+[\.,]?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                return float(num_str)
            except Exception:
                continue
    # 2. Fallback: nearest number heuristic
    num_matches = list(re.finditer(r'[-+]?\$?\d*[\.,]?\d+', query))
    if not num_matches:
        return None
    param_matches = list(re.finditer(re.escape(param_name), query, re.IGNORECASE))
    if not param_matches:
        return None
    min_dist = float('inf')
    best_num = None
    for num_match in num_matches:
        num_pos = num_match.start()
        num_str = num_match.group().replace('$', '').replace(',', '')
        for param_match in param_matches:
            param_pos = param_match.start()
            dist = abs(num_pos - param_pos)
            if dist < min_dist:
                min_dist = dist
                best_num = num_str
    if best_num is not None:
        try:
            return float(best_num)
        except Exception:
            return None
    # 3. Final fallback: tokenization-based window after parameter name, splitting on non-numeric characters
    tokens = re.split(r'[^\w\$\.,/]+', query)
    tokens_lower = [t.lower() for t in tokens]
    for i, token in enumerate(tokens_lower):
        if param_name.lower() == token:
            for j in range(i+1, min(i+6, len(tokens))):
                candidate = tokens[j]
                # Split candidate on non-numeric characters and check each part
                for part in re.split(r'[^\d\.\-\+]', candidate):
                    if part:
                        try:
                            return float(part)
                        except Exception:
                            continue
    return None

class ParsingManagerAgent:
    """
    Dynamic, adaptive manager agent that coordinates ParsingWorkerAgent(s) to extract all required parameters from a query.
    """
    def __init__(self):
        self.worker = ParsingWorkerAgent()
        self.currency_worker = CurrencyUnitWorker()
        self.list_worker = ListSeriesWorker()
        self.simple_worker = SimpleNumberWorker()

    async def extract_all(self, query: str, parameters: list) -> dict:
        results = {}
        logs = {}
        for param in parameters:
            value = None
            explanations = []
            for attempt in range(5):
                if attempt == 0:
                    # Normal extraction
                    result = await self.worker.extract(query, param)
                    explanations.append(result.get("raw"))
                elif attempt == 1:
                    # Step-by-step explanation and retry
                    explain_prompt = f"Explain step by step why you could not extract the value for '{param}' from the following query, then try again to extract the numeric value, ignoring units and currency symbols. Only return the number or null. Query: \"{query}\""
                    result = await self.worker.extract(explain_prompt, param)
                    explanations.append(result.get("raw"))
                elif attempt == 2:
                    # List all numbers and their context, then select the best one
                    context_prompt = f"List all numbers in the query, along with the words immediately before and after each number. Then, select the number most likely to be the value for '{param}'. Return only that number, or null if not found. Query: \"{query}\""
                    result = await self.worker.extract(context_prompt, param)
                    explanations.append(result.get("raw"))
                elif attempt == 3:
                    # Specialized worker routing
                    param_lower = param.lower()
                    if any(x in param_lower for x in ["capex", "opex", "cost", "price", "$", "/"]):
                        value = await self.currency_worker.extract(query, param)
                        explanations.append(f"[currency worker] {value}")
                    elif any(x in param_lower for x in ["cash flow", "return", "series", "annual", "list"]):
                        value = await self.list_worker.extract(query, param)
                        explanations.append(f"[list worker] {value}")
                    else:
                        value = await self.simple_worker.extract(query, param)
                        explanations.append(f"[simple worker] {value}")
                    if value is not None:
                        break
                elif attempt == 4:
                    # Regex/heuristic fallback
                    value = regex_fallback(query, param)
                    explanations.append(f"[regex fallback] {value}")
                    break
                value = result.get("value")
                if value is not None:
                    break
            results[param] = value
            logs[param] = explanations
        return {"success": True, "parameters": results, "logs": logs} 