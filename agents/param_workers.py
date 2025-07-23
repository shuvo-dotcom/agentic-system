import re

class CurrencyUnitWorker:
    async def extract(self, query, parameter):
        # Look for $ or currency/unit patterns after the parameter name
        pattern = rf'{re.escape(parameter)}.*?\$([\d,\.]+)'
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except Exception:
                pass
        # Fallback: any number after parameter name
        pattern2 = rf'{re.escape(parameter)}.*?([\d,\.]+)'
        match2 = re.search(pattern2, query, re.IGNORECASE)
        if match2:
            try:
                return float(match2.group(1).replace(',', ''))
            except Exception:
                pass
        return None

class ListSeriesWorker:
    async def extract(self, query, parameter):
        # Look for a list of numbers after the parameter name
        pattern = rf'{re.escape(parameter)}.*?((?:\$?[\d,\.]+(?:,| and | |$))+)'  # e.g., $50,000, $60,000, ...
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            nums = re.findall(r'\$?([\d,\.]+)', match.group(1))
            try:
                return [float(n.replace(',', '')) for n in nums if n]
            except Exception:
                pass
        return None

class SimpleNumberWorker:
    async def extract(self, query, parameter):
        # Look for a plain number after the parameter name
        pattern = rf'{re.escape(parameter)}.*?([\d,\.]+)'
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except Exception:
                pass
        return None 