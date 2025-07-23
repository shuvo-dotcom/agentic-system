import re
from typing import Literal

class PromptTypeDetectionAgent:
    """
    Detects the type of analysis required for a given sub-prompt.
    Types:
    - 'time_series': Requires trend analysis over time.
    - 'pinpoint_value': Requires a single value calculation.
    - 'comparison': Compares two or more entities/values.
    - 'explanation': Asks for an explanation or reasoning.
    - 'data_retrieval': Asks to get, find, or list data.
    - 'other': Does not fit known types.
    """

    def detect_type(self, prompt: str) -> Literal['time_series', 'pinpoint_value', 'comparison', 'explanation', 'data_retrieval', 'other']:
        prompt_lower = prompt.lower()

        # Order is important to avoid misclassification
        
        # 1. Time series: look for time-related words and trend/forecast words
        if re.search(r'\b(trend|over time|per year|annual|monthly|daily|change over|evolution|history|forecast|projected|historical|time series)\b', prompt_lower):
            return 'time_series'

        # 2. Comparison: look for comparison words
        if re.search(r'\b(compare|difference|vs\.?|versus|greater than|less than|more than|between|which is better)\b', prompt_lower):
            return 'comparison'
            
        # 3. Explanation: look for "why", "explain", etc.
        if re.search(r'\b(why|explain|reason|how does|describe|meaning of|what is the purpose of)\b', prompt_lower):
            return 'explanation'

        # 4. Pinpoint value: look for calculation/determination words
        if re.search(r'\b(what is|calculate|find the value of|determine|compute|result of|how much|how many)\b', prompt_lower):
            return 'pinpoint_value'
            
        # 5. Data retrieval: look for get/find/list words
        if re.search(r'\b(get|find|list|show me|retrieve|what are)\b', prompt_lower):
            return 'data_retrieval'

        return 'other'
