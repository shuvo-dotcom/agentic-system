import json
from utils.llm_provider import get_llm_response

class PromptTypeDetectionAgent:
    """
    Detects the type of analysis required for a given sub-prompt using an LLM dynamically.
    Types:
    - 'time_series': Requires trend analysis over time.
    - 'pinpoint_value': Requires a single value calculation.
    """

    def __init__(self, api_key: str = ""):
        pass  # API key and client are now managed by llm_provider

    def detect_type(self, prompt: str) -> str:
        """
        Classify the prompt into one of the known types using an LLM.
        Returns the type name or 'other' if unrecognized.
        """
        system_prompt = (
            "You are an expert prompt classifier. "
            "Given a user prompt, classify it into one of the following types: "
            "'time_series', 'pinpoint_value'"
            "Respond with only the type name, no extra text."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = get_llm_response(
            messages,
            max_tokens=10,
            temperature=0.0
        )
        detected = response.strip().lower() if response else ""
        valid_types = {'time_series', 'pinpoint_value'}
        return detected if detected in valid_types else 'other'
