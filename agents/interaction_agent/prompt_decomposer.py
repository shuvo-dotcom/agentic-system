import re
from typing import List, Optional, Dict
import json
import os
try:
    import pycountry  # Added for dynamic country extraction
    HAS_PYCOUNTRY = True
except ImportError:
    HAS_PYCOUNTRY = False

class PromptDecomposerAgent:
    """
    Dynamically detects if a prompt is complex (multi-entity, multi-metric, etc.) and decomposes it into atomic sub-prompts.
    Uses robust rule-based logic with dynamic extraction, and falls back to LLM if available.
    """
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        self.api_key = api_key
        # Using centralized LLM provider
        from utils.llm_provider import get_llm_response
        self.get_llm_response = get_llm_response
        
        # Load config-driven lists for metrics
        self.config = self._load_config(config_path)
        self.metrics = self.config.get("metrics", [
            "capacity factor", "efficiency", "output", "cost", "energy consumption", "installed capacity",
            "wind", "solar", "wind power", "solar power", "lcoe", "levelized cost", "generation", "capacity"
        ])
        self.time_patterns = self.config.get("time_patterns", [
            r"\b\d{4}\b", r"\bfrom \d{4} to \d{4}\b", r"\bbetween \d{4} and \d{4}\b", r"\b\d{4}-\d{4}\b"
        ])

    def _load_config(self, config_path: Optional[str]) -> Dict:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def extract_entities(self, prompt: str) -> Dict:
        found_countries = []
        prompt_lower = prompt.lower()
        
        # Manual country list as fallback
        countries_list = [
            "Germany", "France", "United Kingdom", "United States", "Italy", "Spain", 
            "Netherlands", "Belgium", "Austria", "Switzerland", "Sweden", "Norway",
            "Denmark", "Finland", "Poland", "Czech Republic", "Portugal", "Greece",
            "Ireland", "Luxembourg", "Slovenia", "Slovakia", "Estonia", "Latvia",
            "Lithuania", "Hungary", "Romania", "Bulgaria", "Croatia", "Cyprus",
            "Malta", "UK", "USA", "US"
        ]
        
        # Find countries in prompt
        for country in countries_list:
            if country.lower() in prompt_lower:
                if country == "UK":
                    found_countries.append("United Kingdom")
                elif country == "USA" or country == "US":
                    found_countries.append("United States")
                else:
                    found_countries.append(country)
        
        # Enhanced metric extraction: handle 'wind and solar power', 'wind, solar', etc.
        found_metrics = set()
        for m in self.metrics:
            if re.search(rf"\b{re.escape(m)}\b", prompt, re.IGNORECASE):
                found_metrics.add(m)
        metric_group_pattern = r"(wind|solar|hydro|nuclear|coal|gas|biomass|geothermal|oil|lcoe|levelized cost|generation|capacity)(?: power)?(?:\s*(?:,|and|or)\s*(wind|solar|hydro|nuclear|coal|gas|biomass|geothermal|oil|lcoe|levelized cost|generation|capacity)(?: power)?)+"
        matches = re.findall(metric_group_pattern, prompt, re.IGNORECASE)
        if matches:
            for tup in matches:
                for m in tup:
                    if m:
                        found_metrics.add(m.lower())
        if re.search(r"wind\s*(,|and|or)\s*solar( power)?", prompt, re.IGNORECASE):
            found_metrics.add("wind")
            found_metrics.add("solar")
        found_metrics = list(found_metrics) if found_metrics else [None]
        
        found_times = []
        for pat in self.time_patterns:
            found_times += re.findall(pat, prompt)
        if not found_times:
            found_times += re.findall(r"\b\d{4}\b", prompt)
        return {
            "countries": found_countries,
            "metrics": found_metrics,
            "time_periods": found_times
        }

    def is_complex(self, prompt: str) -> bool:
        # Rule-based: complex if >1 country, >1 metric, >1 time, or compound/comparison/trend/difference
        entities = self.extract_entities(prompt)
        multi_country = len(entities["countries"]) > 1
        multi_metric = len(entities["metrics"]) > 1
        multi_time = len(entities["time_periods"]) > 1
        compound = bool(re.search(r"\b(compare|difference|trend|analyze|and|or|,|;|all countries|all years|all metrics)\b", prompt, re.IGNORECASE))
        
        # Additional check for comparison keywords
        comparison_keywords = ["compare", "comparison", "versus", "vs", "difference", "between", "among"]
        has_comparison = any(keyword in prompt.lower() for keyword in comparison_keywords)
        
        return multi_country or multi_metric or multi_time or compound or has_comparison

    def decompose(self, prompt: str) -> List[str]:
        # First, check if prompt is complex
        if not self.is_complex(prompt):
            return [prompt]  # Simple prompt, no decomposition needed
        
        # Rule-based decomposition
        entities = self.extract_entities(prompt)
        countries = entities["countries"] or [None]
        metrics = entities["metrics"] or [None]
        times = entities["time_periods"] or [None]
        
        # Deduplicate and prioritize time periods
        range_patterns = [r"from \d{4} to \d{4}", r"between \d{4} and \d{4}", r"\d{4}-\d{4}"]
        ranges = []
        singles = set()
        for t in times:
            if t is None:
                continue
            for pat in range_patterns:
                if re.fullmatch(pat, t):
                    ranges.append(t)
                    break
            else:
                singles.add(t)
        if ranges:
            unique_times = list(dict.fromkeys(ranges))
        else:
            unique_times = list(dict.fromkeys(singles))
        if not unique_times:
            unique_times = [None]
        
        sub_prompts = []
        
        # If ambiguity (e.g., 'all countries'), generate clarification prompt
        if re.search(r"all countries", prompt, re.IGNORECASE):
            return ["Please specify which countries you want to analyze."]
        if re.search(r"all metrics", prompt, re.IGNORECASE):
            return ["Please specify which metrics you want to analyze."]
        
        # Handle single country, single metric, single time - return as-is
        if len(countries) == 1 and len(metrics) == 1 and len(unique_times) == 1:
            return [prompt]
        
        # Cartesian product: country × metric × time
        for country in countries:
            for metric in metrics:
                for time in unique_times:
                    sub = prompt
                    # Replace country group with single country
                    country_group_pattern = r"((?:in|for) [\w\s,]+) from"
                    match = re.search(country_group_pattern, sub, re.IGNORECASE)
                    if match:
                        group_str = match.group(1)
                        sub = sub.replace(group_str, f"in {country}")
                    else:
                        sub = re.sub(r"(in|for) [\w\s,]+", f"in {country}", sub, flags=re.IGNORECASE)
                    
                    # Replace metric group with single metric
                    metric_group_pattern = r"(of [\w\s,]+) in"
                    match_metric = re.search(metric_group_pattern, sub, re.IGNORECASE)
                    if match_metric:
                        metric_str = match_metric.group(1)
                        sub = sub.replace(metric_str, f"of {metric}")
                    else:
                        sub = re.sub(r"(of|for) [\w\s,]+", f"of {metric}", sub, flags=re.IGNORECASE)
                    
                    # Replace time period cleanly
                    if time:
                        sub = re.sub(r"(from|between) [\d\s]+to [\d\s]+", "", sub, flags=re.IGNORECASE)
                        sub = re.sub(r"(from|between) \d{4} and \d{4}", "", sub, flags=re.IGNORECASE)
                        sub = re.sub(r"\d{4}-\d{4}", "", sub)
                        sub = re.sub(r"\s+", " ", sub).strip()
                        sub = f"{sub} {time}" if time not in sub else sub
                    
                    if sub not in sub_prompts:
                        sub_prompts.append(sub)
        
        if len(sub_prompts) == 1:
            return self._llm_decompose(prompt)
        return sub_prompts if sub_prompts else [prompt]

    def _llm_decompose(self, prompt: str) -> List[str]:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that breaks down a complex analysis prompt into a list of atomic, non-overlapping sub-prompts, each focused on a single entity, metric, and time period. Reply with a numbered list."},
                {"role": "user", "content": prompt}
            ]
            
            response_text = self.get_llm_response(messages, max_tokens=256, temperature=0)
            
            if response_text:
                lines = response_text.strip().split('\n')
                sub_prompts = [re.sub(r"^\d+\.\s*", "", line).strip() for line in lines if line.strip()]
                return [sp for sp in sub_prompts if sp]
        except Exception as e:
            print(f"LLM decomposition failed: {e}")
            
        return [prompt]
