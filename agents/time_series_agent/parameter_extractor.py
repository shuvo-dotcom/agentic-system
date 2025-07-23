"""
Dynamic Parameter Extractor

Extracts parameters from natural language queries using LLM reasoning.
"""

import re
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL

@dataclass
class ExtractedParameter:
    """Represents an extracted parameter with metadata."""
    name: str
    value: float
    unit: str
    confidence: float
    source: str  # 'query', 'llm_inference', 'user_input'
    context: str

class DynamicParameterExtractor:
    """Dynamically extracts parameters from queries using LLM reasoning."""
    
    def __init__(self):
        try:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            self.client = None

    async def extract_parameters_with_llm(self, query: str, calculation_type: str) -> Dict[str, ExtractedParameter]:
        """
        Extract parameters from query using LLM reasoning.
        
        Args:
            query: User query
            calculation_type: Type of calculation being performed
            
        Returns:
            Dictionary of extracted parameters
        """
        if self.client is None:
            return self._extract_parameters_fallback(query)
        
        try:
            prompt = f"""
You are an expert energy analyst. Extract all relevant parameters from the following query for {calculation_type} calculation.

Query: "{query}"

Extract ALL parameters that could be relevant for this calculation. Do not limit yourself to predefined parameter names - identify any numerical values, measurements, or specifications mentioned in the query.

For each parameter you find:
1. Identify the parameter name (use descriptive names)
2. Extract the numerical value
3. Determine the appropriate unit
4. Assess confidence in the extraction (0.0 to 1.0)
5. Provide context about how you identified it

Consider ANY parameters that could be relevant, including but not limited to:
- Financial parameters (costs, prices, rates, investments)
- Technical parameters (capacity, production, efficiency, ratings)
- Temporal parameters (lifetimes, periods, durations)
- Environmental parameters (temperatures, conditions, factors)
- Operational parameters (maintenance, operational costs, performance)

Return your response as a JSON object with this structure:
{{
    "parameters": {{
        "parameter_name": {{
            "value": float,
            "unit": "string",
            "confidence": float,
            "context": "string"
        }}
    }},
    "missing_parameters": ["list of parameters that might be needed but not found"],
    "suggestions": ["suggestions for user to provide missing information"]
}}

Example for "Calculate LCOE for a 2 MW wind turbine with 3000 MWh production over 20 years":
{{
    "parameters": {{
        "rated_capacity": {{
            "value": 2.0,
            "unit": "MW",
            "confidence": 0.95,
            "context": "Explicitly mentioned as '2 MW wind turbine'"
        }},
        "energy_production": {{
            "value": 3000.0,
            "unit": "MWh",
            "confidence": 0.9,
            "context": "Mentioned as '3000 MWh production'"
        }},
        "project_lifetime": {{
            "value": 20.0,
            "unit": "years",
            "confidence": 0.85,
            "context": "Mentioned as 'over 20 years'"
        }}
    }},
    "missing_parameters": ["capital_cost", "om_cost", "discount_rate"],
    "suggestions": ["Please provide capital cost, O&M costs, and discount rate for accurate LCOE calculation"]
}}
"""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert energy analyst specializing in dynamic parameter extraction. Extract ALL relevant parameters from any query without being limited by predefined patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    # Convert to ExtractedParameter objects
                    parameters = {}
                    for param_name, param_data in result.get('parameters', {}).items():
                        parameters[param_name] = ExtractedParameter(
                            name=param_name,
                            value=float(param_data['value']),
                            unit=param_data['unit'],
                            confidence=float(param_data['confidence']),
                            source='llm_inference',
                            context=param_data['context']
                        )
                    
                    return parameters
                    
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Failed to parse LLM response: {e}")
                return self._extract_parameters_fallback(query)
                
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return self._extract_parameters_fallback(query)

    def _extract_parameters_fallback(self, query: str) -> Dict[str, ExtractedParameter]:
        """
        Minimal fallback parameter extraction using basic number detection.
        
        Args:
            query: User query
            
        Returns:
            Dictionary of extracted parameters
        """
        parameters = {}
        
        # Find all numbers in the query
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        
        for i, number in enumerate(numbers):
            try:
                value = float(number)
                
                # Try to determine context from surrounding words
                words = query.lower().split()
                number_idx = -1
                
                # Find the position of this number in the query
                for j, word in enumerate(words):
                    if number in word:
                        number_idx = j
                        break
                
                if number_idx != -1:
                    # Look at surrounding context
                    context_words = words[max(0, number_idx-3):number_idx+4]
                    context = ' '.join(context_words)
                    
                    # Try to infer parameter type from context
                    param_name = f"parameter_{i+1}"
                    unit = "unknown"
                    
                    if any(word in context for word in ['mw', 'megawatt', 'capacity']):
                        param_name = "rated_capacity"
                        unit = "MW"
                    elif any(word in context for word in ['mwh', 'megawatt-hour', 'energy', 'production']):
                        param_name = "energy_production"
                        unit = "MWh"
                    elif any(word in context for word in ['dollar', 'cost', 'investment', 'million', 'thousand']):
                        param_name = "cost_parameter"
                        unit = "dollars"
                    elif any(word in context for word in ['year', 'lifetime', 'period']):
                        param_name = "time_period"
                        unit = "years"
                    elif any(word in context for word in ['percent', '%', 'rate']):
                        param_name = "rate_parameter"
                        unit = "decimal"
                    
                    parameters[param_name] = ExtractedParameter(
                        name=param_name,
                        value=value,
                        unit=unit,
                        confidence=0.3,  # Low confidence for fallback
                        source='fallback',
                        context=f"Extracted from context: {context}"
                    )
                    
            except ValueError:
                continue
        
        return parameters

    async def get_missing_parameters(self, 
                                   extracted_params: Dict[str, ExtractedParameter],
                                   calculation_type: str) -> List[str]:
        """
        Determine which parameters are missing for the calculation using LLM.
        
        Args:
            extracted_params: Already extracted parameters
            calculation_type: Type of calculation
            
        Returns:
            List of missing parameter names
        """
        if self.client is None:
            return self._get_missing_parameters_fallback(extracted_params, calculation_type)
        
        try:
            extracted_names = list(extracted_params.keys())
            
            prompt = f"""
You are an expert energy analyst. Determine what parameters are missing for a {calculation_type} calculation.

Currently extracted parameters: {extracted_names}

For a {calculation_type} calculation, what additional parameters would typically be needed? Consider:
1. What parameters are essential for this calculation type?
2. What parameters would improve accuracy?
3. What parameters are commonly used in industry for this calculation?

Return your response as a JSON object:
{{
    "missing_parameters": ["list of parameter names that are missing"],
    "reasoning": "explanation of why these parameters are needed"
}}

Example for LCOE calculation:
{{
    "missing_parameters": ["capital_cost", "om_cost", "discount_rate"],
    "reasoning": "LCOE requires capital cost, operational costs, and discount rate for accurate calculation"
}}
"""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert energy analyst specializing in parameter requirements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    result = json.loads(json_str)
                    return result.get('missing_parameters', [])
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse LLM missing parameters: {e}")
                
        except Exception as e:
            print(f"LLM missing parameters failed: {e}")
        
        return self._get_missing_parameters_fallback(extracted_params, calculation_type)

    def _get_missing_parameters_fallback(self, 
                                       extracted_params: Dict[str, ExtractedParameter],
                                       calculation_type: str) -> List[str]:
        """Fallback method for determining missing parameters."""
        # Very basic fallback - just suggest common parameters
        common_params = ['capital_cost', 'energy_production', 'rated_capacity', 'om_cost', 'discount_rate']
        extracted_names = set(extracted_params.keys())
        
        return [param for param in common_params if param not in extracted_names]

    async def suggest_default_values(self, 
                                   missing_params: List[str],
                                   calculation_type: str,
                                   context: str) -> Dict[str, Dict[str, Any]]:
        """
        Suggest default values for missing parameters using LLM.
        
        Args:
            missing_params: List of missing parameter names
            calculation_type: Type of calculation
            context: Query context
            
        Returns:
            Dictionary of suggested default values
        """
        if not missing_params or self.client is None:
            return {}
        
        try:
            prompt = f"""
You are an expert energy analyst. Suggest reasonable default values for missing parameters.

Calculation Type: {calculation_type}
Context: "{context}"
Missing Parameters: {', '.join(missing_params)}

For each missing parameter, suggest a reasonable default value based on:
1. The calculation type
2. The context provided
3. Industry standards
4. Typical values for energy projects

Return your response as a JSON object:
{{
    "parameter_name": {{
        "value": float,
        "unit": "string",
        "reasoning": "string",
        "confidence": float
    }}
}}

Example:
{{
    "discount_rate": {{
        "value": 0.08,
        "unit": "decimal",
        "reasoning": "Typical discount rate for energy projects",
        "confidence": 0.7
    }}
}}
"""

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert energy analyst specializing in parameter estimation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse LLM suggestions: {e}")
                
        except Exception as e:
            print(f"LLM suggestion failed: {e}")
        
        return {}

    def convert_to_calculation_format(self, 
                                    extracted_params: Dict[str, ExtractedParameter],
                                    suggested_defaults: Dict[str, Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Convert extracted parameters to the format expected by calculation functions.
        
        Args:
            extracted_params: Extracted parameters
            suggested_defaults: Suggested default values
            
        Returns:
            Dictionary of parameters in calculation format
        """
        calculation_params = {}
        
        # Add extracted parameters
        for param_name, param in extracted_params.items():
            calculation_params[param_name] = param.value
        
        # Add suggested defaults for missing parameters
        if suggested_defaults:
            for param_name, default_data in suggested_defaults.items():
                if param_name not in calculation_params:
                    calculation_params[param_name] = default_data['value']
        
        return calculation_params 