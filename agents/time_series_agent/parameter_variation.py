"""
Parameter Variation Module

Models how different parameters vary over time periods using LLM reasoning.
"""

import math
import random
import json
import asyncio
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import openai
from config.settings import OPENAI_API_KEY, OPENAI_MODEL

class VariationType(Enum):
    """Types of parameter variations."""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    RANDOM = "random"
    LEARNING_CURVE = "learning_curve"
    INFLATION = "inflation"
    DEGRADATION = "degradation"

@dataclass
class VariationModel:
    """Model for parameter variation over time."""
    parameter_name: str
    variation_type: VariationType
    base_value: float
    variation_rate: float = 0.0  # Annual rate of change
    start_year: int = 1
    end_year: Optional[int] = None
    custom_function: Optional[Callable] = None
    metadata: Dict = None

class ParameterVariation:
    """Handles parameter variations over time periods using LLM reasoning."""
    
    def __init__(self):
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            self.client = None

    async def determine_variation_with_llm(self, 
                                         parameter_name: str,
                                         base_value: float,
                                         query: str,
                                         time_period: int) -> Dict[str, any]:
        """
        Use LLM to determine parameter variation characteristics.
        
        Args:
            parameter_name: Name of the parameter
            base_value: Initial value of the parameter
            query: Original user query for context
            time_period: Number of years for the analysis
            
        Returns:
            Dictionary with variation type, rate, and reasoning
        """
        if self.client is None:
            return self._get_fallback_variation(parameter_name)
        
        try:
            # Create prompt for LLM
            prompt = f"""
You are an expert energy analyst. Given a parameter and its context, determine how it should vary over time.

Parameter: {parameter_name}
Base Value: {base_value}
Time Period: {time_period} years
Query Context: "{query}"

Based on industry knowledge and the query context, determine:
1. How this parameter typically varies over time in energy projects
2. The appropriate variation type (constant, linear, exponential, learning_curve, inflation, degradation, step)
3. The annual variation rate (as a decimal, e.g., 0.02 for 2% increase, -0.05 for 5% decrease)
4. Brief reasoning for your choice

Consider:
- Technology learning curves for capital costs
- Inflation effects on operational costs
- System degradation for energy production
- Market trends for prices and rates
- Project-specific factors mentioned in the query
- The specific technology or system type mentioned

Return your response as a JSON object with these fields:
- variation_type: string
- variation_rate: float
- reasoning: string
- confidence: float (0.0 to 1.0)

Example response:
{{
    "variation_type": "learning_curve",
    "variation_rate": -0.05,
    "reasoning": "Capital costs typically decrease due to technology learning curves and economies of scale",
    "confidence": 0.85
}}
"""

            # Call OpenAI API
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert energy analyst specializing in dynamic parameter variation modeling. Consider the specific context and technology mentioned in the query."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    # Fallback to minimal default
                    return self._get_fallback_variation(parameter_name)
                
                # Post-process: map unsupported variation types to supported ones
                supported_types = {v.value for v in VariationType}
                vtype = result.get('variation_type', 'constant').lower()
                if vtype not in supported_types:
                    # Map 'seasonal' and other unsupported types to 'constant'
                    vtype = 'constant'
                # Validate and return result
                return {
                    'variation_type': vtype,
                    'variation_rate': float(result.get('variation_rate', 0.0)),
                    'reasoning': result.get('reasoning', 'No reasoning provided'),
                    'confidence': float(result.get('confidence', 0.5))
                }
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse LLM response for {parameter_name}: {e}")
                return self._get_fallback_variation(parameter_name)
                
        except Exception as e:
            print(f"LLM call failed for {parameter_name}: {e}")
            return self._get_fallback_variation(parameter_name)

    def _get_fallback_variation(self, parameter_name: str) -> Dict[str, any]:
        """Get minimal fallback variation when LLM fails."""
        # Very basic fallback - no hardcoded patterns
        return {
            'variation_type': 'constant',
            'variation_rate': 0.0,
            'reasoning': f'Using constant variation as fallback for {parameter_name}',
            'confidence': 0.3
        }

    def create_variation_model(self, 
                             parameter_name: str,
                             base_value: float,
                             variation_type: VariationType,
                             variation_rate: float = 0.0,
                             start_year: int = 1,
                             end_year: Optional[int] = None,
                             custom_function: Optional[Callable] = None) -> VariationModel:
        """
        Create a variation model for a parameter.
        
        Args:
            parameter_name: Name of the parameter
            base_value: Initial value
            variation_type: Type of variation
            variation_rate: Annual rate of change
            start_year: Year to start variation
            end_year: Year to end variation
            custom_function: Custom variation function
            
        Returns:
            VariationModel object
        """
        return VariationModel(
            parameter_name=parameter_name,
            variation_type=variation_type,
            base_value=base_value,
            variation_rate=variation_rate,
            start_year=start_year,
            end_year=end_year,
            custom_function=custom_function,
            metadata={}
        )

    def calculate_variation(self, model: VariationModel, year: int) -> float:
        """
        Calculate parameter value for a specific year.
        
        Args:
            model: Variation model
            year: Year to calculate for
            
        Returns:
            Parameter value for the year
        """
        if model.custom_function:
            return model.custom_function(year, model)
        
        if model.variation_type == VariationType.CONSTANT:
            return model.base_value
        
        elif model.variation_type == VariationType.LINEAR:
            years_since_start = max(0, year - model.start_year + 1)
            return model.base_value * (1 + model.variation_rate * years_since_start)
        
        elif model.variation_type == VariationType.EXPONENTIAL:
            years_since_start = max(0, year - model.start_year + 1)
            return model.base_value * math.pow(1 + model.variation_rate, years_since_start)
        
        elif model.variation_type == VariationType.LEARNING_CURVE:
            years_since_start = max(0, year - model.start_year + 1)
            return model.base_value * math.pow(1 + model.variation_rate, years_since_start)
        
        elif model.variation_type == VariationType.INFLATION:
            years_since_start = max(0, year - model.start_year + 1)
            return model.base_value * math.pow(1 + model.variation_rate, years_since_start)
        
        elif model.variation_type == VariationType.DEGRADATION:
            years_since_start = max(0, year - model.start_year + 1)
            return model.base_value * math.pow(1 + model.variation_rate, years_since_start)
        
        elif model.variation_type == VariationType.STEP:
            if year >= model.start_year:
                return model.base_value * (1 + model.variation_rate)
            return model.base_value
        
        elif model.variation_type == VariationType.RANDOM:
            # Add some randomness to the base variation
            base_variation = self.calculate_variation(
                VariationModel(
                    parameter_name=model.parameter_name,
                    variation_type=VariationType.LINEAR,
                    base_value=model.base_value,
                    variation_rate=model.variation_rate,
                    start_year=model.start_year
                ),
                year
            )
            random_factor = 1 + (random.random() - 0.5) * 0.1  # ±5% randomness
            return base_variation * random_factor
        
        return model.base_value

    async def generate_parameter_series_with_llm(self, 
                                               parameters: Dict[str, float],
                                               query: str,
                                               num_years: int) -> Dict[str, List[float]]:
        """
        Generate parameter values for all years using LLM reasoning.
        
        Args:
            parameters: Initial parameter values
            query: Original user query for context
            num_years: Number of years to generate
            
        Returns:
            Dictionary with parameter names as keys and lists of values as values
        """
        series = {}
        variation_models = []
        
        # Determine variations for each parameter using LLM
        for param_name, base_value in parameters.items():
            print(f"🤖 Analyzing variation for {param_name} using LLM...")
            
            # Get LLM-based variation
            llm_variation = await self.determine_variation_with_llm(
                param_name, base_value, query, num_years
            )
            
            print(f"   📊 {param_name}: {llm_variation['variation_type']} "
                  f"({llm_variation['variation_rate']:.1%} annually)")
            print(f"   💭 Reasoning: {llm_variation['reasoning']}")
            print(f"   🎯 Confidence: {llm_variation['confidence']:.1%}")
            
            # Create variation model
            variation_type = VariationType(llm_variation['variation_type'])
            model = self.create_variation_model(
                parameter_name=param_name,
                base_value=base_value,
                variation_type=variation_type,
                variation_rate=llm_variation['variation_rate']
            )
            model.metadata = {
                'llm_reasoning': llm_variation['reasoning'],
                'confidence': llm_variation['confidence']
            }
            variation_models.append(model)
        
        # Generate values for each year
        for model in variation_models:
            values = []
            for year in range(1, num_years + 1):
                value = self.calculate_variation(model, year)
                values.append(value)
            series[model.parameter_name] = values
        
        return series

    def generate_parameter_series(self, 
                                models: List[VariationModel], 
                                num_years: int) -> Dict[str, List[float]]:
        """
        Generate parameter values for all years.
        
        Args:
            models: List of variation models
            num_years: Number of years to generate
            
        Returns:
            Dictionary with parameter names as keys and lists of values as values
        """
        series = {}
        
        for model in models:
            values = []
            for year in range(1, num_years + 1):
                value = self.calculate_variation(model, year)
                values.append(value)
            series[model.parameter_name] = values
        
        return series

    async def create_models_from_parameters_with_llm(self, 
                                                   parameters: Dict[str, float],
                                                   query: str,
                                                   variation_hints: Dict[str, str] = None) -> List[VariationModel]:
        """
        Create variation models from initial parameters using LLM reasoning.
        
        Args:
            parameters: Initial parameter values
            query: Original user query for context
            variation_hints: Hints about parameter variations
            
        Returns:
            List of VariationModel objects
        """
        models = []
        
        for param_name, base_value in parameters.items():
            # Check for custom hints first
            custom_rate = None
            custom_type = None
            
            if variation_hints and param_name in variation_hints:
                hint = variation_hints[param_name]
                # Extract rate from hint (simplified)
                if '%' in hint:
                    import re
                    rate_match = re.search(r'(\d+)%', hint)
                    if rate_match:
                        custom_rate = float(rate_match.group(1)) / 100
                        if 'decrease' in hint.lower() or 'reduce' in hint.lower():
                            custom_rate = -abs(custom_rate)
                        elif 'increase' in hint.lower() or 'grow' in hint.lower():
                            custom_rate = abs(custom_rate)
            
            if custom_rate is not None:
                # Use custom hint
                variation_type = custom_type or VariationType.LINEAR
                variation_rate = custom_rate
                reasoning = f"User-specified variation: {variation_hints[param_name]}"
                confidence = 0.9
            else:
                # Use LLM reasoning
                llm_variation = await self.determine_variation_with_llm(
                    param_name, base_value, query, len(parameters)
                )
                variation_type = VariationType(llm_variation['variation_type'])
                variation_rate = llm_variation['variation_rate']
                reasoning = llm_variation['reasoning']
                confidence = llm_variation['confidence']
            
            model = self.create_variation_model(
                parameter_name=param_name,
                base_value=base_value,
                variation_type=variation_type,
                variation_rate=variation_rate
            )
            model.metadata = {
                'reasoning': reasoning,
                'confidence': confidence,
                'source': 'llm' if custom_rate is None else 'user_hint'
            }
            
            models.append(model)
        
        return models

    def validate_variation_models(self, models: List[VariationModel]) -> List[str]:
        """
        Validate variation models for potential issues.
        
        Args:
            models: List of variation models
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        for model in models:
            # Check for extreme variations
            if abs(model.variation_rate) > 0.5:  # 50% annual change
                warnings.append(f"High variation rate ({model.variation_rate:.1%}) for {model.parameter_name}")
            
            # Check for negative values
            if model.base_value < 0:
                warnings.append(f"Negative base value for {model.parameter_name}")
            
            # Check for unrealistic learning curves
            if (model.variation_type == VariationType.LEARNING_CURVE and 
                model.variation_rate > 0):
                warnings.append(f"Positive learning curve rate for {model.parameter_name} (should be negative)")
        
        return warnings 