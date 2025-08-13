"""
Dynamic Time Series Agent

Handles multi-period calculations with intelligent parameter variation modeling.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .time_detector import TimeDetector, TimePeriod, ForecastingRequest, TimeGranularity
from .parameter_extractor import DynamicParameterExtractor
from .parameter_variation import VariationModel
from config.settings import OPENAI_MODEL
from .parameter_extractor import DynamicParameterExtractor, ExtractedParameter
from .parameter_variation import ParameterVariation
from .scenario_generator import ScenarioGenerator
from config.settings import OPENAI_MODEL
from config.constants import TimeConstants

@dataclass
class AnalysisResult:
    """Result of time series analysis."""
    scenarios: List[Dict]
    summary: Dict[str, Any]
    parameter_variations: Dict[str, Any]
    user_preferences: Dict[str, Any]
    execution_time: float

class DynamicTimeSeriesAgent:
    """Dynamic time series analysis agent with interactive parameter handling."""
    
    def __init__(self):
        self.time_detector = TimeDetector()
        self.parameter_extractor = DynamicParameterExtractor()
        self.parameter_variation = ParameterVariation()
        self.scenario_generator = ScenarioGenerator()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def analyze_query(self, query: str, user_responses: Dict[str, str] = None, session_id: int = None, agent_name: str = 'DynamicTimeSeriesAgent', call_tree=None, get_node_id=None) -> AnalysisResult:
        """
        Analyze a query with dynamic parameter extraction and user interaction.
        
        Args:
            query: User query
            user_responses: User responses to questions (if already provided)
            session_id: Session identifier for logging
            agent_name: Name of the agent for logging
        Returns:
            AnalysisResult object
        """
        # Add session_id and agent_name to all log records
        class ContextFilter(logging.Filter):
            def filter(self, record):
                record.session_id = session_id
                record.agent_name = agent_name
                return True
        self.logger.addFilter(ContextFilter())

        self.logger.info(f"Starting DynamicTimeSeriesAgent analysis for query: {query}", extra={'session_id': session_id, 'agent_name': agent_name})
        print(f"🚀 Starting Dynamic Analysis...")
        print(f"Query: {query}")
        print("-" * 50)
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Step 1: Detect time period and calculation type
            time_period = self.time_detector.detect_time_period(query)
            calculation_type = self.time_detector.extract_calculation_type(query)
            self.logger.info(f"Detected calculation_type: {calculation_type}", extra={'session_id': session_id, 'agent_name': agent_name})
            
            print(f"📊 Calculation Type: {calculation_type}")
            
            # Handle user-provided time period
            if not time_period and user_responses and 'time_period' in user_responses:
                time_value = int(user_responses['time_period'])
                granularity_str = user_responses.get('granularity', 'years').lower()
                
                granularity_map = {
                    'years': TimeGranularity.YEARS,
                    'months': TimeGranularity.MONTHS,
                    'days': TimeGranularity.DAYS
                }
                
                granularity = granularity_map.get(granularity_str, TimeGranularity.YEARS)
                
                time_period = TimePeriod(
                    value=time_value,
                    granularity=granularity,
                    is_forecast_requested=True,
                    confidence=0.95
                )
                print(f"⏰ Using user-provided time period: {time_period.value} {time_period.granularity.value}")
            
            if time_period:
                print(f"⏰ Time Period Detected: {time_period.value} {time_period.granularity.value}")
                print(f"🎯 Confidence: {time_period.confidence:.1%}")
            else:
                print("❌ No time period detected - this may be a single-period calculation")
                # Proceed as single-period calculation (legacy behavior)
                # time_period remains None
            
            # Step 2: Extract parameters dynamically
            print(f"\n🔍 Extracting parameters from query...")
            extracted_params = await self.parameter_extractor.extract_parameters_with_llm(
                query, calculation_type
            )
            
            # Add parameter extraction nodes
            if call_tree and get_node_id:
                for param_name, param in extracted_params.items():
                    param_node_id = get_node_id('param')
                    call_tree[param_node_id] = {
                        'agent_name': agent_name,
                        'operation': 'param_extract',
                        'parent_id': None, # Root node for parameter extraction
                        'parameters': {'param': param_name, 'value': param.value, 'unit': param.unit, 'confidence': param.confidence, 'source': getattr(param, 'source', 'unknown'), 'context': getattr(param, 'context', '')},
                        'result': None
                    }
            print(f"📋 Extracted Parameters:")
            for param_name, param in extracted_params.items():
                print(f"   {param_name}: {param.value} {param.unit} (confidence: {param.confidence:.1%})")
            
            # Step 3: Check for missing parameters
            missing_params = await self.parameter_extractor.get_missing_parameters(
                extracted_params, calculation_type
            )
            
            suggested_defaults = {}
            if missing_params:
                print(f"\n⚠️  Missing Parameters: {', '.join(missing_params)}")
                
                # Get user responses if not provided
                if user_responses is None:
                    # First, try to get LLM suggestions
                    suggested_defaults = await self.parameter_extractor.suggest_default_values(
                        missing_params, calculation_type, query
                    )
                    
                    # Add LLM suggestion nodes
                    if call_tree and get_node_id and suggested_defaults:
                        for param_name, default_data in suggested_defaults.items():
                            llm_node_id = get_node_id('llm_suggest')
                            call_tree[llm_node_id] = {
                                'agent_name': agent_name,
                                'operation': 'llm_suggest_default',
                                'parent_id': None, # Root node for LLM suggestions
                                'parameters': {'param': param_name, **default_data},
                                'result': None
                            }
                    if suggested_defaults:
                        print(f"\n💡 LLM Suggested Default Values:")
                        for param_name, default_data in suggested_defaults.items():
                            print(f"   {param_name}: {default_data['value']} {default_data['unit']}")
                            print(f"      Reasoning: {default_data['reasoning']}")
                            print(f"      Confidence: {default_data['confidence']:.1%}")
                        
                        # Ask user if they want to use defaults or provide custom values
                        # For non-interactive mode (when called by orchestrator), automatically use defaults
                        try:
                            use_defaults = input("\n🤔 Use LLM-suggested defaults? (yes/no): ").lower().strip()
                        except EOFError:
                            # Non-interactive mode, automatically use defaults
                            use_defaults = 'yes'
                            print("\n🤖 Running in non-interactive mode - automatically using LLM-suggested defaults")
                        
                        if call_tree and get_node_id:
                            user_choice_node_id = get_node_id('user_choice')
                            call_tree[user_choice_node_id] = {
                                'agent_name': agent_name,
                                'operation': 'user_choice_llm_defaults',
                                'parent_id': None, # Root node for user choice
                                'parameters': {'use_defaults': use_defaults},
                                'result': None
                            }
                        if use_defaults == 'yes':
                            print("✅ Using LLM-suggested defaults")
                            user_responses = {'use_llm_defaults': True}
                        else:
                            # Get custom user input
                            user_responses = await self._get_user_input(
                                time_period, calculation_type, extracted_params, missing_params, call_tree, get_node_id, None
                            )
                    else:
                        # No LLM suggestions available, get user input
                        user_responses = await self._get_user_input(
                            time_period, calculation_type, extracted_params, missing_params, call_tree, get_node_id, None
                        )
                
                # If user didn't choose to use LLM defaults, get suggestions again
                if not user_responses.get('use_llm_defaults', False):
                    # Only get new suggestions if we don't already have them
                    if not suggested_defaults:
                        suggested_defaults = await self.parameter_extractor.suggest_default_values(
                            missing_params, calculation_type, query
                        )
                        
                        # Add LLM suggestion nodes
                        if call_tree and get_node_id and suggested_defaults:
                            for param_name, default_data in suggested_defaults.items():
                                llm_node_id = get_node_id('llm_suggest')
                                call_tree[llm_node_id] = {
                                    'agent_name': agent_name,
                                    'operation': 'llm_suggest_default',
                                    'parent_id': None, # Root node for LLM suggestions
                                    'parameters': {'param': param_name, **default_data},
                                    'result': None
                                }
                        if suggested_defaults:
                            print(f"\n💡 Using Suggested Default Values:")
                            for param_name, default_data in suggested_defaults.items():
                                print(f"   {param_name}: {default_data['value']} {default_data['unit']}")
                                print(f"      Reasoning: {default_data['reasoning']}")
                                print(f"      Confidence: {default_data['confidence']:.1%}")
                    else:
                        # We already have suggestions, just use them
                        print(f"\n✅ Using previously suggested default values.")
            
            # Step 4: Convert to calculation format with intelligent normalization
            calculation_params = await self.parameter_extractor.convert_to_calculation_format_with_normalization(
                extracted_params, 
                suggested_defaults if 'suggested_defaults' in locals() else None,
                query
            )
            
            print(f"\n📊 Final Parameters for Calculation:")
            for param_name, value in calculation_params.items():
                print(f"   {param_name}: {value}")
            
            # Step 5: Generate scenarios with LLM-based variation
            print(f"\n🤖 Generating scenarios with LLM-based parameter variation...")
            # If time_period is None, create a default single-period TimePeriod
            if time_period is None:
                # Try to get from calculation_params or user_responses
                period_value = None
                if "time_period" in calculation_params:
                    period_value = calculation_params["time_period"]
                elif user_responses and "time_period" in user_responses:
                    try:
                        period_value = int(user_responses["time_period"])
                    except Exception:
                        period_value = 1
                if period_value is None:
                    period_value = 1
                time_period = TimePeriod(
                    value=int(period_value),
                    granularity=TimeGranularity.YEARS,
                    is_forecast_requested=False,
                    confidence=1.0
                )
            scenarios = await self.scenario_generator.generate_scenarios_with_llm(
                time_period, calculation_params, query, user_responses
            )
            
            # Step 6: Execute scenarios with mock calculation function
            print(f"\n⚡ Executing {len(scenarios)} scenarios...")
            
            # Truly dynamic calculation function
            async def dynamic_calculation_function(scenario_params: Dict[str, float], year: int) -> Dict[str, Any]:
                """Fully dynamic calculation function that adapts to any parameter combination."""
                print(f"🔧 Calculating for year {year} with params: {scenario_params}")
                
                # Use LLM to determine what calculation to perform based on available parameters
                try:
                    query_safe = query.replace('"', "'")  # Replace quotes to avoid string issues
                    calc_prompt = f"""
You are an expert energy analyst. The user's original query was: "{query_safe}"

Given these parameters, determine what calculation should be performed and calculate the result.

Parameters: {scenario_params}
Year: {year}

IMPORTANT: The calculation should match the user's intent from their original query. If they asked for:
- "efficiency" → Calculate efficiency metrics (thermal efficiency, electrical efficiency, overall efficiency)
- "capacity factor" → Calculate capacity factor
- "LCOE" or "cost" → Calculate Levelized Cost of Energy
- "NPV" or "financial" → Calculate Net Present Value
- "payback" → Calculate payback period
- "IRR" → Calculate Internal Rate of Return

Available calculation types:
- Efficiency: Requires power_output, electrical_efficiency, thermal_efficiency, or similar efficiency parameters
- Capacity Factor: Requires rated_capacity and energy_production (or degradation rate)
- LCOE (Levelized Cost of Energy): Requires capital_cost, om_cost, energy_production, discount_rate, project_lifetime
- NPV (Net Present Value): Requires capital_cost, energy_production, discount_rate, project_lifetime, electricity_price
- IRR (Internal Rate of Return): Requires capital_cost, energy_production, project_lifetime, electricity_price
- Payback Period: Requires capital_cost, energy_production, electricity_price, om_cost

Return your response as JSON:
{{
    "calculation_type": "string",
    "result": float,
    "unit": "string",
    "reasoning": "string",
    "confidence": float
}}

Example for efficiency calculation:
{{
    "calculation_type": "efficiency",
    "result": 85.5,
    "unit": "%",
    "reasoning": "Calculated overall efficiency based on electrical and thermal efficiency parameters",
    "confidence": 0.9
}}

Example for LCOE calculation:
{{
    "calculation_type": "lcoe",
    "result": 67.50,
    "unit": "USD/MWh",
    "reasoning": "Calculated using (Capital Cost + O&M Cost * Project Lifetime) / (Energy Production * Project Lifetime)",
    "confidence": 0.9
}}
"""

                    from utils.llm_provider import get_llm_response
                    
                    messages = [
                        {"role": "system", "content": "You are an expert energy analyst specializing in dynamic calculations. Always match the user's intent from their original query."},
                        {"role": "user", "content": calc_prompt}
                    ]
                    
                    response_text = await asyncio.to_thread(
                        get_llm_response,
                        messages,
                        temperature=0.1,
                        max_tokens=300
                    )
                    
                    content = response_text.strip()
                    
                    # Extract JSON from response
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx != 0:
                        json_str = content[start_idx:end_idx]
                        result = json.loads(json_str)
                        
                        print(f"   📊 {result['calculation_type'].upper()} calculated: {result['result']} {result['unit']}")
                        print(f"   💭 {result['reasoning']}")
                        
                        return {
                            'success': True,
                            'result': float(result['result']),
                            'unit': result['unit'],
                            'year': year,
                            'parameters': scenario_params,
                            'calculation_type': result['calculation_type'],
                            'reasoning': result['reasoning'],
                            'confidence': float(result['confidence'])
                        }
                    else:
                        print(f"   ⚠️  Failed to parse LLM response: {content}")
                
                except Exception as e:
                    print(f"   ⚠️  LLM calculation failed: {e}")
                
                # Fallback to simple calculations
                return await self._fallback_calculation(scenario_params, year)

            async def _fallback_calculation(scenario_params: Dict[str, float], year: int) -> Dict[str, Any]:
                """Fallback calculation when LLM is not available."""
                # Simple capacity factor if rated_capacity is available
                if 'rated_capacity' in scenario_params:
                    rated_capacity = scenario_params['rated_capacity']
                    if 'energy_production' in scenario_params:
                        energy_production = scenario_params['energy_production']
                        capacity_factor = (energy_production / (rated_capacity * TimeConstants.HOURS_PER_YEAR)) * 100
                        return {
                            'success': True,
                            'result': capacity_factor,
                            'unit': '%',
                            'year': year,
                            'parameters': scenario_params,
                            'calculation_type': 'capacity_factor'
                        }
                    elif 'energy_production_degradation_rate' in scenario_params:
                        degradation_rate = scenario_params['energy_production_degradation_rate'] / 100
                        base_energy = rated_capacity * TimeConstants.HOURS_PER_YEAR
                        degraded_energy = base_energy * (1 - degradation_rate)
                        capacity_factor = (degraded_energy / (rated_capacity * TimeConstants.HOURS_PER_YEAR)) * 100
                        return {
                            'success': True,
                            'result': capacity_factor,
                            'unit': '%',
                            'year': year,
                            'parameters': scenario_params,
                            'calculation_type': 'capacity_factor'
                        }
                
                # Simple cost analysis if capital_cost is available
                if 'capital_cost' in scenario_params:
                    capital_cost = scenario_params['capital_cost']
                    return {
                        'success': True,
                        'result': capital_cost,
                        'unit': 'USD',
                        'year': year,
                        'parameters': scenario_params,
                        'calculation_type': 'cost_analysis'
                    }
                
                return {
                    'success': False,
                    'error': 'No suitable calculation method found for given parameters',
                    'year': year,
                    'parameters': scenario_params
                }
            
            results = await self.scenario_generator.execute_scenarios(scenarios, dynamic_calculation_function)
            
            # Step 7: Aggregate and analyze results
            print(f"\n📈 Analyzing results...")
            summary = self.scenario_generator.analyze_results(results)
            
            # Step 8: Prepare analysis result
            execution_time = asyncio.get_event_loop().time() - start_time
            
            analysis_result = AnalysisResult(
                scenarios=results,
                summary=summary,
                parameter_variations={
                    'extracted_params': extracted_params,
                    'suggested_defaults': suggested_defaults if 'suggested_defaults' in locals() else {},
                    'calculation_params': calculation_params
                },
                user_preferences=user_responses or {},
                execution_time=execution_time
            )
            
            print(f"\n✅ Dynamic Analysis Completed Successfully!")
            print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", extra={'session_id': session_id, 'agent_name': agent_name})
            print(f"❌ Analysis failed: {e}")
            return None

    async def _get_user_input(self, 
                             time_period: TimePeriod,
                             calculation_type: str,
                             extracted_params: Dict[str, ExtractedParameter],
                             missing_params: List[str],
                             call_tree=None,
                             get_node_id=None,
                             parent_id=None) -> Dict[str, str]:
        """
        Get user input for missing information.
        
        Args:
            time_period: Detected time period
            calculation_type: Type of calculation
            extracted_params: Already extracted parameters
            missing_params: Missing parameters
            
        Returns:
            Dictionary of user responses
        """
        print(f"\n🤔 Need additional information for accurate analysis...")
        
        user_responses = {}
        
        # Ask for time period if not specified
        if time_period is None or getattr(time_period, "value", 0) == 0:
            question = "How many years would you like to forecast? (e.g., 5, 10, 20): "
            time_input = input(question)
            user_responses['time_period'] = time_input
            if time_period is not None:
                time_period.value = int(time_input)
            if call_tree and get_node_id:
                node_id = get_node_id('user_input')
                call_tree[node_id] = {
                    'agent_name': 'User',
                    'operation': 'user_input',
                    'parent_id': parent_id,
                    'parameters': {'question': question, 'response': time_input},
                    'result': None
                }
        
        # Ask for time granularity
        question = "What time granularity would you prefer? (years/months/days): "
        granularity_input = input(question)
        user_responses['granularity'] = granularity_input
        if call_tree and get_node_id:
            node_id = get_node_id('user_input')
            call_tree[node_id] = {
                'agent_name': 'User',
                'operation': 'user_input',
                'parent_id': parent_id,
                'parameters': {'question': question, 'response': granularity_input},
                'result': None
            }
        
        # Ask about parameter variations
        question = "Would you like me to model parameter variations over time? (yes/no): "
        variation_input = input(question)
        user_responses['model_variations'] = variation_input
        if call_tree and get_node_id:
            node_id = get_node_id('user_input')
            call_tree[node_id] = {
                'agent_name': 'User',
                'operation': 'user_input',
                'parent_id': parent_id,
                'parameters': {'question': question, 'response': variation_input},
                'result': None
            }
        
        if variation_input.lower() == 'yes':
            question = "Any specific parameter trends to consider? (e.g., 'capital costs decreasing', 'energy prices increasing'): "
            trends_input = input(question)
            user_responses['parameter_trends'] = trends_input
            if call_tree and get_node_id:
                node_id = get_node_id('user_input')
                call_tree[node_id] = {
                    'agent_name': 'User',
                    'operation': 'user_input',
                    'parent_id': parent_id,
                    'parameters': {'question': question, 'response': trends_input},
                    'result': None
                }
        
        # Ask about missing parameters
        if missing_params:
            print(f"\nMissing parameters: {', '.join(missing_params)}")
            print("I can suggest default values, or you can provide specific values.")
            
            for param in missing_params:
                question = f"Value for {param} (or press Enter for default): "
                param_input = input(question)
                if param_input.strip():
                    user_responses[f'param_{param}'] = param_input
                if call_tree and get_node_id:
                    node_id = get_node_id('user_input')
                    call_tree[node_id] = {
                        'agent_name': 'User',
                        'operation': 'user_input',
                        'parent_id': parent_id,
                        'parameters': {'question': question, 'response': param_input},
                        'result': None
                    }
        
        # Ask about output format
        question = "Do you want year-by-year results or summary statistics? (year_by_year/summary): "
        output_input = input(question)
        user_responses['output_format'] = output_input
        if call_tree and get_node_id:
            node_id = get_node_id('user_input')
            call_tree[node_id] = {
                'agent_name': 'User',
                'operation': 'user_input',
                'parent_id': parent_id,
                'parameters': {'question': question, 'response': output_input},
                'result': None
            }
        return user_responses

    async def process_interactive_query(self, initial_query: str) -> AnalysisResult:
        """
        Process a query with full interactive parameter handling.
        
        Args:
            initial_query: Initial user query
            
        Returns:
            AnalysisResult object
        """
        # First pass: extract what we can
        time_period = self.time_detector.detect_time_period(initial_query)
        calculation_type = self.time_detector.extract_calculation_type(initial_query)
        
        print(f"📊 Calculation Type: {calculation_type}")
        
        user_responses = {}
        
        if not time_period:
            print("❌ No time period detected in query.")
            print("🤔 Would you like to do a time series analysis?")
            
            # Ask if user wants forecasting
            forecasting_choice = input("Do you want to forecast over time? (yes/no): ")
            
            if forecasting_choice.lower() == 'yes':
                # Create a time period for forecasting
                time_input = input("How many years would you like to forecast? (e.g., 5, 10, 20): ")
                user_responses['time_period'] = time_input
                user_responses['granularity'] = 'years'
                user_responses['model_variations'] = 'yes'
                user_responses['output_format'] = 'year_by_year'
                
                print(f"✅ Time period set to {time_input} years")
            else:
                print("💡 This appears to be a single-period calculation.")
                print("💡 Try the regular calculation system for single-period analysis.")
                return None
        
        # Check if user input is needed
        if (time_period and self.time_detector.needs_user_input(time_period)) or not time_period:
            print("🤔 I need some additional information to provide accurate analysis...")
            
            # Generate questions
            questions = self.time_detector.generate_forecasting_questions(
                time_period, calculation_type, {}
            )
            
            print("\n📝 Please answer the following questions:")
            
            for i, question in enumerate(questions, 1):
                print(f"\n{i}. {question}")
                response = input("Your answer: ")
                user_responses[f'q{i}'] = response
            
            # Parse user preferences
            forecasting_request = self.time_detector.parse_user_preferences(
                user_responses, initial_query
            )
        
        # Now run the analysis with user preferences
        return await self.analyze_query(initial_query, user_responses) 

    def log_decision_point(self, session_id, agent_name, step_id, parent_step_id, decision_type, reason, result_summary, sub_prompt=None, stage='decision_point', confidence=None, parameters=None, llm_rationale=None, extra_context=None):
        log_data = {
            'session_id': session_id,
            'agent_name': agent_name,
            'stage': stage,
            'step_id': step_id,
            'parent_step_id': parent_step_id,
            'decision_type': decision_type,
            'decision_reason': reason,
            'decision_result': result_summary,
            'sub_prompt': sub_prompt,
            'confidence': confidence,
            'parameters': parameters,
            'llm_rationale': llm_rationale,
        }
        if extra_context:
            log_data.update(extra_context)
        self.logger.info(
            f"Decision: {decision_type} | Reason: {reason} | Result: {result_summary} | Confidence: {confidence} | Parameters: {parameters} | LLM Rationale: {llm_rationale}",
            extra=log_data
        )

    # After main analysis, log accept or retry/fallback as appropriate
    # Example: Accept result
    def log_decision_point(self, session_id, agent_name, step_id, parent_step_id, decision_type, reason, result_summary, sub_prompt=None, stage='decision_point', confidence=None, parameters=None, llm_rationale=None, extra_context=None):
        log_data = {
            'session_id': session_id,
            'agent_name': agent_name,
            'stage': stage,
            'step_id': step_id,
            'parent_step_id': parent_step_id,
            'decision_type': decision_type,
            'decision_reason': reason,
            'decision_result': result_summary,
            'sub_prompt': sub_prompt,
            'confidence': confidence,
            'parameters': parameters,
            'llm_rationale': llm_rationale,
        }
        if extra_context:
            log_data.update(extra_context)
        self.logger.info(
            f"Decision: {decision_type} | Reason: {reason} | Result: {result_summary} | Confidence: {confidence} | Parameters: {parameters} | LLM Rationale: {llm_rationale}",
            extra=log_data
        )
