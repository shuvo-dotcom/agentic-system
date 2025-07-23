"""
Scenario Generator Module

Generates multi-year calculation scenarios with parameter variations.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Handle imports for both module and standalone execution
try:
    from .time_detector import TimePeriod
    from .parameter_variation import ParameterVariation, VariationModel
except ImportError:
    from time_detector import TimePeriod
    from parameter_variation import ParameterVariation, VariationModel

@dataclass
class CalculationScenario:
    """Represents a calculation scenario for a specific year."""
    year: int
    parameters: Dict[str, float]
    scenario_id: str
    metadata: Dict[str, Any] = None

@dataclass
class TimeSeriesResult:
    """Represents the complete time series calculation result."""
    scenarios: List[CalculationScenario]
    summary: Dict[str, Any]
    parameter_series: Dict[str, List[float]]
    time_period: TimePeriod
    metadata: Dict[str, Any] = None

class ScenarioGenerator:
    """Generates and manages calculation scenarios for time series analysis."""
    
    def __init__(self):
        self.parameter_variation = ParameterVariation()
        self.scenario_counter = 0

    def generate_scenarios(self, 
                          time_period: TimePeriod,
                          base_parameters: Dict[str, float],
                          variation_hints: Dict[str, str] = None) -> List[CalculationScenario]:
        """
        Generate calculation scenarios for all years.
        
        Args:
            time_period: Time period information
            base_parameters: Initial parameter values
            variation_hints: Hints about parameter variations
            
        Returns:
            List of calculation scenarios
        """
        scenarios = []
        
        # Create variation models
        variation_models = self.parameter_variation.create_models_from_parameters(
            base_parameters, variation_hints
        )
        
        # Generate parameter series
        parameter_series = self.parameter_variation.generate_parameter_series(
            variation_models, time_period.value
        )
        
        # Create scenarios for each year
        for year in range(1, time_period.value + 1):
            year_parameters = {}
            for param_name, values in parameter_series.items():
                year_parameters[param_name] = values[year - 1]
            
            scenario = CalculationScenario(
                year=year,
                parameters=year_parameters,
                scenario_id=f"year_{year}_{self.scenario_counter}",
                metadata={
                    'time_period': time_period,
                    'variation_hints': variation_hints,
                    'base_parameters': base_parameters
                }
            )
            scenarios.append(scenario)
        
        self.scenario_counter += 1
        return scenarios

    async def generate_scenarios_with_llm(self, 
                                        time_period: TimePeriod,
                                        base_parameters: Dict[str, float],
                                        query: str,
                                        variation_hints: Dict[str, str] = None) -> List[CalculationScenario]:
        """
        Generate calculation scenarios for all years using LLM reasoning.
        
        Args:
            time_period: Time period information
            base_parameters: Initial parameter values
            query: Original user query for context
            variation_hints: Hints about parameter variations
            
        Returns:
            List of calculation scenarios
        """
        scenarios = []
        
        print(f"🤖 Using LLM to determine parameter variations for {time_period.value} years...")
        
        # Create variation models using LLM
        variation_models = await self.parameter_variation.create_models_from_parameters_with_llm(
            base_parameters, query, variation_hints
        )
        
        # Generate parameter series using LLM
        parameter_series = await self.parameter_variation.generate_parameter_series_with_llm(
            base_parameters, query, time_period.value
        )
        
        # Create scenarios for each year
        for year in range(1, time_period.value + 1):
            year_parameters = {}
            for param_name, values in parameter_series.items():
                year_parameters[param_name] = values[year - 1]
            
            scenario = CalculationScenario(
                year=year,
                parameters=year_parameters,
                scenario_id=f"year_{year}_{self.scenario_counter}",
                metadata={
                    'time_period': time_period,
                    'variation_hints': variation_hints,
                    'base_parameters': base_parameters,
                    'query': query,
                    'llm_models': [
                        {
                            'parameter': model.parameter_name,
                            'variation_type': model.variation_type.value,
                            'variation_rate': model.variation_rate,
                            'reasoning': model.metadata.get('reasoning', ''),
                            'confidence': model.metadata.get('confidence', 0.5)
                        }
                        for model in variation_models
                    ]
                }
            )
            scenarios.append(scenario)
        
        self.scenario_counter += 1
        return scenarios

    async def execute_scenarios(self, 
                              scenarios: List[CalculationScenario],
                              calculation_function) -> List[Dict[str, Any]]:
        """
        Execute calculations for all scenarios.
        
        Args:
            scenarios: List of calculation scenarios
            calculation_function: Function to execute calculations
            
        Returns:
            List of calculation results
        """
        results = []
        
        for scenario in scenarios:
            try:
                # Execute calculation with scenario parameters and year
                result = await calculation_function(scenario.parameters, scenario.year)
                
                # Add scenario metadata
                result['scenario_id'] = scenario.scenario_id
                result['year'] = scenario.year
                result['parameters'] = scenario.parameters
                
                results.append(result)
                
            except Exception as e:
                # Handle calculation errors
                error_result = {
                    'scenario_id': scenario.scenario_id,
                    'year': scenario.year,
                    'parameters': scenario.parameters,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
        
        return results

    def _create_scenario_query(self, scenario: CalculationScenario) -> str:
        """
        Create a query string for a specific scenario.
        
        Args:
            scenario: Calculation scenario
            
        Returns:
            Query string for the calculation
        """
        # This is a simplified version - in practice, you might want to
        # preserve the original query structure and just update parameters
        param_str = ", ".join([f"{k}: {v}" for k, v in scenario.parameters.items()])
        return f"Calculate with parameters: {param_str}"

    def aggregate_results(self, 
                         results: List[Dict[str, Any]],
                         time_period: TimePeriod) -> TimeSeriesResult:
        """
        Aggregate individual scenario results into a time series result.
        
        Args:
            results: List of calculation results
            time_period: Time period information
            
        Returns:
            TimeSeriesResult object
        """
        # Extract parameter series from results
        parameter_series = {}
        if results and 'parameters' in results[0]:
            param_names = list(results[0]['parameters'].keys())
            for param_name in param_names:
                parameter_series[param_name] = [
                    result.get('parameters', {}).get(param_name, 0) 
                    for result in results
                ]
        
        # Create summary statistics
        summary = self._create_summary(results)
        
        # Create scenarios list
        scenarios = []
        for result in results:
            scenario = CalculationScenario(
                year=result.get('year', 0),
                parameters=result.get('parameters', {}),
                scenario_id=result.get('scenario_id', ''),
                metadata={'result': result}
            )
            scenarios.append(scenario)
        
        return TimeSeriesResult(
            scenarios=scenarios,
            summary=summary,
            parameter_series=parameter_series,
            time_period=time_period,
            metadata={
                'total_scenarios': len(scenarios),
                'successful_scenarios': len([r for r in results if r.get('success', False)]),
                'failed_scenarios': len([r for r in results if not r.get('success', False)]),
                'generated_at': datetime.now().isoformat()
            }
        )

    def _create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary statistics from results.
        
        Args:
            results: List of calculation results
            
        Returns:
            Summary statistics
        """
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {
                'total_results': len(results),
                'successful_results': 0,
                'failed_results': len(results),
                'success_rate': 0.0
            }
        
        # Extract result values
        result_values = []
        for result in successful_results:
            if 'final_result' in result:
                final_result = result['final_result']
                if isinstance(final_result, dict) and 'result' in final_result:
                    result_values.append(final_result['result'])
                elif isinstance(final_result, (int, float)):
                    result_values.append(final_result)
        
        if not result_values:
            return {
                'total_results': len(results),
                'successful_results': len(successful_results),
                'failed_results': len(results) - len(successful_results),
                'success_rate': len(successful_results) / len(results)
            }
        
        # Calculate statistics
        summary = {
            'total_results': len(results),
            'successful_results': len(successful_results),
            'failed_results': len(results) - len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'min_value': min(result_values),
            'max_value': max(result_values),
            'mean_value': sum(result_values) / len(result_values),
            'value_range': max(result_values) - min(result_values)
        }
        
        # Add trend analysis
        if len(result_values) > 1:
            trend = 'increasing' if result_values[-1] > result_values[0] else 'decreasing'
            summary['trend'] = trend
            summary['total_change'] = result_values[-1] - result_values[0]
            summary['percent_change'] = ((result_values[-1] - result_values[0]) / result_values[0]) * 100
        
        return summary

    def export_results(self, 
                      time_series_result: TimeSeriesResult,
                      format: str = 'json') -> str:
        """
        Export time series results to various formats.
        
        Args:
            time_series_result: Time series result to export
            format: Export format ('json', 'csv', 'summary')
            
        Returns:
            Exported data as string
        """
        if format == 'json':
            return json.dumps({
                'time_period': {
                    'value': time_series_result.time_period.value,
                    'unit': time_series_result.time_period.unit,
                    'start_year': time_series_result.time_period.start_year,
                    'end_year': time_series_result.time_period.end_year
                },
                'summary': time_series_result.summary,
                'parameter_series': time_series_result.parameter_series,
                'scenarios': [
                    {
                        'year': s.year,
                        'parameters': s.parameters,
                        'scenario_id': s.scenario_id
                    }
                    for s in time_series_result.scenarios
                ],
                'metadata': time_series_result.metadata
            }, indent=2)
        
        elif format == 'summary':
            summary = time_series_result.summary
            return f"""
Time Series Analysis Summary
============================
Period: {time_series_result.time_period.value} {time_series_result.time_period.unit}
Total Scenarios: {summary.get('total_results', 0)}
Successful: {summary.get('successful_results', 0)}
Failed: {summary.get('failed_results', 0)}
Success Rate: {summary.get('success_rate', 0):.1%}

Value Statistics:
- Min: {summary.get('min_value', 'N/A')}
- Max: {summary.get('max_value', 'N/A')}
- Mean: {summary.get('mean_value', 'N/A')}
- Range: {summary.get('value_range', 'N/A')}

Trend: {summary.get('trend', 'N/A')}
Total Change: {summary.get('total_change', 'N/A')}
Percent Change: {summary.get('percent_change', 'N/A')}%
"""
        
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def validate_scenarios(self, scenarios: List[CalculationScenario]) -> List[str]:
        """
        Validate generated scenarios for potential issues.
        
        Args:
            scenarios: List of calculation scenarios
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        if not scenarios:
            warnings.append("No scenarios generated")
            return warnings
        
        # Check for parameter consistency
        param_names = set(scenarios[0].parameters.keys())
        for scenario in scenarios[1:]:
            if set(scenario.parameters.keys()) != param_names:
                warnings.append(f"Inconsistent parameters in scenario {scenario.scenario_id}")
        
        # Check for extreme parameter values
        for scenario in scenarios:
            for param_name, value in scenario.parameters.items():
                if value < 0 and param_name not in ['discount_rate']:
                    warnings.append(f"Negative value for {param_name} in year {scenario.year}")
                if value > 1e9:  # Very large values
                    warnings.append(f"Very large value for {param_name} in year {scenario.year}")
        
        return warnings 

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze and summarize the results of scenario executions.
        
        Args:
            results: List of scenario results
            
        Returns:
            Summary statistics and analysis
        """
        if not results:
            return {
                'trend': 'no_data',
                'mean': 0,
                'min': 0,
                'max': 0,
                'total_change': 0,
                'percent_change': 0,
                'successful_scenarios': 0,
                'total_scenarios': 0
            }
        
        # Extract successful results
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {
                'trend': 'failed',
                'mean': 0,
                'min': 0,
                'max': 0,
                'total_change': 0,
                'percent_change': 0,
                'successful_scenarios': 0,
                'total_scenarios': len(results)
            }
        
        # Extract values
        values = [r.get('result', 0) for r in successful_results]
        years = [r.get('year', i+1) for i, r in enumerate(successful_results)]
        
        # Calculate statistics
        min_val = min(values)
        max_val = max(values)
        mean_val = sum(values) / len(values)
        
        # Calculate trend
        if len(values) > 1:
            first_val = values[0]
            last_val = values[-1]
            total_change = last_val - first_val
            percent_change = (total_change / first_val) * 100 if first_val != 0 else 0
            
            if total_change > 0:
                trend = 'increasing'
            elif total_change < 0:
                trend = 'decreasing'
            else:
                trend = 'constant'
        else:
            total_change = 0
            percent_change = 0
            trend = 'single_value'
        
        return {
            'trend': trend,
            'mean': mean_val,
            'min': min_val,
            'max': max_val,
            'total_change': total_change,
            'percent_change': percent_change,
            'successful_scenarios': len(successful_results),
            'total_scenarios': len(results),
            'values': values,
            'years': years
        } 