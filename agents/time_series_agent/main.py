"""
Time Series Agent Main Entry Point

This is the independent entry point for the Time Series Agent.
It can run standalone or be called from the central main.py.
"""

import asyncio
import sys
import json
from config.constants import EnergyDefaults, FinancialDefaults, TimeConstants
import logging
from typing import Dict, Any
import argparse
import os

# Add the current directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from time_series_agent import TimeSeriesAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def mock_calculation_function(query: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Mock calculation function for testing.
    In practice, this would be replaced with your actual calculation function.
    
    Args:
        query: Calculation query
        parameters: Parameters for calculation
        
    Returns:
        Mock calculation result
    """
    # Simulate some calculation delay
    await asyncio.sleep(0.1)
    
    # Mock LCOE calculation
    if 'lcoe' in query.lower() or 'levelized' in query.lower():
        capital_cost = parameters.get('capital_cost', FinancialDefaults.DEFAULT_CAPEX_PER_KW * 1000)  # Convert to total project cost
        om_cost = parameters.get('om_cost', FinancialDefaults.DEFAULT_OPEX_PER_KW_YEAR * 1000)
        energy_production = parameters.get('energy_production', EnergyDefaults.DEFAULT_CAPACITY_MW * EnergyDefaults.DEFAULT_LOAD_FACTOR * TimeConstants.HOURS_PER_YEAR)
        discount_rate = parameters.get('discount_rate', FinancialDefaults.DEFAULT_DISCOUNT_RATE)
        project_lifetime = parameters.get('project_lifetime', 25)
        
        # Simple LCOE calculation
        total_cost = capital_cost + (om_cost * project_lifetime)
        total_energy = energy_production * project_lifetime
        
        if total_energy > 0:
            lcoe = total_cost / total_energy
        else:
            lcoe = 0
        
        return {
            'success': True,
            'final_result': {
                'result': lcoe,
                'unit': '$/MWh'
            },
            'calculation_type': 'LCOE',
            'parameters_used': parameters
        }
    
    # Mock capacity factor calculation
    elif 'capacity' in query.lower():
        energy_generated = parameters.get('energy_generated', EnergyDefaults.DEFAULT_CAPACITY_MW * EnergyDefaults.DEFAULT_LOAD_FACTOR * TimeConstants.HOURS_PER_YEAR)
        rated_capacity = parameters.get('rated_capacity', EnergyDefaults.DEFAULT_CAPACITY_MW)
        hours_in_year = parameters.get('hours_in_year', TimeConstants.HOURS_PER_YEAR)
        
        if rated_capacity > 0 and hours_in_year > 0:
            capacity_factor = energy_generated / (rated_capacity * hours_in_year)
        else:
            capacity_factor = 0
        
        return {
            'success': True,
            'final_result': {
                'result': capacity_factor,
                'unit': 'dimensionless'
            },
            'calculation_type': 'Capacity Factor',
            'parameters_used': parameters
        }
    
    # Default mock result
    else:
        return {
            'success': True,
            'final_result': {
                'result': 42.0,
                'unit': 'units'
            },
            'calculation_type': 'Generic',
            'parameters_used': parameters
        }

async def main():
    """Main function for the Time Series Agent."""
    parser = argparse.ArgumentParser(description="Time Series Analysis Agent")
    parser.add_argument('--query', type=str, help='Natural language query to analyze')
    parser.add_argument('--context', type=str, help='JSON context for the analysis')
    parser.add_argument('--info', action='store_true', help='Show agent information')
    parser.add_argument('--test', action='store_true', help='Run test scenarios')
    parser.add_argument('--format', type=str, default='json', choices=['json', 'summary'], 
                       help='Output format')
    
    args = parser.parse_args()
    
    # Initialize the agent
    agent = TimeSeriesAgent()
    
    if args.info:
        # Show agent information
        info = agent.get_agent_info()
        print("🤖 Time Series Agent Information")
        print("=" * 50)
        print(f"Name: {info['name']}")
        print(f"Description: {info['description']}")
        print("\nCapabilities:")
        for capability in info['capabilities']:
            print(f"  • {capability}")
        print("\nSupported Variations:")
        for variation in info['supported_variations']:
            print(f"  • {variation}")
        print("\nDefault Parameter Variations:")
        for param, desc in info['default_parameters'].items():
            print(f"  • {param}: {desc}")
        return
    
    if args.test:
        # Run test scenarios
        await run_test_scenarios(agent)
        return
    
    if not args.query:
        print("❌ Error: You must provide a query with --query")
        print("\nExample usage:")
        print("  python main.py --query 'Calculate LCOE for next 10 years'")
        print("  python main.py --query 'Analyze capacity factor over 5 years'")
        print("  python main.py --info")
        print("  python main.py --test")
        return
    
    # Parse context if provided
    context = None
    if args.context:
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError:
            print("❌ Error: Invalid JSON context provided")
            return
    
    # Run the analysis
    print("🚀 Starting Time Series Analysis...")
    print(f"Query: {args.query}")
    print("-" * 50)
    
    try:
        result = await agent.analyze_query(
            query=args.query,
            calculation_function=mock_calculation_function,
            context=context
        )
        
        # Display results
        if result.get('success'):
            print("✅ Time Series Analysis Completed Successfully!")
            
            if args.format == 'summary':
                summary = await agent.export_analysis(result, 'summary')
                print(summary)
            else:
                # JSON format
                print(f"⏱️  Execution Time: {result.get('execution_time', 0):.2f} seconds")
                print(f"📊 Analysis Type: {result.get('analysis_type', 'N/A')}")
                
                if result.get('time_series'):
                    metadata = result.get('metadata', {})
                    print(f"📈 Scenarios: {metadata.get('total_scenarios', 0)} total, "
                          f"{metadata.get('successful_scenarios', 0)} successful")
                    
                    summary = result.get('summary', {})
                    if summary:
                        print(f"📊 Results Summary:")
                        print(f"   Min: {summary.get('min_value', 'N/A')}")
                        print(f"   Max: {summary.get('max_value', 'N/A')}")
                        print(f"   Mean: {summary.get('mean_value', 'N/A')}")
                        print(f"   Trend: {summary.get('trend', 'N/A')}")
                        print(f"   Change: {summary.get('percent_change', 'N/A')}%")
                else:
                    # Single period result
                    final_result = result.get('final_result', {})
                    if final_result:
                        print(f"🎯 Result: {final_result.get('result', 'N/A')}")
                        if 'unit' in final_result:
                            print(f"   Unit: {final_result['unit']}")
            
            # Show validation warnings if any
            warnings = result.get('validation_warnings', [])
            if warnings:
                print(f"\n⚠️  Validation Warnings:")
                for warning in warnings:
                    print(f"   • {warning}")
        
        else:
            print("❌ Time Series Analysis Failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

async def run_test_scenarios(agent: TimeSeriesAgent):
    """Run test scenarios to demonstrate the agent's capabilities."""
    print("🧪 Running Test Scenarios...")
    print("=" * 50)
    
    test_queries = [
        {
            'query': 'Calculate LCOE for next 10 years with capital cost 2 million dollars',
            'description': 'Multi-year LCOE with learning curve effect'
        },
        {
            'query': 'Analyze capacity factor over 5 years with degradation',
            'description': 'Capacity factor with system degradation'
        },
        {
            'query': 'Calculate simple payback period',
            'description': 'Single period calculation (should delegate)'
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}: {test['description']}")
        print(f"Query: {test['query']}")
        print("-" * 30)
        
        try:
            result = await agent.analyze_query(
                query=test['query'],
                calculation_function=mock_calculation_function
            )
            
            if result.get('success'):
                print("✅ Success")
                if result.get('time_series'):
                    metadata = result.get('metadata', {})
                    print(f"   Scenarios: {metadata.get('total_scenarios', 0)}")
                    print(f"   Type: {result.get('analysis_type', 'N/A')}")
                else:
                    print("   Type: Single period")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print(f"\n🎉 Test scenarios completed!")

if __name__ == "__main__":
    asyncio.run(main()) 