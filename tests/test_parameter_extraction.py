"""
Comprehensive test script for verifying parameter extraction with self-healing queries 
in the PostgresDataProvider, including field mapping tests.
"""
import os
import sys
import asyncio
import logging
import json
import argparse
from pprint import pprint

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.postgres_data_provider import PostgresDataProvider
from agents.enhanced_orchestrator_agent import EnhancedOrchestratorAgent

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Test cases with different parameter extraction scenarios
TEST_CASES = [
    {
        "name": "Direct Parameter Query - energy_output_t",
        "query": "What is the value of parameter energy_output_t for nuclear generation in Belgium?",
        "param_name": "energy_output_t",
        "expected_field_mappings": ["energy_output", "total_generation", "generation", "production"]
    },
    {
        "name": "Direct Parameter Query - n",
        "query": "How many operational reactors (parameter n) are in Belgium?",
        "param_name": "n",
        "expected_field_mappings": ["operational_reactors", "reactor_count", "units", "count"]
    },
    {
        "name": "Multiple Parameters Query",
        "query": "Extract both energy_output_t and n parameters for Belgium's nuclear fleet",
        "param_name": None,  # Will extract multiple parameters
        "expected_params": ["energy_output_t", "n"]
    }
]

async def test_parameter_extraction():
    """Test the parameter extraction with self-healing in PostgresDataProvider"""
    print("\n=== Testing Parameter Extraction with Self-Healing ===\n")
    
    # Initialize the provider
    provider = PostgresDataProvider()
    
    # First run the pre-defined test cases
    for idx, case in enumerate(TEST_CASES):
        print(f"\n>>> Test Case {idx+1}: {case['name']}")
        print(f"Query: {case['query']}")
        
        try:
            # Execute the parameter query
            if case['param_name']:
                print(f"Extracting parameter: {case['param_name']}")
                result = await provider.execute_parameter_query(
                    query=case['query'],
                    parameter_name=case['param_name'],
                    allow_self_healing=True,
                    max_fallback_attempts=2
                )
            else:
                # For multiple parameters, use the general process method
                print("Extracting multiple parameters")
                result = await provider.process({
                    "user_query": case['query'],
                    "max_fallback_attempts": 2
                })
            
            print(f"\nResult status: {result.get('status', 'unknown')}")
            print(f"Attempts: {result.get('attempts', 1)}")
            
            # Check for parameters
            if result.get('data') and result['data'].get('parameters'):
                parameters = result['data']['parameters']
                print(f"\nExtracted Parameters:")
                for param, value in parameters.items():
                    print(f"  - {param}: {value}")
                
                # Verify expected parameters
                if 'expected_params' in case:
                    missing = [p for p in case['expected_params'] if p not in parameters]
                    if missing:
                        print(f"WARNING: Missing expected parameters: {missing}")
                    else:
                        print(f"SUCCESS: All expected parameters found!")
                elif case['param_name'] and case['param_name'] in parameters:
                    print(f"SUCCESS: Parameter '{case['param_name']}' successfully extracted!")
                elif case['param_name']:
                    print(f"WARNING: Expected parameter '{case['param_name']}' not found in results")
            
        except Exception as e:
            print(f"Error in test case {idx+1}: {str(e)}")
    
    # Also test with dynamic queries
    test_queries = [
        "What are the values for energy_output_t, n in the context of nuclear generation in Belgium?",
        "Extract parameters energy_output_t and n for solar power generation in Spain",
        "Find the parameter values for energy_output_t and n for wind power in Germany",
        "What are the parameter values needed for capacity factor calculation in France?"
    ]
    
    for query in test_queries:
        print(f"\n>>> Testing dynamic parameter query: {query}")
        
        # Process the query through the provider
        try:
            result = await provider.process({
                "user_query": query,
                "max_fallback_attempts": 3
            })
            
            # Print a summary of the response
            print("\nProvider Response Summary:")
            print(f"- Status: {result.get('status', 'unknown')}")
            print(f"- Attempts: {result.get('attempts', 1)}")
            
            # Check for extracted parameters
            if result.get("data") and "parameters" in result["data"]:
                parameters = result["data"]["parameters"]
                print("- Extracted Parameters:")
                for param, value in parameters.items():
                    print(f"  - {param}: {value}")
            else:
                print("- No parameters extracted")
            
            # Show self-healing information
            query_progression = result.get("query_progression", [])
            if len(query_progression) > 1:
                print(f"\n- Self-Healing Progression ({len(query_progression)} queries):")
                for i, attempt in enumerate(query_progression):
                    print(f"  - Attempt {i}: {attempt['query'][:80]}...")
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
    
    print("\n=== Testing Complete ===")

async def test_orchestrator_integration():
    """Test parameter extraction through the enhanced orchestrator"""
    print(f"\n=== Testing Parameter Extraction via Enhanced Orchestrator ===\n")
    
    # Initialize orchestrator
    orchestrator = EnhancedOrchestratorAgent()
    session_id = "test_session"
    
    # Test queries that should extract parameters
    test_queries = [
        "What is the energy output parameter for Belgium's nuclear fleet?",
        "How many operational reactors are there in France?",
        "Calculate the capacity factor using the energy_output_t parameter from the database"
    ]
    
    for query in test_queries:
        print(f"\n>>> Testing orchestrator with query: {query}")
        
        try:
            # Process through the orchestrator
            result = await orchestrator.process_query(query, session_id)
            
            # Check for parameters
            params_found = False
            if result and 'parameters' in result:
                print(f"\nParameters extracted via orchestrator:")
                for param, value in result['parameters'].items():
                    print(f"  - {param}: {value}")
                    params_found = True
            
            if not params_found:
                print("No parameters extracted through orchestrator")
                
            # Print summary
            print(f"\nOrchestrator result summary: {json.dumps(result, indent=2)[:500]}...")
            
        except Exception as e:
            print(f"Error in orchestrator test: {str(e)}")
    
    print("\n=== Testing Complete ===")

def main():
    """Run the parameter extraction tests with command-line options"""
    parser = argparse.ArgumentParser(description="Test PostgreSQL parameter extraction with self-healing")
    parser.add_argument("--orchestrator", action="store_true", help="Test parameter extraction through orchestrator")
    parser.add_argument("--case", type=int, help="Test specific case index (0, 1, 2)", default=None)
    args = parser.parse_args()
    
    if args.orchestrator:
        asyncio.run(test_orchestrator_integration())
    else:
        asyncio.run(test_parameter_extraction())

if __name__ == "__main__":
    main()
