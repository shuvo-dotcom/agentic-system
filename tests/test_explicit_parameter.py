"""
Test script for explicit parameter extraction using the execute_parameter_query method
in PostgresDataProvider.
"""
import os
import sys
import asyncio
import logging
from pprint import pprint

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.postgres_data_provider import PostgresDataProvider

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_explicit_parameter_extraction():
    """Test the explicit parameter extraction method in PostgresDataProvider"""
    print("\n=== Testing Explicit Parameter Extraction ===\n")
    
    # Initialize the provider
    provider = PostgresDataProvider()
    
    # Test parameter extraction with direct queries
    test_cases = [
        {
            "query": "SELECT energy_output FROM nuclear_generation WHERE country = 'Belgium' AND year = 2023",
            "parameter": "energy_output_t",
            "description": "Direct nuclear energy output for Belgium"
        },
        {
            "query": "SELECT value FROM solar_generation WHERE country = 'Spain' AND year = 2023",
            "parameter": "energy_output_t",
            "description": "Solar energy with value field mapping"
        },
        {
            "query": "SELECT reactor_count FROM nuclear_plants WHERE country = 'France' AND status = 'operational'",
            "parameter": "n",
            "description": "Operational reactor count for France"
        },
        {
            "query": "SELECT total_generation, operational_reactors FROM energy_summary WHERE country = 'Germany' AND year = 2023",
            "parameter": "energy_output_t",
            "description": "Total generation with field mapping"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        param = test_case["parameter"]
        description = test_case["description"]
        
        print(f"\n>>> Test Case {i}: {description}")
        print(f"Query: {query}")
        print(f"Parameter to extract: {param}")
        
        # Execute the parameter query
        try:
            result = await provider.execute_parameter_query(
                query=query,
                parameter_name=param,
                allow_self_healing=True,
                max_fallback_attempts=2
            )
            
            # Print a summary of the response
            print("\nResponse Summary:")
            print(f"- Status: {result.get('status', 'unknown')}")
            
            # Check for extracted parameters
            if "data" in result and "parameters" in result["data"]:
                parameters = result["data"]["parameters"]
                print("- Extracted Parameters:")
                for param_name, value in parameters.items():
                    print(f"  - {param_name}: {value}")
            else:
                print("- No parameters extracted")
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_explicit_parameter_extraction())
