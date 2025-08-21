"""
Test the parameter extraction logic with explicit parameter naming
"""
import os
import sys
import asyncio
import logging
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.postgres_data_provider import PostgresDataProvider

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Mock data for different parameter types
MOCK_RESPONSES = {
    "energy_output_t": [
        {"energy_output": 48.3, "year": 2023, "country": "Belgium"},
        {"value": 48.3, "unit": "TWh"}
    ],
    "n": [
        {"reactor_count": 7, "country": "Belgium", "status": "operational"},
        {"operational_reactors": 7, "total_reactors": 8}
    ]
}

async def test_parameter_processing():
    """Test the parameter extraction with properly formatted queries"""
    provider = PostgresDataProvider()
    
    # Test case 1: Explicit parameter mention
    print("\n>>> Test Case 1: Explicit Parameter Mention")
    api_data = MOCK_RESPONSES["energy_output_t"]
    query = "Get the value for parameter energy_output_t from nuclear generation"
    
    # Call the parameter processing method
    processed = await provider._process_parameter_api_response(api_data, query)
    print(f"Input data: {api_data}")
    print(f"Query: {query}")
    print(f"Processed result: {processed}")
    
    # Test case 2: Explicit parameter for n
    print("\n>>> Test Case 2: Explicit Parameter for n")
    api_data = MOCK_RESPONSES["n"]
    query = "Get the value for parameter n from nuclear plants"
    
    # Call the parameter processing method
    processed = await provider._process_parameter_api_response(api_data, query)
    print(f"Input data: {api_data}")
    print(f"Query: {query}")
    print(f"Processed result: {processed}")
    
    # Test case 3: Multiple parameters
    print("\n>>> Test Case 3: Multiple Parameters")
    combined_data = [
        {"energy_output": 48.3, "operational_reactors": 7, "country": "Belgium"}
    ]
    query = "Get values for energy_output_t, n from nuclear generation"
    
    # Call the parameter processing method
    processed = await provider._process_parameter_api_response(combined_data, query)
    print(f"Input data: {combined_data}")
    print(f"Query: {query}")
    print(f"Processed result: {processed}")

if __name__ == "__main__":
    asyncio.run(test_parameter_processing())
