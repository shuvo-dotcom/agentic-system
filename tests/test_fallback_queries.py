"""
Test the self-healing fallback query generation functionality of the PostgresDataProvider agent.
"""
import asyncio
import os
import sys
import json
import logging
from pprint import pprint

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.postgres_data_provider import PostgresDataProvider

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_fallback_query_generation():
    """Test the fallback query generation functionality"""
    provider = PostgresDataProvider()
    
    # Set a test endpoint
    provider.endpoint_url = "http://localhost:5678/webhook/sql-query"
    
    # Test query that's likely to need fallbacks
    original_query = "What was the nuclear power production in Belgium during the summer of 2022, broken down by week?"
    
    # Test fallback query generation with different attempt numbers
    for attempt in range(1, 5):
        fallback_query = await provider._generate_fallback_query(
            original_query=original_query,
            previous_query="SELECT date, sum(production_mwh) FROM nuclear_production WHERE country='Belgium' AND date BETWEEN '2022-06-21' AND '2022-09-22' GROUP BY date ORDER BY date",
            attempt=attempt
        )
        
        print(f"\n--- Fallback Attempt #{attempt} ---")
        print(f"Strategy: {provider._get_relaxation_strategy(attempt)}")
        print(f"Generated query: {fallback_query}")

async def test_process_with_fallbacks():
    """Test the end-to-end process method with fallbacks"""
    provider = PostgresDataProvider()
    
    # Set a test endpoint - use a non-existent endpoint to force fallbacks
    provider.endpoint_url = "http://localhost:5678/webhook/sql-query"
    
    # Process data with a test query
    result = await provider.process({
        "user_query": "What was the nuclear power production in Belgium during the summer of 2022, broken down by week?",
        "max_fallback_attempts": 3
    })
    
    print("\n--- Process Result with Fallbacks ---")
    pprint(result)

async def main():
    """Run the test functions"""
    print("Testing fallback query generation...")
    await test_fallback_query_generation()
    
    print("\nTesting process with fallbacks...")
    await test_process_with_fallbacks()

if __name__ == "__main__":
    # Run the async tests
    asyncio.run(main())
