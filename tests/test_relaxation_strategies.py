"""
Test the relaxation strategies of the PostgresDataProvider agent without calling external APIs.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.postgres_data_provider import PostgresDataProvider

def test_relaxation_strategies():
    """Test the different relaxation strategies"""
    provider = PostgresDataProvider()
    
    print("\n--- Testing Relaxation Strategies ---")
    for attempt in range(1, 5):
        strategy = provider._get_relaxation_strategy(attempt)
        print(f"\nAttempt #{attempt} Strategy:")
        print(strategy)
    
    # Test extracting key entities
    print("\n--- Testing Entity Extraction ---")
    test_queries = [
        "What was the nuclear power production in Belgium during 2022?",
        "How much solar energy was generated in France last year?",
        "What are the wind power statistics for Germany from 2018-2022?",
        "What is the trend in hydro electricity production in Europe?",
        "How does coal usage compare to renewable energy in Spain?"
    ]
    
    for query in test_queries:
        entity = provider._extract_key_entity(query)
        print(f"Query: '{query}'")
        print(f"Extracted Entity: '{entity}'")
        print()

if __name__ == "__main__":
    test_relaxation_strategies()
