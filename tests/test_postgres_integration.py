"""
Test script for verifying the integration of the enhanced PostgresDataProvider with self-healing queries
in the EnhancedOrchestratorAgent workflow.
"""
import os
import sys
import asyncio
import logging
import json
from pprint import pprint

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.enhanced_orchestrator_agent import EnhancedOrchestratorAgent

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_postgres_integration():
    """Test the integration of PostgresDataProvider with self-healing in the enhanced orchestrator"""
    print("\n=== Testing PostgreSQL Integration with Self-Healing Queries ===\n")
    
    # Initialize the orchestrator
    orchestrator = EnhancedOrchestratorAgent()
    
    # Test queries that should use PostgreSQL data
    test_queries = [
        "What was the nuclear power production in Belgium during the summer of 2022?",
        "Show me energy generation data for France in 2023",
        "What is the capacity factor of solar plants in Spain during the last quarter?",
        "Compare renewable energy production between Germany and France"
    ]
    
    for query in test_queries:
        print(f"\n>>> Testing query: {query}")
        
        # Process the query through the orchestrator
        try:
            result = await orchestrator.analyze(query)
            
            # Print a summary of the response
            print("\nOrchestrator Response Summary:")
            print(f"- Status: {result.get('status', 'unknown')}")
            
            # Check for PostgreSQL data
            if 'postgres_data_used' in result:
                postgres_info = result['postgres_data_used']
                print("- PostgreSQL Data Source Used:")
                print(f"  - Source: {postgres_info.get('metadata', {}).get('data_source', 'PostgreSQL API')}")
                
                # Show self-healing information
                if 'self_healing' in postgres_info:
                    healing_info = postgres_info['self_healing']
                    attempts = healing_info.get('attempts', 1)
                    print(f"  - Self-Healing Active: {'Yes' if attempts > 1 else 'No'}")
                    print(f"  - Query Attempts: {attempts}")
                    print(f"  - Success Level: {healing_info.get('success_level', 'unknown')}")
            else:
                print("- PostgreSQL Data Not Used")
            
            # Check for CSV data
            if 'csv_data_used' in result:
                csv_info = result['csv_data_used']
                print("- CSV Data Source Used:")
                print(f"  - File: {csv_info.get('file_name', 'N/A')}")
            else:
                print("- CSV Data Not Used")
            
            # Print calculation results if present
            if 'results' in result and result['results']:
                print(f"- {len(result['results'])} calculation results found")
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_postgres_integration())
