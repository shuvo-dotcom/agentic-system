"""
Enhanced Orchestrator Agent - Integrates CSV data capabilities and PostgreSQL data provider with the original orchestrator flow.
"""
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

from agents.orchestrator_agent import OrchestratorAgent
from agents.csv_selector_agent import CSVSelectorAgent  
from agents.intelligent_data_extractor import IntelligentDataExtractor
from agents.postgres_data_provider import PostgresDataProvider
from core.simple_base_agent import SimpleBaseAgent
from utils.llm_provider import get_llm_response


class EnhancedOrchestratorAgent(OrchestratorAgent):
    """
    Enhanced orchestrator that integrates multiple data sources:
    1. PostgreSQL data API endpoints for precise data retrieval
    2. CSV data extraction capabilities for local data
    
    Uses the most appropriate data source for parameter resolution,
    with fallback to LLM defaults when needed.
    """
    
    def __init__(self):
        super().__init__()
        self.csv_selector = CSVSelectorAgent()
        self.data_extractor = IntelligentDataExtractor()
        self.postgres_provider = PostgresDataProvider()
        self.csv_data_cache = {}  # Cache for loaded CSV data
        
    async def _analyze_data_source_relevance(self, query: str) -> Dict[str, Any]:
        """
        Analyze which data source would be most appropriate for the query.
        
        Args:
            query: The user query to analyze
            
        Returns:
            Dictionary with data source relevance analysis
        """
        data_relevance_prompt = f"""
        Analyze this query to determine the best data source to answer it:
        
        Query: "{query}"
        
        Based on the query, assign a relevance score from 0 to 10 for each data source:
        
        1. PostgreSQL Database: Contains real-time, structured data from production systems including:
           - Energy production metrics
           - Capacity factors and efficiency data
           - Historical generation statistics
           - Reactor and power plant information
           - Precise numerical values for various parameters
        
        2. CSV Files: Contains historical, tabulated data from reports including:
           - Summarized annual/monthly statistics
           - Consolidated performance reports
           - Trend analyses and forecasts
           - Data that may require calculations or transformations
        
        3. LLM Knowledge: The model's pre-existing knowledge about:
           - General energy concepts and principles
           - Typical values and ranges for parameters
           - Formulas and calculation methods
           - Contextual understanding of energy systems
        
        Return a JSON object with scores and reasoning.
        """
        
        try:
            response_text = await get_llm_response(data_relevance_prompt, temperature=0, max_tokens=500)
            
            # Extract the JSON part from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                relevance_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the structure from the text
                relevance_data = self._parse_relevance_from_text(response_text)
                
            self.logger.info(f"Data source relevance analysis: {relevance_data}")
            return relevance_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing data source relevance: {str(e)}")
            # Default to using all sources equally
            return {
                "PostgreSQL": {"score": 7, "reasoning": "Default fallback due to error"},
                "CSV": {"score": 7, "reasoning": "Default fallback due to error"},
                "LLM": {"score": 7, "reasoning": "Default fallback due to error"}
            }

    def _parse_relevance_from_text(self, text: str) -> Dict[str, Any]:
        """
        Parse relevance scores from text when JSON parsing fails.
        
        Args:
            text: The response text to parse
            
        Returns:
            Parsed relevance data
        """
        relevance_data = {
            "PostgreSQL": {"score": 5, "reasoning": "Could not parse from response"},
            "CSV": {"score": 5, "reasoning": "Could not parse from response"},
            "LLM": {"score": 5, "reasoning": "Could not parse from response"}
        }
        
        # Try to extract scores using regex
        postgres_match = re.search(r'PostgreSQL.*?(\d+)[\/\s]*10', text, re.DOTALL | re.IGNORECASE)
        csv_match = re.search(r'CSV.*?(\d+)[\/\s]*10', text, re.DOTALL | re.IGNORECASE)
        llm_match = re.search(r'LLM.*?(\d+)[\/\s]*10', text, re.DOTALL | re.IGNORECASE)
        
        # Extract reasoning for each source
        postgres_reason = re.search(r'PostgreSQL[^\n]*\n\s*(.+?)(?=\n\s*\d+\.|\Z)', text, re.DOTALL | re.IGNORECASE)
        csv_reason = re.search(r'CSV[^\n]*\n\s*(.+?)(?=\n\s*\d+\.|\Z)', text, re.DOTALL | re.IGNORECASE)
        llm_reason = re.search(r'LLM[^\n]*\n\s*(.+?)(?=\n\s*\d+\.|\Z)', text, re.DOTALL | re.IGNORECASE)
        
        # Update the relevance data with extracted information
        if postgres_match:
            relevance_data["PostgreSQL"]["score"] = int(postgres_match.group(1))
        if postgres_reason:
            relevance_data["PostgreSQL"]["reasoning"] = postgres_reason.group(1).strip()
            
        if csv_match:
            relevance_data["CSV"]["score"] = int(csv_match.group(1))
        if csv_reason:
            relevance_data["CSV"]["reasoning"] = csv_reason.group(1).strip()
            
        if llm_match:
            relevance_data["LLM"]["score"] = int(llm_match.group(1))
        if llm_reason:
            relevance_data["LLM"]["reasoning"] = llm_reason.group(1).strip()
            
        return relevance_data
    
    async def _fetch_postgres_data(self, query: str, session_id: str, param_name=None) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from PostgreSQL API endpoint with self-healing capability.
        If the initial query fails, the system will progressively relax constraints
        to attempt to retrieve relevant data.
        
        Args:
            query: The user query
            session_id: Current session ID for logging
            param_name: Optional parameter name to extract (for parameter-specific queries)
            
        Returns:
            Dictionary with PostgreSQL data and metadata, or None if retrieval failed
        """
        self.logger.info(f"[Postgres Integration] Starting data retrieval for query: {query}",
                       extra={'session_id': session_id, 'stage': 'postgres_data_retrieval'})
        
        # Use the PostgreSQL data provider to get refined data with self-healing
        try:
            if param_name:
                # Use parameter-specific query handling
                self.logger.info(f"[Postgres Integration] Executing parameter-specific query for '{param_name}'",
                               extra={'session_id': session_id, 'stage': 'postgres_parameter_retrieval'})
                               
                postgres_result = await self.postgres_provider.execute_parameter_query(
                    query, 
                    param_name, 
                    allow_self_healing=True,
                    max_fallback_attempts=3
                )
            else:
                # Use standard query handling
                postgres_result = await self.postgres_provider.process({
                    "user_query": query,
                    "max_fallback_attempts": 3,  # Allow up to 3 fallback attempts with relaxed constraints
                    "context": {
                        "session_id": session_id,
                        "importance": "high"  # Signal this is an important query worth retry attempts
                    }
                })
            
            # Check for success or partial success (where fallbacks were used)
            if postgres_result.get("status") in ["success", "partial_success"]:
                self.logger.info(f"[Postgres Integration] Successfully retrieved data from PostgreSQL API",
                               extra={'session_id': session_id, 'stage': 'postgres_data_retrieval'})
                
                # Log fallback attempts if any were made
                query_progression = postgres_result.get("query_progression", [])
                if len(query_progression) > 1:  # If more than just the initial query
                    self.logger.info(f"[Postgres Integration] Self-healing query feature used {len(query_progression)-1} fallback attempts",
                                   extra={'session_id': session_id, 'stage': 'postgres_data_retrieval'})
                    
                    # Log the progression of queries
                    for attempt in query_progression:
                        self.logger.info(f"[Postgres Integration] Attempt #{attempt['attempt']} ({attempt['type']}): {attempt['query'][:100]}...",
                                       extra={'session_id': session_id, 'stage': 'postgres_data_retrieval'})
                
                # Check if this is a parameter query result
                if param_name:
                    self.logger.info(f"[Postgres Integration] Parameter extraction result for '{param_name}'")
                    # Extract parameters for the orchestrator to use directly
                    if "data" in postgres_result and "parameters" in postgres_result["data"]:
                        extracted_params = postgres_result["data"]["parameters"]
                        
                        if extracted_params and param_name in extracted_params:
                            extracted_value = extracted_params[param_name]
                            self.logger.info(f"[Postgres Integration] Successfully extracted parameter '{param_name}' = '{extracted_value}'")
                
                # Extract and return the data with metadata and self-healing information
                return {
                    "data": postgres_result.get("data", {}),
                    "refined_query": postgres_result.get("refined_query", ""),
                    "final_query": postgres_result.get("final_query", ""),
                    "metadata": postgres_result.get("metadata", {}),
                    "parameters": postgres_result.get("data", {}).get("parameters", {}),
                    "self_healing": {
                        "attempts": postgres_result.get("attempts", 1),
                        "success_level": postgres_result.get("status", "unknown"),
                        "query_progression": postgres_result.get("query_progression", [])
                    },
                    "status": "success"
                }
            else:
                self.logger.warning(f"[Postgres Integration] Failed to retrieve PostgreSQL data after {postgres_result.get('attempts', 1)} attempts: {postgres_result.get('error', 'Unknown error')}",
                                  extra={'session_id': session_id, 'stage': 'postgres_data_retrieval'})
                return None
        except Exception as e:
            self.logger.error(f"[Postgres Integration] Error retrieving PostgreSQL data: {str(e)}",
                            extra={'session_id': session_id, 'stage': 'postgres_data_retrieval'})
            return None
