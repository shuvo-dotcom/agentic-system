"""
Postgres Data Provider Agent - Connects to external PostgreSQL-based data API endpoints
and provides refined data requests based on user queries.
"""
import aiohttp
import json
import logging
import re
from typing import Any, Dict, List, Optional
import os
import asyncio

from core.simple_base_agent import SimpleBaseAgent
from utils.llm_provider import get_llm_response

async def async_get_llm_response(prompt=None, messages=None, **kwargs):
    """
    Async wrapper for get_llm_response to handle async calls properly.
    This function ensures messages are properly formatted.
    """
    # Make sure we handle prompt vs messages correctly
    if prompt is not None and messages is None:
        # If prompt is provided but not messages, create a user message
        messages = [{"role": "user", "content": prompt}]
        prompt = None
    
    # If system_prompt is provided, handle it properly
    system_prompt = kwargs.pop("system_prompt", None)
    if system_prompt and isinstance(messages, list):
        # Insert system prompt at the beginning if it's not already there
        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
    
    # Use synchronous version but await to be compatible with async code
    return get_llm_response(messages=messages, prompt=prompt, **kwargs)

def load_postgres_config():
    """Load PostgreSQL API configuration from file or environment variables"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'postgres_settings.json')
    config = {}
    
    # Try to load from file
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load PostgreSQL config from file: {e}")
    
    # Override with environment variables if set
    if os.environ.get("POSTGRES_API_ENDPOINT"):
        config['postgres_api_endpoint'] = os.environ.get("POSTGRES_API_ENDPOINT")
    if os.environ.get("POSTGRES_API_KEY"):
        config['api_key'] = os.environ.get("POSTGRES_API_KEY")
    
    return config


class PostgresDataProvider(SimpleBaseAgent):
    """
    Agent responsible for making requests to external PostgreSQL-based data API endpoints.
    Refines user queries to be more specific and structured for data retrieval.
    """
    
    def __init__(self, endpoint_url=None):
        super().__init__(
            name="PostgresDataProvider",
            description="Connects to external PostgreSQL-based data API endpoints to retrieve precise data based on refined user queries"
        )
        
        # Load configuration
        config = load_postgres_config()
        
        # Set configuration properties
        self.endpoint_url = endpoint_url or config.get('postgres_api_endpoint')
        self.api_key = config.get('api_key', '')
        self.connection_timeout = config.get('connection_timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2)
        
        # Setup dedicated logger
        self.logger = logging.getLogger(__name__)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data retrieval request with self-healing capability.
        If a query fails, the system will progressively relax constraints and try again.
        
        Args:
            input_data: Dictionary containing:
                - user_query: The original user query
                - context: Additional context for request refinement (optional)
                - direct_query: A pre-refined query (optional, bypasses refinement step)
                - api_endpoint: Optional override for the endpoint URL
                - max_fallback_attempts: Maximum number of fallback attempts (optional)
                
        Returns:
            Dictionary with retrieved data and metadata
        """
        try:
            user_query = input_data.get("user_query", "")
            context = input_data.get("context", {})
            direct_query = input_data.get("direct_query")
            endpoint_url = input_data.get("api_endpoint") or self.endpoint_url
            max_fallback_attempts = input_data.get("max_fallback_attempts", 3)
            
            if not endpoint_url:
                return self.create_error_response("No API endpoint URL configured")
            
            # Initialize tracking for self-healing attempts
            fallback_attempts = 0
            query_progression = []
            current_query = None
            final_api_response = None
            success = False
            
            # Either use the direct query or refine the user query
            if direct_query:
                refined_query = direct_query
                self.logger.info(f"Using direct pre-refined query: {refined_query[:100]}...")
            else:
                # Initial refinement - most specific and constrained
                refined_query = await self._refine_user_query(user_query, context)
                self.logger.info(f"Initial refined query: {refined_query[:100]}...")
            
            # Add the initial query to our progression tracking
            current_query = refined_query
            query_progression.append({
                "attempt": fallback_attempts,
                "query": current_query,
                "type": "initial" 
            })
            
            # Try the query with progressively relaxed constraints
            while fallback_attempts <= max_fallback_attempts and not success:
                try:
                    # Make the API request with current query
                    self.logger.info(f"Attempt {fallback_attempts}: Making API request with query: {current_query[:100]}...")
                    api_response = await self._make_api_request(endpoint_url, current_query)
                    
                    # Process the API response
                    # Check if this is a parameter extraction query
                    is_parameter_query = "parameter" in user_query.lower() or "value" in user_query.lower() or any(param in user_query.lower() for param in ["energy_output", "capacity", "generation", "reactor"])
                    
                    # If this is a parameter query, handle it differently
                    if is_parameter_query:
                        return await self._process_parameter_api_response(api_response, user_query)
                    
                    # Check if the response indicates success
                    if (isinstance(api_response, dict) and "error" not in api_response) or \
                       (isinstance(api_response, list) and len(api_response) > 0):
                        
                        # Success! We have a valid response
                        self.logger.info("API request successful")
                        final_api_response = api_response
                        success = True
                        
                    else:
                        # Query failed, try with less constraints if we have attempts left
                        error_msg = api_response.get("error", "Unknown error") if isinstance(api_response, dict) else "Empty result set"
                        self.logger.warning(f"Attempt {fallback_attempts} failed: {error_msg}")
                        
                        if fallback_attempts < max_fallback_attempts:
                            # Generate a fallback query with relaxed constraints
                            fallback_attempts += 1
                            current_query = await self._generate_fallback_query(current_query, fallback_attempts, is_parameter_query)
                            
                            # Add to query progression
                            query_progression.append({
                                "attempt": fallback_attempts,
                                "query": current_query,
                                "type": f"fallback_{fallback_attempts}" 
                            })
                            
                            self.logger.info(f"Trying fallback query #{fallback_attempts}: {current_query[:100]}...")
                        else:
                            # We've exhausted our fallback attempts
                            self.logger.warning("Max fallback attempts reached with no successful result")
                            break
                            
                except Exception as e:
                    self.logger.error(f"Error during API request cycle: {str(e)}")
                    if fallback_attempts < max_fallback_attempts:
                        # Try with a fallback query
                        fallback_attempts += 1
                        current_query = await self._generate_fallback_query(current_query, fallback_attempts, is_parameter_query)
                        
                        # Add to query progression
                        query_progression.append({
                            "attempt": fallback_attempts,
                            "query": current_query,
                            "type": f"fallback_after_error_{fallback_attempts}" 
                        })
                    else:
                        # We've exhausted our fallback attempts
                        self.logger.warning("Max fallback attempts reached after errors")
                        break
            
            # Prepare the final result
            if success:
                return {
                    "status": "success" if fallback_attempts == 0 else "partial_success",
                    "data": final_api_response,
                    "refined_query": refined_query,
                    "final_query": current_query,
                    "attempts": fallback_attempts + 1,
                    "query_progression": query_progression,
                    "original_query": user_query,
                    "timestamp": self.get_timestamp()
                }
            else:
                return {
                    "status": "failed",
                    "error": "Failed to retrieve data after multiple attempts",
                    "attempts": fallback_attempts + 1,
                    "query_progression": query_progression,
                    "original_query": user_query,
                    "refined_query": refined_query,
                    "timestamp": self.get_timestamp()
                }
                
        except Exception as e:
            self.logger.error(f"Error in process method: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "original_query": input_data.get("user_query", ""),
                "timestamp": self.get_timestamp()
            }
    
    async def execute_query(self, query: str, allow_self_healing=True, max_fallback_attempts=3):
        """
        Execute a PostgreSQL query with self-healing capability.
        
        Args:
            query: The query to execute
            allow_self_healing: Whether to allow self-healing (fallback) attempts
            max_fallback_attempts: Maximum number of fallback attempts
            
        Returns:
            Query results with metadata
        """
        max_attempts = max_fallback_attempts if allow_self_healing else 0
        
        return await self.process({
            "user_query": query,
            "direct_query": query,  # Skip the refinement step
            "max_fallback_attempts": max_attempts
        })
    
    async def execute_parameter_query(self, query: str, parameter_name: str, allow_self_healing=True, max_fallback_attempts=3):
        """
        Execute a PostgreSQL query specifically for parameter extraction.
        This method is specialized for queries that aim to extract specific parameter values.
        
        Args:
            query: The query to execute
            parameter_name: The name of the parameter to extract
            allow_self_healing: Whether to allow self-healing (fallback) attempts
            max_fallback_attempts: Maximum number of fallback attempts
            
        Returns:
            Query results with extracted parameters
        """
        self.logger.info(f"Executing parameter query for '{parameter_name}': {query[:100]}...")
        
        max_attempts = max_fallback_attempts if allow_self_healing else 0
        
        # Create a parameter-focused query if not already parameter-focused
        param_focused_query = query
        if parameter_name not in query.lower() and "value" not in query.lower():
            param_focused_query = f"SELECT * FROM ({query}) AS param_query WHERE 1=1 -- Looking for parameter {parameter_name}"
        
        # Process the parameter query
        result = await self.process({
            "user_query": f"Get the value for parameter {parameter_name}",  # Make user_query parameter-focused
            "direct_query": param_focused_query,  # Skip the refinement step
            "max_fallback_attempts": max_attempts,
            "context": {
                "parameter_name": parameter_name,
                "query_type": "parameter_extraction"
            }
        })
        
        # Ensure the parameter is included in the response
        if result.get("status") in ["success", "partial_success"]:
            # First check if we already have parameters in the expected location
            has_parameter = False
            if result.get("data") and isinstance(result["data"], dict):
                if "parameters" in result["data"] and parameter_name in result["data"]["parameters"]:
                    has_parameter = True
                    self.logger.info(f"Parameter '{parameter_name}' already found in result")
            
            # If not found, try to extract from the data
            if not has_parameter and "data" in result:
                # Try to process the data to extract the parameter
                api_data = result["data"]
                self.logger.info(f"Extracting parameter '{parameter_name}' from data: {type(api_data)}")
                
                # If api_data is a list and we're expecting to find a parameter, look at first item
                if isinstance(api_data, list) and len(api_data) > 0:
                    first_item = api_data[0]
                    self.logger.info(f"Looking at first item in list: {list(first_item.keys())}")
                
                # Extract the parameter value
                parameter_value = self._extract_parameter_value(api_data, parameter_name)
                
                if parameter_value is not None:
                    # Add the parameter to the result
                    if "data" not in result:
                        result["data"] = {}
                    if "parameters" not in result["data"]:
                        result["data"]["parameters"] = {}
                    
                    result["data"]["parameters"][parameter_name] = parameter_value
                    self.logger.info(f"Successfully extracted parameter '{parameter_name}' = '{parameter_value}'")
                else:
                    self.logger.warning(f"Failed to extract parameter '{parameter_name}' from result")
                    
                    # Last resort: try running _process_parameter_api_response directly
                    if isinstance(api_data, list) and len(api_data) > 0:
                        processed_params = await self._process_parameter_api_response(api_data, f"Extract parameter {parameter_name}")
                        if processed_params and "parameters" in processed_params:
                            if parameter_name in processed_params["parameters"]:
                                if "parameters" not in result["data"]:
                                    result["data"]["parameters"] = {}
                                result["data"]["parameters"][parameter_name] = processed_params["parameters"][parameter_name]
                                self.logger.info(f"Extracted parameter via _process_parameter_api_response: {parameter_name} = {processed_params['parameters'][parameter_name]}")
        
        return result
    
    def _extract_parameter_value(self, api_data, parameter_name: str):
        """
        Extract parameter value from API data with enhanced field detection.
        
        Args:
            api_data: The API response data
            parameter_name: The name of the parameter to extract
            
        Returns:
            Extracted parameter value or None if not found
        """
        # Log what we're trying to extract and the data type
        self.logger.info(f"Extracting parameter '{parameter_name}' from data of type: {type(api_data)}")
        
        # Ensure api_data is in a consistent format (list of records)
        if api_data is None:
            self.logger.warning("API data is None, cannot extract parameter")
            return None
            
        if isinstance(api_data, dict):
            # Direct parameter check at top level
            if parameter_name in api_data:
                self.logger.info(f"Found direct parameter '{parameter_name}' at top level")
                return api_data[parameter_name]
                
            # Handle parameter in nested 'parameters'
            if "parameters" in api_data and parameter_name in api_data["parameters"]:
                self.logger.info(f"Found parameter '{parameter_name}' in nested parameters")
                return api_data["parameters"][parameter_name]
                
            # Handle single record case
            records = [api_data]
        elif isinstance(api_data, list):
            records = api_data
        else:
            self.logger.warning(f"Unsupported API data format: {type(api_data)}")
            return None
        
        if not records:
            self.logger.warning("Empty records list, cannot extract parameter")
            return None
            
        # Debug what fields are available
        if records and len(records) > 0:
            self.logger.info(f"Available fields in first record: {list(records[0].keys())}")
            
        # Try direct field match first with variations
        field_alternatives = [
            parameter_name,  # Direct match
            parameter_name.lower(),  # Lowercase
            parameter_name.upper(),  # Uppercase
            f"{parameter_name}_value",  # Common suffix
            f"value_{parameter_name}",  # Common prefix
            f"{parameter_name}_amt",  # Abbreviation
            "value",  # Generic value field
            "amount",  # Common value alternatives
            "quantity",
            "result"
        ]
        
        # Special handling for specific parameters
        if parameter_name in ["energy_output_t"]:
            field_alternatives.extend([
                "total_generation", "annual_energy_output", "generation", 
                "output", "production", "energy", "total_energy_output",
                "energy_output"  # Added direct energy_output field
            ])
        elif parameter_name in ["n"]:
            field_alternatives.extend([
                "operational_reactors", "number_of_units", "units", 
                "reactors", "plants", "count", "solar_facilities"
            ])
        
        # Try each field alternative
        for field in field_alternatives:
            for record in records:
                if field in record and record[field] is not None:
                    return record[field]
        
        # If not found, try numeric detection for certain parameters
        if parameter_name in ["n"]:
            # Look for any numeric field that's not in certain exclusions
            for record in records:
                for key, value in record.items():
                    if key not in ["energy_output_t", "date", "timestamp", "id"] and value is not None:
                        try:
                            numeric_value = float(value)
                            # If it's a small integer, likely a count
                            if numeric_value.is_integer() and 0 < numeric_value < 100:
                                return int(numeric_value)
                        except (ValueError, TypeError):
                            pass
        
        # Fall back to any numeric value for energy-related parameters
        if parameter_name in ["energy_output_t"]:
            for record in records:
                for key, value in record.items():
                    if key not in ["n", "date", "timestamp", "id"] and value is not None:
                        try:
                            numeric_value = float(value)
                            # If it's a large number, likely an energy value
                            if numeric_value > 1000:
                                return numeric_value
                        except (ValueError, TypeError):
                            pass
        
        return None
        
    async def _make_api_request(self, endpoint_url: str, query: str) -> Dict[str, Any]:
        """
        Make the API request to the PostgreSQL endpoint with retry logic.
        
        Args:
            endpoint_url: The URL of the API endpoint
            query: The refined query to send
            
        Returns:
            API response data
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Add API key if available
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Format the payload based on the endpoint's expected structure
        # For host.docker.internal:5678/webhook/sql-query endpoint
        payload = {
            "query": query,
            "source": "agentic_system",
            "format": "json",
            "timeout": self.connection_timeout
        }
        
        self.logger.info(f"Making API request to {endpoint_url} with query: {query}")
        
        # Implement retry logic
        for attempt in range(self.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(endpoint_url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limiting
                            retry_after = int(response.headers.get('Retry-After', self.retry_delay * 2))
                            self.logger.warning(f"Rate limited. Retrying after {retry_after} seconds.")
                            await asyncio.sleep(retry_after)
                        else:
                            error_text = await response.text()
                            self.logger.error(f"API request failed with status {response.status}: {error_text}")
                            
                            # For certain status codes, we don't want to retry
                            if response.status in [400, 401, 403, 404]:
                                return {
                                    "error": f"API request failed with status {response.status}",
                                    "details": error_text
                                }
                            
                            # For other errors, we'll retry after a delay
                            retry_delay = self.retry_delay * (attempt + 1)  # Exponential backoff
                            self.logger.warning(f"Retrying in {retry_delay} seconds (attempt {attempt + 1}/{self.max_retries})")
                            await asyncio.sleep(retry_delay)
            except asyncio.TimeoutError:
                self.logger.warning(f"Request timed out. Retrying in {self.retry_delay} seconds (attempt {attempt + 1}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"API request error: {str(e)}")
                # For connection errors, we'll retry after a delay
                retry_delay = self.retry_delay * (attempt + 1)  # Exponential backoff
                self.logger.warning(f"Retrying in {retry_delay} seconds (attempt {attempt + 1}/{self.max_retries})")
                await asyncio.sleep(retry_delay)
        
        # If we've exhausted all retries
        return {
            "error": "Failed to connect to the PostgreSQL API after multiple attempts",
            "details": "Maximum retry attempts reached"
        }
    
    async def _process_api_response(self, api_response: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Process the API response based on its content and format.
        This includes interpreting errors, converting formats, and extracting key data.
        
        Args:
            api_response: The raw API response
            original_query: The original query for context
            
        Returns:
            Processed data
        """
        # Check if this is a parameter extraction query
        is_parameter_query = "parameter" in original_query.lower() or "value" in original_query.lower() or any(param in original_query.lower() for param in ["energy_output", "capacity", "generation", "reactor"])
        
        # If this is a parameter query, handle it differently
        if is_parameter_query:
            return await self._process_parameter_api_response(api_response, original_query)
        
        # Process a general data query
        if "error" in api_response:
            return {
                "error": api_response["error"],
                "details": api_response.get("details", "No additional details available")
            }
        
        # Standard response processing
        if isinstance(api_response, list):
            return {
                "data": api_response,
                "record_count": len(api_response),
                "metadata": {
                    "source": "postgres_api",
                    "timestamp": self.get_timestamp(),
                    "fields": list(api_response[0].keys()) if api_response else []
                }
            }
        elif isinstance(api_response, dict):
            # Single record response
            return {
                "data": [api_response],
                "record_count": 1,
                "metadata": {
                    "source": "postgres_api",
                    "timestamp": self.get_timestamp(),
                    "fields": list(api_response.keys())
                }
            }
        else:
            return {
                "error": "Unexpected API response format",
                "details": f"Response was of type {type(api_response)}"
            }
    
    async def _generate_fallback_query(self, current_query: str, attempt: int, is_parameter_query: bool = False) -> str:
        """
        Generate a fallback query with progressively relaxed constraints.
        
        Args:
            current_query: The current query that failed
            attempt: The attempt number (1-based) for progressive relaxation
            is_parameter_query: Whether this is a parameter-specific query
            
        Returns:
            A new query with relaxed constraints
        """
        self.logger.info(f"Generating fallback query for attempt {attempt}")
        
        # Create different prompts based on the query type
        # Check if this is a parameter-specific query
        is_parameter_query = "parameter" in user_query.lower() or "value" in user_query.lower() or any(param in user_query.lower() for param in ["energy_output", "capacity", "generation", "reactor"])
        
        # Create a different prompt based on whether this is a parameter query or general query
        if is_parameter_query:
            prompt = f"""
            The following query to extract parameter values from a PostgreSQL database failed and we need to make it less constrained:
            
            ```sql
            {current_query}
            ```
            
            For attempt #{attempt}, please create a less constrained version of this query by:
            
            {self._get_param_relaxation_strategy(attempt)}
            
            Make sure the new query is still focused on extracting parameter values but with fewer restrictions.
            Return only the SQL query, without any comments or explanation.
            """
        else:
            prompt = f"""
            The following PostgreSQL query failed and we need to make it less constrained:
            
            ```sql
            {current_query}
            ```
            
            For attempt #{attempt}, please create a less constrained version of this query by:
            
            {self._get_relaxation_strategy(attempt)}
            
            Make sure the new query still serves the same purpose but with fewer restrictions.
            Return only the SQL query, without any comments or explanation.
            """
        
        # Get the fallback query from the LLM
        try:
            fallback_query = await async_get_llm_response(
                prompt=prompt, 
                max_tokens=300, 
                temperature=0.3, 
                system_prompt="You are an expert PostgreSQL data engineer. Your task is to modify SQL queries to make them less restricted when they fail."
            )
            
            # Clean up the response
            fallback_query = fallback_query.strip()
            if fallback_query.startswith("```sql"):
                fallback_query = fallback_query.split("```sql")[1]
            if fallback_query.endswith("```"):
                fallback_query = fallback_query.split("```")[0]
            
            fallback_query = fallback_query.strip()
            self.logger.info(f"Generated fallback query: {fallback_query}")
            
            return fallback_query
        except Exception as e:
            self.logger.error(f"Error generating fallback query: {str(e)}")
            
            # In case of error, provide a simplified version by removing some common constraints
            simplified = self._simplify_query(current_query, attempt)
            self.logger.info(f"Using simplified query instead: {simplified}")
            return simplified
    
    def _simplify_query(self, query: str, attempt: int) -> str:
        """
        Simple query simplification as fallback.
        
        Args:
            query: The query to simplify
            attempt: The attempt number
            
        Returns:
            Simplified query
        """
        # Level 1: Remove WHERE clauses
        if attempt == 1:
            # Remove specific conditions but keep the basic structure
            query = re.sub(r'WHERE\s+.*?(?=GROUP BY|ORDER BY|LIMIT|$)', ' WHERE 1=1 ', query, flags=re.IGNORECASE)
            
        # Level 2: Remove GROUP BY, HAVING clauses
        elif attempt == 2:
            query = re.sub(r'WHERE\s+.*?(?=GROUP BY|ORDER BY|LIMIT|$)', ' WHERE 1=1 ', query, flags=re.IGNORECASE)
            query = re.sub(r'GROUP BY\s+.*?(?=HAVING|ORDER BY|LIMIT|$)', ' ', query, flags=re.IGNORECASE)
            query = re.sub(r'HAVING\s+.*?(?=ORDER BY|LIMIT|$)', ' ', query, flags=re.IGNORECASE)
            
        # Level 3: Remove ORDER BY, LIMIT clauses and simplify to basic query
        else:
            query = re.sub(r'WHERE\s+.*?(?=GROUP BY|ORDER BY|LIMIT|$)', ' WHERE 1=1 ', query, flags=re.IGNORECASE)
            query = re.sub(r'GROUP BY\s+.*?(?=HAVING|ORDER BY|LIMIT|$)', ' ', query, flags=re.IGNORECASE)
            query = re.sub(r'HAVING\s+.*?(?=ORDER BY|LIMIT|$)', ' ', query, flags=re.IGNORECASE)
            query = re.sub(r'ORDER BY\s+.*?(?=LIMIT|$)', ' ', query, flags=re.IGNORECASE)
            query = re.sub(r'LIMIT\s+\d+', ' LIMIT 100', query, flags=re.IGNORECASE)
        
        return query
    
    def _get_relaxation_strategy(self, attempt: int) -> str:
        """
        Get constraint relaxation strategy based on attempt number.
        
        Args:
            attempt: The attempt number (1-based)
            
        Returns:
            Strategy description
        """
        strategies = [
            "1. Remove specific date ranges or time constraints, replacing them with wider ranges.\n2. Remove any numeric range constraints (e.g., change 'value > 500 AND value < 1000' to just 'value IS NOT NULL').",
            "1. Remove specific filtering conditions in WHERE clauses, keeping only essential table joins.\n2. Remove any string matching patterns or make them less specific (e.g., change LIKE 'nuclear%' to LIKE '%').",
            "1. Remove GROUP BY clauses and associated aggregation functions.\n2. Remove ORDER BY clauses.\n3. Replace complex subqueries with simpler direct queries.",
            "1. Simplify to a basic SELECT query with minimal constraints.\n2. Focus only on retrieving the core data without filtering, grouping or sorting."
        ]
        
        # Return the appropriate strategy based on attempt number (cap at the last strategy)
        strategy_index = min(attempt - 1, len(strategies) - 1)
        return strategies[strategy_index]
    
    def _get_param_relaxation_strategy(self, attempt: int) -> str:
        """
        Get parameter-specific constraint relaxation strategy.
        These strategies are optimized for parameter extraction queries.
        
        Args:
            attempt: The attempt number (1-based)
            
        Returns:
            Strategy description
        """
        strategies = [
            "1. Remove specific date ranges or time constraints.\n2. Keep the focus on retrieving the parameter value but with fewer conditions.",
            "1. Simplify table joins if present, keeping only the essential ones.\n2. Remove any string matching patterns or make them less specific.",
            "1. Remove aggregation functions but keep the focus on the parameter.\n2. Try different tables or columns that might contain the same information.",
            "1. Create a very basic query that focuses solely on retrieving any available value for the parameter from any relevant table."
        ]
        
        # Return the appropriate strategy based on attempt number (cap at the last strategy)
        strategy_index = min(attempt - 1, len(strategies) - 1)
        return strategies[strategy_index]
    
    async def _refine_user_query(self, user_query: str, context: Dict[str, Any] = None) -> str:
        """
        Refine a user's natural language query into a structured PostgreSQL query.
        
        Args:
            user_query: The original user query
            context: Additional context for refinement
            
        Returns:
            Refined SQL query
        """
        context = context or {}
        
        # Check if a specific parameter name is provided in the context
        parameter_name = context.get("parameter_name")
        
        prompt = f"""
        Translate this natural language query to a PostgreSQL query:
        
        ```
        {user_query}
        ```
        
        Your task is to create a precise, efficient SQL query that extracts the requested information.
        
        {f"Important: This query should focus on extracting the value for the parameter: {parameter_name}" if parameter_name else ""}
        
        Return only the SQL query, without any comments or explanation.
        """
        
        try:
            refined_query = await async_get_llm_response(
                prompt=prompt, 
                max_tokens=300, 
                temperature=0.1, 
                system_prompt="You are an expert PostgreSQL engineer who specializes in creating efficient, precise SQL queries from natural language requests."
            )
            
            # Clean up the response
            refined_query = refined_query.strip()
            if refined_query.startswith("```sql"):
                refined_query = refined_query.split("```sql")[1]
            elif refined_query.startswith("```"):
                refined_query = refined_query.split("```")[1]
            if refined_query.endswith("```"):
                refined_query = refined_query.split("```")[0]
                
            refined_query = refined_query.strip()
            return refined_query
            
        except Exception as e:
            self.logger.error(f"Error refining query: {str(e)}")
            # If refinement fails, return a simple version of the user query
            return f"SELECT * FROM data WHERE description LIKE '%{user_query}%' LIMIT 100"
            
    def get_timestamp(self):
        """Get the current timestamp string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            error_message: The error message
            
        Returns:
            Error response dictionary
        """
        return {
            "status": "error",
            "error": error_message,
            "timestamp": self.get_timestamp()
        }
    
    async def _process_parameter_api_response(self, api_response: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Process API responses for parameter extraction queries.
        This method is specialized for extracting parameter values from the data.
        
        Args:
            api_response: The raw API response
            original_query: The original user query for context
            
        Returns:
            Processed parameters ready for use
        """
        self.logger.info(f"Processing parameter API response: {json.dumps(api_response)[:200]}...")
        
        # Extract parameter names from the query
        param_names = self._extract_parameter_names(original_query)
        self.logger.info(f"Looking for parameters: {param_names}")
        
        # Check for specific parameter in the context
        context_param = None
        if "context" in original_query or "parameter_name" in original_query:
            for param in param_names:
                if param in original_query:
                    context_param = param
                    self.logger.info(f"Found context parameter: {context_param}")
                    break
        
        # Check for errors in the API response
        if isinstance(api_response, dict) and "error" in api_response:
            return {
                "error": api_response.get("error"),
                "parameters": {}
            }
            
        # Check if the response is empty or None
        if not api_response or (isinstance(api_response, list) and len(api_response) == 0):
            return {
                "error": "No parameter data found in API response",
                "parameters": {}
            }
            
        # Special case: if we have a dict with 'data' key containing our actual records
        if isinstance(api_response, dict) and "data" in api_response and (
            isinstance(api_response["data"], list) or 
            (isinstance(api_response["data"], dict) and not "parameters" in api_response["data"])
        ):
            self.logger.info("Found nested data structure, using data field")
            api_response = api_response["data"]
        
        # Ensure we have a list of records
        api_data = api_response if isinstance(api_response, list) else [api_response]
        
        # If there's a specific parameter requested in the context, ensure it's in param_names
        if context_param and context_param not in param_names:
            param_names.append(context_param)
        
        # Start building the parameter mappings
        parameters = {}
        
        # Common field mappings for parameters - define first for referencing
        field_mappings = {
            "energy_output_t": ["energy_output", "total_generation", "annual_energy_output", "generation", "output", "production", "energy", "total_energy_output", "value"],
            "n": ["operational_reactors", "number_of_units", "units", "reactors", "plants", "count", "solar_facilities", "reactor_count"]
        }
        
        # First, try the most specific mapped fields for each parameter
        for param_name in param_names:
            # Skip if already found
            if param_name in parameters:
                continue
                
            # Check if we have mappings for this parameter
            if param_name in field_mappings:
                # Try each field candidate in order
                for field in field_mappings[param_name]:
                    value = self._extract_first_non_null_value(api_data, field)
                    if value is not None:
                        parameters[param_name] = value
                        self.logger.info(f"Extracted parameter '{param_name}' = '{value}' from field '{field}'")
                        break
        
        # Next, try direct parameter name matching for any we didn't find
        for param_name in param_names:
            if param_name not in parameters:
                value = self._extract_first_non_null_value(api_data, param_name)
                if value is not None:
                    parameters[param_name] = value
                    self.logger.info(f"Extracted parameter '{param_name}' directly with value '{value}'")
                    
        
        # If we still don't have all parameters but have at least one field with values,
        # make some educated guesses based on the data
        if len(parameters) < len(param_names) and api_data and len(api_data[0]) > 0:
            self.logger.info(f"Making educated guesses from available data: {list(api_data[0].keys())}")
            
            # Special case: if we have a 'value' field and energy_output_t is missing, use it
            if "energy_output_t" in param_names and "energy_output_t" not in parameters:
                for record in api_data:
                    if "value" in record and record["value"] is not None:
                        parameters["energy_output_t"] = record["value"]
                        self.logger.info(f"Using 'value' field for energy_output_t: {record['value']}")
                        break
            
            # Look for any numeric fields if n is missing
            if "n" in param_names and "n" not in parameters:
                numeric_fields = []
                for key in api_data[0].keys():
                    if key not in ["energy_output_t", "value", "date_date", "timestamp"] and api_data[0][key] is not None:
                        try:
                            float(api_data[0][key])
                            numeric_fields.append(key)
                        except (ValueError, TypeError):
                            pass
                
                if numeric_fields:
                    # Use the first numeric field that's not already mapped
                    field = numeric_fields[0]
                    parameters["n"] = api_data[0][field]
                    self.logger.info(f"Using numeric field '{field}' for n: {parameters['n']}")
                        
        # Return the processed data with extracted parameters
        return {
            "parameters": parameters,
            "raw_data": api_data,
            "timestamp": self.get_timestamp(),
            "parameter_extraction": "success" if parameters else "partial_success" if api_data else "failed"
        }
        
    def _extract_parameter_names(self, query: str) -> List[str]:
        """
        Extract parameter names from a query
        
        Args:
            query: The original query
            
        Returns:
            List of parameter names
        """
        self.logger.info(f"Extracting parameter names from query: {query}")
        
        # First check for direct parameter context
        direct_pattern = r"(?:parameter|param)\s+([a-zA-Z0-9_]+)"
        direct_match = re.search(direct_pattern, query, re.IGNORECASE)
        if direct_match:
            param = direct_match.group(1)
            self.logger.info(f"Found direct parameter mention: {param}")
            return [param]
        
        # Common parameter patterns
        param_patterns = [
            r"values?\s+for\s+([a-zA-Z0-9_,\s]+)(?:\s+from|\s+in|\s+for|\s+of)?",
            r"parameters?\s+([a-zA-Z0-9_,\s]+)(?:\s+from|\s+in|\s+for|\s+of)?",
            r"extract\s+([a-zA-Z0-9_,\s]+)(?:\s+from|\s+in|\s+for|\s+of)?"
        ]
        
        # Try to extract parameter names using patterns
        for pattern in param_patterns:
            matches = re.search(pattern, query, re.IGNORECASE)
            if matches:
                # Split by commas and clean up
                param_text = matches.group(1)
                
                # Handle "and" between parameters
                param_text = param_text.replace(" and ", ", ")
                
                raw_params = [p.strip() for p in param_text.split(",")]
                
                # Further clean each parameter name
                params = []
                for p in raw_params:
                    # Remove any trailing "from", "in", "for", etc.
                    cleaned = re.sub(r'\s+(from|in|for|of)\s+.*$', '', p, flags=re.IGNORECASE)
                    params.append(cleaned.strip())
                
                self.logger.info(f"Found parameters using pattern: {params}")
                return params
        
        # If no parameters found, check for common parameters
        common_params = ["energy_output_t", "n", "capacity_factor", "capacity"]
        for param in common_params:
            if param in query.lower():
                self.logger.info(f"Found common parameter in query: {param}")
                return [param]
        
        # Default to assume it's about energy output if we can't detect anything
        self.logger.info("No specific parameter found, defaulting to energy_output_t")
        return ["energy_output_t"]
    
    def _extract_first_non_null_value(self, data_list: List[Dict[str, Any]], field_name: str) -> Any:
        """
        Extract the first non-null value for a field from a list of dictionaries.
        
        Args:
            data_list: List of data dictionaries
            field_name: The field name to extract
            
        Returns:
            The first non-null value, or None if not found
        """
        # Try the exact field name first
        for item in data_list:
            if field_name in item and item[field_name] is not None:
                return item[field_name]
                
        # Try case-insensitive match
        lowercase_field = field_name.lower()
        for item in data_list:
            for key in item:
                if key.lower() == lowercase_field and item[key] is not None:
                    return item[key]
        
        # Try with common field name variations
        variations = [
            f"{field_name}_value",
            f"value_{field_name}",
            f"{field_name}_amt",
            f"{field_name}_amount"
        ]
        
        for variation in variations:
            for item in data_list:
                if variation in item and item[variation] is not None:
                    return item[variation]
        
        return None
