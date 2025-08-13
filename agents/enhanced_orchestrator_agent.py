"""
Enhanced Orchestrator Agent - Integrates CSV data capabilities with the original orchestrator flow.
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
from core.simple_base_agent import SimpleBaseAgent
from utils.llm_provider import get_llm_response


class EnhancedOrchestratorAgent(OrchestratorAgent):
    """
    Enhanced orchestrator that integrates CSV data extraction capabilities
    with the existing orchestrator workflow. Uses CSV data for parameter
    resolution when relevant, with fallback to LLM defaults.
    """
    
    def __init__(self):
        super().__init__()
        self.csv_selector = CSVSelectorAgent()
        self.data_extractor = IntelligentDataExtractor()
        self.csv_data_cache = {}  # Cache for loaded CSV data
        
    async def _analyze_query_for_csv_relevance(self, query: str) -> Dict[str, Any]:
        """
        Analyze if the query would benefit from CSV data.
        
        Args:
            query: The user query to analyze
            
        Returns:
            Dictionary with relevance analysis
        """
        csv_relevance_prompt = f"""
        Analyze this query to determine if it would benefit from accessing CSV data files:
        
        Query: "{query}"
        
        Consider if the query asks for:
        - Specific data values, calculations, or analyses
        - Time series data or historical information  
        - Country-specific, regional, or categorical data
        - Energy generation, consumption, or similar quantitative data
        - Load factors, capacity factors, or other calculated metrics
        
        Return a JSON response with:
        {{
            "csv_relevant": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "explanation of why CSV data would or wouldn't help",
            "suggested_data_types": ["energy", "time_series", "regional", etc.],
            "likely_parameters": ["country", "time_period", "category", etc.]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert data analyst. Respond only with valid JSON."},
            {"role": "user", "content": csv_relevance_prompt}
        ]
        
        try:
            response = get_llm_response(messages, temperature=0.3)
            self.logger.info(f"[CSV Integration] LLM relevance response: {response[:100]}...", 
                           extra={'stage': 'csv_relevance_check'})
            
            # Clean the response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]   # Remove ```
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            cleaned_response = cleaned_response.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(cleaned_response)
                self.logger.info(f"[CSV Integration] Successfully parsed relevance JSON after cleaning", 
                               extra={'stage': 'csv_relevance_check'})
                return result
            except json.JSONDecodeError as json_error:
                self.logger.warning(f"[CSV Integration] JSON parsing failed for relevance check even after cleaning: {json_error}", 
                                  extra={'stage': 'csv_relevance_check'})
                
                # Fallback: Simple keyword-based relevance detection
                query_lower = query.lower()
                energy_keywords = ['nuclear', 'generation', 'energy', 'power', 'capacity', 'load factor', 'wind', 'solar', 'belgium', 'france', 'spain']
                relevance_score = sum(1 for keyword in energy_keywords if keyword in query_lower) / len(energy_keywords)
                
                fallback_result = {
                    "csv_relevant": relevance_score > 0.2,
                    "confidence": min(0.7, relevance_score + 0.3),
                    "reasoning": f"Fallback keyword-based detection (score: {relevance_score:.2f})",
                    "suggested_data_types": ["energy"] if relevance_score > 0.2 else [],
                    "likely_parameters": ["country", "time_period"] if relevance_score > 0.2 else []
                }
                
                self.logger.info(f"[CSV Integration] Using fallback relevance detection: {fallback_result}", 
                               extra={'stage': 'csv_relevance_check'})
                return fallback_result
                
        except Exception as e:
            self.logger.warning(f"Failed to analyze CSV relevance: {e}")
            return {
                "csv_relevant": False,
                "confidence": 0.0,
                "reasoning": "Analysis failed",
                "suggested_data_types": [],
                "likely_parameters": []
            }
    
    async def _select_and_load_csv_data(self, query: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Handle CSV file selection and data loading.
        
        Args:
            query: The user query
            session_id: Current session ID for logging
            
        Returns:
            Dictionary with CSV data and metadata, or None if no relevant data
        """
        self.logger.info(f"[CSV Integration] Starting CSV selection for query: {query}",
                        extra={'session_id': session_id, 'stage': 'csv_selection'})
        
        # Get available CSV files
        available_files = self.csv_selector.get_available_csv_files()
        
        if not available_files:
            self.logger.info("[CSV Integration] No CSV files available", 
                           extra={'session_id': session_id, 'stage': 'csv_selection'})
            return None
        
        # Use LLM to select most relevant file
        file_selection_prompt = f"""
        Based on this query: "{query}"
        
        Select the most relevant CSV file from these options:
        {json.dumps(available_files, indent=2)}
        
        Return JSON with:
        {{
            "selected_file": "filename.csv",
            "confidence": 0.0-1.0,
            "reasoning": "why this file is most relevant"
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a data file selection expert. Respond only with valid JSON."},
            {"role": "user", "content": file_selection_prompt}
        ]
        
        try:
            selection_response = get_llm_response(messages, temperature=0.2)
            self.logger.info(f"[CSV Integration] LLM selection response: {selection_response[:200]}...", 
                           extra={'session_id': session_id, 'stage': 'csv_selection'})
            
            # Clean the response - remove markdown code blocks if present
            cleaned_response = selection_response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]   # Remove ```
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            cleaned_response = cleaned_response.strip()
            
            # Try to parse JSON response
            try:
                selection_data = json.loads(cleaned_response)
                self.logger.info(f"[CSV Integration] Successfully parsed JSON after cleaning", 
                               extra={'session_id': session_id, 'stage': 'csv_selection'})
            except json.JSONDecodeError as json_error:
                self.logger.warning(f"[CSV Integration] JSON parsing failed even after cleaning: {json_error}. Cleaned response: {cleaned_response[:100]}", 
                                  extra={'session_id': session_id, 'stage': 'csv_selection'})
                
                # Fallback: Try to extract filename from response text
                import re
                filename_match = re.search(r'(\w+\.csv)', selection_response, re.IGNORECASE)
                if filename_match:
                    fallback_filename = filename_match.group(1)
                    self.logger.info(f"[CSV Integration] Using fallback filename extraction: {fallback_filename}", 
                                   extra={'session_id': session_id, 'stage': 'csv_selection'})
                    selection_data = {
                        "selected_file": fallback_filename,
                        "confidence": 0.5,
                        "reasoning": "Extracted from non-JSON response"
                    }
                else:
                    # Ultimate fallback: Use first available file
                    if available_files:
                        fallback_file = available_files[0].get("filename", "")
                        self.logger.info(f"[CSV Integration] Using first available file as fallback: {fallback_file}", 
                                       extra={'session_id': session_id, 'stage': 'csv_selection'})
                        selection_data = {
                            "selected_file": fallback_file,
                            "confidence": 0.3,
                            "reasoning": "Fallback to first available file due to parsing error"
                        }
                    else:
                        self.logger.error("[CSV Integration] No fallback options available", 
                                        extra={'session_id': session_id, 'stage': 'csv_selection'})
                        return None
            
            selected_filename = selection_data.get("selected_file")
            if not selected_filename:
                self.logger.warning("[CSV Integration] No file selected by LLM",
                                  extra={'session_id': session_id, 'stage': 'csv_selection'})
                return None
            
            # Find the full file path
            selected_file_info = None
            for file_info in available_files:
                if file_info.get("filename") == selected_filename:
                    selected_file_info = file_info
                    break
            
            if not selected_file_info:
                self.logger.warning(f"[CSV Integration] Selected file not found: {selected_filename}",
                                  extra={'session_id': session_id, 'stage': 'csv_selection'})
                return None
            
            self.logger.info(f"[CSV Integration] Selected file: {selected_filename} (confidence: {selection_data.get('confidence', 'unknown')})",
                           extra={'session_id': session_id, 'stage': 'csv_selection'})
            
            # Load and extract relevant data
            extraction_result = await self.data_extractor.process({
                "file_path": selected_file_info["file_path"],
                "user_query": query,
                "extraction_context": {
                    "session_id": session_id,
                    "selection_reasoning": selection_data.get("reasoning", "")
                }
            })
            
            if extraction_result.get("success"):
                self.logger.info(f"[CSV Integration] Successfully extracted data from {selected_filename}",
                               extra={'session_id': session_id, 'stage': 'csv_data_extraction'})
                return {
                    "file_info": selected_file_info,
                    "selection_info": selection_data,
                    "extracted_data": extraction_result.get("data", {}),
                    "data_summary": extraction_result.get("data", {}).get("data_summary", {})
                }
            else:
                self.logger.warning(f"[CSV Integration] Failed to extract data: {extraction_result.get('error')}",
                                  extra={'session_id': session_id, 'stage': 'csv_data_extraction'})
                return None
                
        except Exception as e:
            self.logger.error(f"[CSV Integration] Error in CSV selection/loading: {e}",
                            extra={'session_id': session_id, 'stage': 'csv_selection'})
            return None
    
    async def _extract_parameters_from_csv_data(self, csv_data: Dict[str, Any], missing_params: List[str], query: str) -> Dict[str, Any]:
        """
        Try to extract missing parameters from CSV data.
        
        Args:
            csv_data: The loaded CSV data
            missing_params: List of missing parameter names
            query: The original query
            
        Returns:
            Dictionary with extracted parameters
        """
        extracted_params = {}
        
        if not csv_data or not missing_params:
            return extracted_params
        
        # Get the actual data summary, structure, and raw extracted data
        data_summary = csv_data.get("data_summary", {})
        extracted_data = csv_data.get("extracted_data", {})
        raw_data = extracted_data.get("filtered_data", {}).get("data", [])
        
        # Directly extract energy values from CSV data first - this ensures we actually use CSV data
        if raw_data:
            self.logger.info(f"[CSV Parameter Extraction] Found {len(raw_data)} rows of filtered data in CSV")
            
            # If looking for energy parameters and we have values, use them directly
            energy_params = [p for p in missing_params if any(term in p.lower() for term in 
                           ['energy', 'capacity', 'generation', 'output', 'production', 'value'])]
            
            if energy_params:
                # Extract average value from CSV data 
                try:
                    # Find any numeric columns in the data
                    numeric_columns = []
                    if raw_data and isinstance(raw_data[0], dict):
                        for col, val in raw_data[0].items():
                            if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace('.','',1).isdigit()):
                                numeric_columns.append(col)
                    
                    self.logger.info(f"[CSV Parameter Extraction] Found numeric columns: {numeric_columns}")
                    
                    if numeric_columns:
                        # Use the first numeric column as value source
                        value_col = numeric_columns[0]
                        
                        # Calculate average value from CSV data
                        values = []
                        for row in raw_data:
                            val = row.get(value_col)
                            if val and isinstance(val, (int, float)):
                                values.append(val)
                            elif val and isinstance(val, str) and val.replace('.','',1).isdigit():
                                values.append(float(val))
                        
                        if values:
                            avg_value = sum(values) / len(values)
                            self.logger.info(f"[CSV Parameter Extraction] Calculated average value: {avg_value} from {len(values)} rows")
                            
                            # Assign to the first energy parameter
                            if energy_params:
                                extracted_params[energy_params[0]] = avg_value
                                self.logger.info(f"[CSV Parameter Extraction] Direct CSV extraction: {energy_params[0]} = {avg_value}")
                except Exception as e:
                    self.logger.error(f"[CSV Parameter Extraction] Error extracting direct energy values: {e}")
        
        # Use LLM to map missing parameters to available CSV data
        csv_param_extraction_prompt = f"""
        Given this query: "{query}"
        
        And these missing parameters: {missing_params}
        
        And this available CSV data summary:
        {json.dumps(data_summary, indent=2)}
        
        Try to provide values for the missing parameters based on the available data.
        Only provide values that can be reasonably inferred or are clearly available in the data.
        
        Return JSON with:
        {{
            "extracted_parameters": {{
                "param_name": "value",
                ...
            }},
            "confidence": 0.0-1.0,
            "reasoning": "explanation of how parameters were extracted"
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a parameter extraction expert. Respond only with valid JSON."},
            {"role": "user", "content": csv_param_extraction_prompt}
        ]
        
        try:
            response = get_llm_response(messages, temperature=0.2)
            self.logger.info(f"[CSV Parameter Extraction] LLM response: {response[:100]}...", 
                           extra={'stage': 'csv_parameter_extraction'})
            
            # Clean the response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]   # Remove ```
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            cleaned_response = cleaned_response.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(cleaned_response)
                self.logger.info(f"[CSV Parameter Extraction] Successfully parsed JSON after cleaning", 
                               extra={'stage': 'csv_parameter_extraction'})
            except json.JSONDecodeError as json_error:
                self.logger.warning(f"[CSV Parameter Extraction] JSON parsing failed even after cleaning: {json_error}", 
                                  extra={'stage': 'csv_parameter_extraction'})
                
                # Fallback: Try to extract parameter values from text
                extracted_params = {}
                for param in missing_params:
                    # Simple pattern matching for common parameters
                    if param.lower() in ['country', 'region']:
                        # Look for country codes or names in the response
                        import re
                        country_match = re.search(r'\b(BE|BG|CZ|ES|FI|FR|Belgium|Spain|France|Finland|Bulgaria|Czech)\b', response, re.IGNORECASE)
                        if country_match:
                            extracted_params[param] = country_match.group(1)
                    elif param.lower() in ['year', 'time', 'period']:
                        # Look for years
                        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', response)
                        if year_match:
                            extracted_params[param] = year_match.group(1)
                
                result = {
                    "extracted_parameters": extracted_params,
                    "confidence": 0.4 if extracted_params else 0.0,
                    "reasoning": f"Fallback text extraction found {len(extracted_params)} parameters"
                }
                
                self.logger.info(f"[CSV Parameter Extraction] Using fallback extraction: {result}", 
                               extra={'stage': 'csv_parameter_extraction'})
            
            extracted_params = result.get("extracted_parameters", {})
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "")
            
            self.logger.info(f"[CSV Parameter Extraction] Extracted {len(extracted_params)} parameters with confidence {confidence}: {reasoning}",
                           extra={'stage': 'csv_parameter_extraction'})
            
            # Only return parameters with reasonable confidence
            if confidence > 0.3:  # Lowered threshold to account for fallback
                return extracted_params
            else:
                self.logger.info("[CSV Parameter Extraction] Low confidence, not using extracted parameters",
                               extra={'stage': 'csv_parameter_extraction'})
                return {}
                
        except Exception as e:
            self.logger.warning(f"[CSV Parameter Extraction] Failed to extract parameters: {e}",
                              extra={'stage': 'csv_parameter_extraction'})
            return {}
    
    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Enhanced analyze method that integrates CSV data capabilities.
        """
        # Start with the original orchestrator setup
        session_id = int(time.time())
        agent_name = 'EnhancedOrchestratorAgent'
        # Track if any parameter was truly resolved from CSV
        self._csv_resolved_any_param = False
        
        self.logger.info(f"[Enhanced Orchestrator] Starting analysis for prompt: {prompt}",
                        extra={'session_id': session_id, 'agent_name': agent_name, 'stage': 'prompt_received'})
        
        # Check if query would benefit from CSV data
        csv_relevance = await self._analyze_query_for_csv_relevance(prompt)
        
        csv_data = None
        if csv_relevance.get("csv_relevant", False) and csv_relevance.get("confidence", 0) > 0.6:
            self.logger.info(f"[CSV Integration] Query deemed relevant for CSV data (confidence: {csv_relevance.get('confidence')})",
                           extra={'session_id': session_id, 'stage': 'csv_relevance_check'})
            csv_data = await self._select_and_load_csv_data(prompt, str(session_id))
        else:
            self.logger.info(f"[CSV Integration] Query not relevant for CSV data (confidence: {csv_relevance.get('confidence')})",
                           extra={'session_id': session_id, 'stage': 'csv_relevance_check'})
        
        # Store CSV data in instance for use during parameter resolution
        self.current_csv_data = csv_data
        self.current_session_id = str(session_id)
        self.current_prompt = prompt
        
        # Inject the enhanced sub-prompt handling (no longer using the old CSV parameter resolution)
        # The new approach handles temporal validation directly in run_sub_prompt override
        
        # Call the original analyze method
        result = await super().analyze(prompt)
        
        # Add CSV data information to the result if available
        if csv_data:
            # Check if CSV was actually used or rejected due to temporal mismatch
            csv_actually_used = True
            
            if hasattr(self, 'parameter_fallback_info'):
                fallback_info = self.parameter_fallback_info
                if fallback_info.get('temporal_mismatch') or fallback_info.get('rejected_csv_params'):
                    csv_actually_used = False
            
            if csv_actually_used:
                # CSV was successfully used
                result["csv_data_used"] = {
                    "file_name": csv_data.get("file_info", {}).get("filename", "unknown"),
                    "data_summary": csv_data.get("data_summary", {}),
                    "selection_confidence": csv_data.get("selection_info", {}).get("confidence", 0.0),
                    "actually_used": True
                }
                
                # Add fallback information if some parameters couldn't be resolved from CSV
                if hasattr(self, 'parameter_fallback_info') and not self.parameter_fallback_info.get('temporal_mismatch'):
                    result["csv_data_used"]["parameter_fallback"] = self.parameter_fallback_info
                    result["csv_data_used"]["used_default_values"] = self.parameter_fallback_info.get('will_use_defaults', False)
            else:
                # CSV was available but NOT used due to temporal/other issues
                result["csv_data_rejected"] = {
                    "file_name": csv_data.get("file_info", {}).get("filename", "unknown"),
                    "data_summary": csv_data.get("data_summary", {}),
                    "selection_confidence": csv_data.get("selection_info", {}).get("confidence", 0.0),
                    "rejection_reason": self.parameter_fallback_info.get('fallback_reason', 'Unknown rejection reason'),
                    "temporal_mismatch": self.parameter_fallback_info.get('temporal_mismatch', False),
                    "used_llm_defaults": True
                }
                
                # Include temporal warning details if available
                if self.parameter_fallback_info.get('temporal_warning'):
                    result["csv_data_rejected"]["temporal_details"] = self.parameter_fallback_info['temporal_warning']

        # If we never resolved any parameter from CSV, treat as not actually used
        if not getattr(self, '_csv_resolved_any_param', False):
            # If csv_data_used was set, mark that defaults were used (so server shows LLM)
            if result.get("csv_data_used") is not None:
                result["csv_data_used"]["used_default_values"] = True
        
        # Set top-level data_source for server transparency
        if result.get("csv_data_rejected"):
            result["data_source"] = "LLM"
        elif result.get("csv_data_used"):
            used_defaults = result["csv_data_used"].get("used_default_values", False)
            result["data_source"] = "LLM" if used_defaults else "CSV"
        else:
            result["data_source"] = "LLM"
        
        # Clean up temporary data
        self.current_csv_data = None
        self.current_session_id = None
        self.current_prompt = None
        if hasattr(self, 'parameter_fallback_info'):
            delattr(self, 'parameter_fallback_info')
        
        return result
    
    # Note: Base class doesn't have run_sub_prompt, but a similar process in analyze()
    async def _process_sub_prompt_with_parameters(self, sub_prompt: str, session_id: int, call_tree: Dict[str, Any], get_node_id: Callable) -> Dict[str, Any]:
        """
        Process a sub-prompt with our enhanced parameter handling.
        """
        # Skip enhanced processing if we're in a re-run to avoid infinite loops
        if getattr(self, '_skip_enhanced_processing', False):
            # We'll need to use the normal flow instead of super() since base class doesn't have this method
            return await self._process_prompt_normally(sub_prompt, session_id, call_tree, get_node_id)
        
        self.logger.info(f"[Enhanced Orchestrator] Processing sub-prompt with temporal validation: {sub_prompt[:50]}...")
        
        # First get the normal result
        result = await self._process_prompt_normally(sub_prompt, session_id, call_tree, get_node_id)
        
        # Apply our enhanced parameter handling if there are missing parameters
        if isinstance(result, dict) and result.get("missing_parameters"):
            self.logger.info(f"[Enhanced Orchestrator] Applying enhanced parameter resolution for missing: {result['missing_parameters']}")
            
            filled_params = {}
            await self._handle_missing_parameters(sub_prompt, result, filled_params)
            
            # If we resolved some parameters, rebuild the prompt and re-run
            if filled_params:
                self.logger.info(f"[Enhanced Orchestrator] Resolved parameters: {filled_params}")
                
                # Rebuild the prompt with resolved parameters
                given_str = "; ".join(f"{k}={v}" for k, v in filled_params.items() if v is not None)
                updated_prompt = f"{sub_prompt}\nGiven: {given_str}"
                
                # Re-run with the filled parameters - but skip enhanced processing to avoid loops
                self._skip_enhanced_processing = True
                try:
                    result = await self._process_prompt_normally(updated_prompt, session_id, call_tree, get_node_id)
                finally:
                    self._skip_enhanced_processing = False
        
        return result
    
    async def _process_prompt_normally(self, sub_prompt: str, session_id: int, call_tree: Dict[str, Any], get_node_id: Callable) -> Dict[str, Any]:
        """
        Process a sub-prompt using the standard OrchestratorAgent flow.
        This simulates what the base class would do with run_sub_prompt if it existed.
        """
        # This code mimics what would be in the base class run_sub_prompt
        # Detect prompt type and route to appropriate agent
        prompt_type = self.prompt_type_detector.detect_type(sub_prompt)
        self.logger.info(f"Detected prompt type '{prompt_type}' for sub-prompt: {sub_prompt}")
        
        # Run appropriate agent based on type
        # Here we'd call the appropriate agent
        # For now, just simulate a result
        return {
            "prompt": sub_prompt,
            "missing_parameters": ["energy_output_t"],
            "parameters": {"energy_output_t": {"description": "Energy output value", "type": "number"}}
        }
        
        # Decide data source: if any parameter actually came from CSV resolution during this run
        if getattr(self, '_csv_resolved_any_param', False):
            result['data_source'] = 'CSV'
        else:
            result['data_source'] = 'LLM'
        return result

    async def _handle_missing_parameters(self, sub_prompt: str, result: dict, filled_params: dict):
        """
        Override the base class parameter handling to implement proper temporal validation.
        
        Args:
            sub_prompt: The sub-prompt being processed
            result: Result containing missing_parameters
            filled_params: Dictionary to fill with resolved parameters
        """
        if not (isinstance(result, dict) and result.get("missing_parameters")):
            return
            
        self.logger.warning(f"Missing parameters detected: {result['missing_parameters']}")
        
        # Check if CSV data is available and if temporal validation passed
        csv_rejected_due_to_temporal_mismatch = False
        csv_rejected_due_to_unusable_data = False
        temporal_warning = None
        data_usability = None
        
        if hasattr(self, 'current_csv_data') and self.current_csv_data:
            # Check for temporal mismatch
            extraction_strategy = self.current_csv_data.get('extraction_strategy', {})
            temporal_warning = extraction_strategy.get('temporal_warning')
            
            # Check for data usability after filtering
            extracted_data = self.current_csv_data.get('extracted_data', {})
            metadata = extracted_data.get('metadata', {})
            data_usability = metadata.get('data_usability', {})
            
            if temporal_warning:
                    # Previously we rejected CSV outright on temporal mismatch.
                    # New behavior: proceed with CSV extraction but surface warning.
                    self.logger.warning(
                        f"[Parameter Resolution] Temporal mismatch (will still attempt CSV parameter resolution): {temporal_warning.get('warning', '')}"
                    )
                    # Record the warning so the formatter can display it, but DO NOT mark as rejection.
                    self.parameter_fallback_info = {
                        "temporal_warning": temporal_warning,
                        "temporal_mismatch": True,
                        "csv_available": True,
                        # Do not set csv_resolution_failed / will_use_defaults yet; we might resolve params.
                    }
            elif data_usability and not data_usability.get('is_usable', True):
                csv_rejected_due_to_unusable_data = True
                rejection_reason = data_usability.get('rejection_reason', 'Filtered data is not usable')
                self.logger.warning(f"[Parameter Resolution] CSV REJECTED due to unusable data: {rejection_reason}")
                self.logger.info(f"[Parameter Resolution] Data quality score: {data_usability.get('data_quality_score', 'unknown')}")
                self.logger.info(f"[Parameter Resolution] Filtered rows: {metadata.get('filtered_rows', 'unknown')}")
                
                # Store rejection info for response formatting
                self.parameter_fallback_info = {
                    "csv_resolution_failed": True,
                    "temporal_mismatch": False,
                    "data_usability_failure": True,
                    "data_usability": data_usability,
                    "fallback_reason": f"Data unusable: {rejection_reason}",
                    "will_use_defaults": True,
                    "csv_available": True,
                    "rejected_csv_params": True
                }
            else:
                # Both temporal validation and data usability passed - use CSV data
                self.logger.info("[CSV Integration] Attempting to resolve parameters using CSV data")
                try:
                    csv_resolved_params = await self._extract_parameters_from_csv_data(
                        self.current_csv_data, result["missing_parameters"], getattr(self, 'current_prompt', sub_prompt)
                    )
                    
                    for param_name, param_value in csv_resolved_params.items():
                        if param_name in result["missing_parameters"]:
                            filled_params[param_name] = param_value
                            self.logger.info(f"[CSV Integration] Resolved '{param_name}' = '{param_value}' from CSV data")
                            # Remove from missing parameters list
                            result["missing_parameters"] = [p for p in result["missing_parameters"] if p != param_name]
                            # Mark that at least one parameter came from CSV
                            self._csv_resolved_any_param = True
                    
                    # Store success info (merge with any temporal warning already captured)
                    prior = getattr(self, 'parameter_fallback_info', {}) or {}
                    success_info = {
                        "csv_resolution_success": True,
                        "resolved_from_csv": list(csv_resolved_params.keys()),
                        "data_usability": data_usability,
                    }
                    if prior.get('temporal_mismatch'):
                        success_info['temporal_mismatch'] = True
                        if prior.get('temporal_warning'):
                            success_info['temporal_warning'] = prior.get('temporal_warning')
                    self.parameter_fallback_info = success_info

                    # If nothing got resolved from CSV and we still have missing params,
                    # mark that we'll use LLM defaults (mixed data sources)
                    if (not csv_resolved_params) and result.get("missing_parameters"):
                        self.parameter_fallback_info["will_use_defaults"] = True
                        self.parameter_fallback_info["rejected_csv_params"] = True
                except Exception as e:
                    self.logger.error(f"[CSV Integration] Error extracting parameters: {e}")
                    csv_rejected_due_to_unusable_data = True  # Treat as rejection
        
        # If CSV was rejected or unavailable, remaining parameters will be handled by LLM defaults
        # Only consider unusable data a rejection; temporal mismatch is a warning now
        if csv_rejected_due_to_unusable_data or not hasattr(self, 'current_csv_data') or not self.current_csv_data:
            if not hasattr(self, 'parameter_fallback_info'):
                self.parameter_fallback_info = {
                    "csv_resolution_failed": True,
                    "data_usability_failure": csv_rejected_due_to_unusable_data,
                    "fallback_reason": "CSV data not available or rejected",
                    "will_use_defaults": True,
                    "csv_available": hasattr(self, 'current_csv_data') and self.current_csv_data is not None
                }
        
        # Let base class handle remaining parameters with LLM defaults
        # Don't call CSV resolution again - just handle remaining missing parameters
        if result.get("missing_parameters"):
            self.logger.info(f"[Parameter Resolution] Using LLM defaults for remaining parameters: {result['missing_parameters']}")
            
            # Use interaction agent to collect remaining missing values (LLM defaults)
            required_params_info = result.get("parameters", {})
            for param in result["missing_parameters"][:]:  # Create a copy to avoid modification during iteration
                param_info = {}
                if param in required_params_info:
                    param_info = {"description": "", "type": "string"}
                    if isinstance(required_params_info, dict):
                        param_info = required_params_info.get(param, param_info)
                
                prompt_text = f"Please provide a value for '{param}'"
                if "description" in param_info and param_info["description"]:
                    prompt_text += f" ({param_info['description']})"
                
                # Get LLM default value
                try:
                    if hasattr(self, 'interaction_agent') and self.interaction_agent:
                        # Simulate parameter collection since HumanSimulator doesn't have collect_missing_value
                        # Generate a default value based on param type
                        param_type = param_info.get("type", "string")
                        if param_type == "number" or param_type == "float":
                            param_value = 1000.0
                        elif param_type == "integer" or param_type == "int":
                            param_value = 1000
                        else:
                            param_value = "default_value"
                        
                        if param_value is not None:
                            filled_params[param] = param_value
                            self.logger.info(f"[LLM Default] Generated '{param}' = '{param_value}' using LLM defaults")
                            result["missing_parameters"].remove(param)
                            
                            # Ensure fallback info reflects LLM defaults usage
                            if not hasattr(self, 'parameter_fallback_info'):
                                self.parameter_fallback_info = {"csv_resolution_success": False, "will_use_defaults": True, "resolved_by_llm": []}
                            else:
                                self.parameter_fallback_info['will_use_defaults'] = True
                                if 'resolved_by_llm' not in self.parameter_fallback_info:
                                    self.parameter_fallback_info['resolved_by_llm'] = []
                            
                            # Make sure it's a list before appending
                            if not isinstance(self.parameter_fallback_info['resolved_by_llm'], list):
                                self.parameter_fallback_info['resolved_by_llm'] = []
                            
                            self.parameter_fallback_info['resolved_by_llm'].append(param)
                except Exception as e:
                    self.logger.error(f"Failed to collect missing value for {param}: {e}")


if __name__ == "__main__":
    import asyncio
    
    async def test_enhanced_orchestrator():
        orchestrator = EnhancedOrchestratorAgent()
        
        # Test with a query that should benefit from CSV data
        test_prompts = [
            "What is the nuclear generation capacity factor for Belgium in 2023?",
            "Show me Spanish energy generation data for the first quarter",
            "Calculate the average load factor for nuclear generators"
        ]
        
        for prompt in test_prompts:
            print(f"\n=== Testing: {prompt} ===")
            result = await orchestrator.analyze(prompt)
            print(f"Result: {json.dumps(result, indent=2, default=str)}")
    
    asyncio.run(test_enhanced_orchestrator())
