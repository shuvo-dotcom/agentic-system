"""
Enhanced Query Orchestrator - Orchestrates the complete flow including CSV selection and intelligent data extraction.
"""
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from core.simple_base_agent import SimpleBaseAgent
from agents.csv_selector_agent import CSVSelectorAgent
from agents.intelligent_data_extractor import IntelligentDataExtractor
from agents.llm_formula_resolver import LLMFormulaResolver
from agents.calc_executor import CalcExecutor
from utils.llm_provider import get_llm_response


class EnhancedQueryOrchestrator(SimpleBaseAgent):
    """
    Enhanced orchestrator that handles the complete flow:
    1. CSV file selection
    2. Intelligent data extraction 
    3. Query processing with real data
    4. Calculation execution
    """
    
    def __init__(self, data_directory: str = "data/csv"):
        super().__init__(
            name="EnhancedQueryOrchestrator",
            description="Orchestrates the complete analysis flow with CSV selection and intelligent data extraction."
        )
        self.csv_selector = CSVSelectorAgent(data_directory)
        self.data_extractor = IntelligentDataExtractor()
        self.formula_resolver = LLMFormulaResolver()
        self.calc_executor = CalcExecutor()
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the complete query orchestration.
        
        Args:
            input_data: Dictionary containing:
                - user_query: The original user query
                - mode: "interactive" or "auto" (default: interactive)
                - selected_file: Optional pre-selected file path
                
        Returns:
            Dictionary with complete analysis results
        """
        try:
            user_query = input_data.get("user_query", "")
            mode = input_data.get("mode", "interactive")
            selected_file = input_data.get("selected_file")
            
            if not user_query:
                return self.create_error_response("No user query provided")
            
            # Initialize execution log
            execution_log = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "mode": mode,
                "steps": []
            }
            
            # Step 1: CSV File Selection
            if not selected_file:
                if mode == "interactive":
                    csv_selection_result = await self._handle_interactive_csv_selection()
                else:
                    csv_selection_result = await self._handle_auto_csv_selection(user_query)
                
                execution_log["steps"].append({
                    "step": "csv_selection",
                    "result": csv_selection_result
                })
                
                if not csv_selection_result.get("success"):
                    return self.create_error_response("CSV selection failed", {"execution_log": execution_log})
                
                selected_file = csv_selection_result["data"]["selected_file"]["file_path"]
            else:
                execution_log["steps"].append({
                    "step": "csv_selection",
                    "result": {"success": True, "message": f"Pre-selected file: {selected_file}"}
                })
            
            # Step 2: Intelligent Data Extraction
            extraction_result = await self._extract_relevant_data(selected_file, user_query)
            execution_log["steps"].append({
                "step": "data_extraction",
                "result": extraction_result
            })
            
            if not extraction_result.get("success"):
                return self.create_error_response("Data extraction failed", {"execution_log": execution_log})
            
            # Step 3: Enhanced Query Processing with Real Data
            processing_result = await self._process_query_with_data(
                user_query, 
                extraction_result["data"]
            )
            execution_log["steps"].append({
                "step": "query_processing",
                "result": processing_result
            })
            
            if not processing_result.get("success"):
                return self.create_error_response("Query processing failed", {"execution_log": execution_log})
            
            # Step 4: Code Execution
            execution_result = await self._execute_calculation(
                processing_result["data"]["executable_code"],
                processing_result["data"].get("parameters", {}),
                extraction_result["data"]["extracted_data"]
            )
            execution_log["steps"].append({
                "step": "calculation_execution",
                "result": execution_result
            })
            
            # Prepare final response
            final_result = {
                "user_query": user_query,
                "selected_file": selected_file,
                "data_extraction": extraction_result["data"],
                "query_processing": processing_result["data"],
                "execution_result": execution_result,
                "execution_log": execution_log,
                "accuracy_indicators": self._calculate_accuracy_indicators(execution_log)
            }
            
            return self.create_success_response(final_result)
            
        except Exception as e:
            self.logger.error(f"Error in enhanced orchestration: {str(e)}")
            return self.create_error_response(f"Enhanced orchestration failed: {str(e)}")
    
    async def _handle_interactive_csv_selection(self) -> Dict[str, Any]:
        """Handle interactive CSV file selection."""
        try:
            # Get available files
            csv_result = await self.csv_selector.process({"action": "list"})
            
            if not csv_result.get("success"):
                return csv_result
            
            available_files = csv_result["data"]["available_files"]
            
            if not available_files:
                return self.create_error_response("No CSV files available for selection")
            
            # Present options to user (in a real implementation, this would be interactive)
            # For now, we'll return the options and expect the selection in a follow-up call
            return {
                "success": True,
                "requires_user_input": True,
                "data": {
                    "available_files": available_files,
                    "presentation": csv_result["data"]["presentation"],
                    "message": "Please select a CSV file for analysis"
                }
            }
            
        except Exception as e:
            return self.create_error_response(f"Interactive CSV selection failed: {str(e)}")
    
    async def _handle_auto_csv_selection(self, user_query: str) -> Dict[str, Any]:
        """Automatically select the most relevant CSV file based on the query."""
        try:
            # Get available files
            csv_result = await self.csv_selector.process({"action": "list"})
            
            if not csv_result.get("success"):
                return csv_result
            
            available_files = csv_result["data"]["available_files"]
            
            if not available_files:
                return self.create_error_response("No CSV files available for auto-selection")
            
            # Use LLM to select the most relevant file
            selected_file = await self._llm_select_relevant_file(user_query, available_files)
            
            if not selected_file:
                # Fallback to first available file
                selected_file = available_files[0]
            
            return {
                "success": True,
                "data": {
                    "selected_file": selected_file,
                    "selection_method": "auto_llm" if selected_file else "fallback_first",
                    "message": f"Auto-selected: {selected_file['filename']}"
                }
            }
            
        except Exception as e:
            return self.create_error_response(f"Auto CSV selection failed: {str(e)}")
    
    async def _llm_select_relevant_file(self, user_query: str, available_files: List[Dict]) -> Optional[Dict]:
        """Use LLM to select the most relevant CSV file for the query."""
        
        system_prompt = """You are an expert data analyst. Given a user query and a list of available CSV files, select the most relevant file for answering the query.

Respond with ONLY the index number (0-based) of the most relevant file, or -1 if none are clearly relevant."""

        files_description = []
        for i, file_info in enumerate(available_files):
            desc = f"{i}: {file_info['filename']}"
            if 'columns' in file_info:
                desc += f" - Columns: {', '.join(file_info['columns'][:5])}"
            if 'preview_rows' in file_info and file_info['preview_rows']:
                desc += f" - Sample data available"
            files_description.append(desc)
        
        user_prompt = f"""
User Query: "{user_query}"

Available Files:
{chr(10).join(files_description)}

Which file is most relevant for answering the user's query?
"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = get_llm_response(
                messages,
                max_tokens=10,
                temperature=0.0
            )
            
            index = int(response.strip())
            if 0 <= index < len(available_files):
                return available_files[index]
                
        except Exception as e:
            self.logger.warning(f"LLM file selection failed: {str(e)}")
        
        return None
    
    async def _extract_relevant_data(self, file_path: str, user_query: str) -> Dict[str, Any]:
        """Extract relevant data from the selected CSV file."""
        try:
            extraction_input = {
                "file_path": file_path,
                "user_query": user_query,
                "extraction_context": {
                    "prioritize_accuracy": True,
                    "avoid_assumptions": True
                }
            }
            
            return await self.data_extractor.process(extraction_input)
            
        except Exception as e:
            return self.create_error_response(f"Data extraction failed: {str(e)}")
    
    async def _process_query_with_data(self, user_query: str, extraction_data: Dict) -> Dict[str, Any]:
        """Process the query with the extracted data context."""
        try:
            # Enhance the query with real data context
            enhanced_input = {
                "operation": "resolve_and_generate_code",
                "query": user_query,
                "data_context": {
                    "extracted_data": extraction_data["extracted_data"],
                    "file_info": extraction_data["file_info"],
                    "data_summary": extraction_data["data_summary"],
                    "extraction_strategy": extraction_data["extraction_strategy"]
                },
                "accuracy_mode": True,
                "avoid_assumptions": True
            }
            
            return await self.formula_resolver.process(enhanced_input)
            
        except Exception as e:
            return self.create_error_response(f"Query processing with data failed: {str(e)}")
    
    async def _execute_calculation(self, code: str, parameters: Dict, data_context: Dict) -> Dict[str, Any]:
        """Execute the calculation with the real data context."""
        try:
            execution_input = {
                "code": code,
                "parameters": parameters,
                "data_context": data_context,
                "validation_enabled": True
            }
            
            return await self.calc_executor.process(execution_input)
            
        except Exception as e:
            return self.create_error_response(f"Calculation execution failed: {str(e)}")
    
    def _calculate_accuracy_indicators(self, execution_log: Dict) -> Dict[str, Any]:
        """Calculate indicators of system accuracy and reliability."""
        
        indicators = {
            "data_source_accuracy": "high",  # Using real selected data
            "assumption_reliance": "low",    # Minimal assumptions
            "extraction_confidence": "high", # Intelligent extraction
            "total_steps_completed": len(execution_log.get("steps", [])),
            "failed_steps": 0,
            "success_rate": 1.0
        }
        
        # Count failed steps
        for step in execution_log.get("steps", []):
            if not step.get("result", {}).get("success", True):
                indicators["failed_steps"] += 1
        
        # Calculate success rate
        total_steps = indicators["total_steps_completed"]
        if total_steps > 0:
            indicators["success_rate"] = (total_steps - indicators["failed_steps"]) / total_steps
        
        # Adjust confidence based on success rate
        if indicators["success_rate"] < 0.8:
            indicators["extraction_confidence"] = "medium"
            indicators["data_source_accuracy"] = "medium"
        
        return indicators

    async def handle_user_file_selection(self, selection: Any) -> Dict[str, Any]:
        """Handle user's file selection input."""
        try:
            selection_result = await self.csv_selector.process({
                "action": "select",
                "selection": selection
            })
            
            return selection_result
            
        except Exception as e:
            return self.create_error_response(f"File selection handling failed: {str(e)}")
