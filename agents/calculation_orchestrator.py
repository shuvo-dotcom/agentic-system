"""
Calculation Orchestrator - Coordinates the 3 main calculation agents in a round-robin fashion
with safety mechanisms and data validation between each step.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from agents.autogen_parser_agent import AutoGenParserAgent
from agents.llm_formula_resolver import LLMFormulaResolver
from agents.calc_executor import CalcExecutor
from agents.messages import TextMessage


@dataclass
class AgentStep:
    """Represents a step in the calculation pipeline"""
    name: str
    agent: Any
    input_validator: callable
    output_validator: callable
    error_handler: callable


class CalculationOrchestrator:
    """
    Orchestrates the 3 main calculation agents:
    1. DataExtractor (formerly AutoGenParserAgent)
    2. FormulaGenerator (formerly LLMFormulaResolver) 
    3. ComputationEngine (formerly CalcExecutor)
    
    Implements round-robin chaining with safety mechanisms between each step.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.execution_history = []
        
        # Initialize the 3 agents with new names
        self.data_extractor = AutoGenParserAgent()
        self.formula_generator = LLMFormulaResolver()
        self.computation_engine = CalcExecutor()
        
        # Define the calculation pipeline with safety mechanisms
        self.pipeline = [
            AgentStep(
                name="DataExtractor",
                agent=self.data_extractor,
                input_validator=self._validate_query_input,
                output_validator=self._validate_extracted_data,
                error_handler=self._handle_extraction_error
            ),
            AgentStep(
                name="FormulaGenerator", 
                agent=self.formula_generator,
                input_validator=self._validate_formula_input,
                output_validator=self._validate_generated_code,
                error_handler=self._handle_generation_error
            ),
            AgentStep(
                name="ComputationEngine",
                agent=self.computation_engine,
                input_validator=self._validate_computation_input,
                output_validator=self._validate_computation_result,
                error_handler=self._handle_computation_error
            )
        ]
    
    async def process_calculation(self, query: str, context: Dict[str, Any] = None, session_id: int = None, agent_name: str = 'CalculationOrchestrator') -> Dict[str, Any]:
        """
        Process a calculation query through the 3-agent pipeline with safety mechanisms.
        
        Args:
            query: Natural language calculation query
            context: Additional context for the calculation
            session_id: Session identifier for logging
            agent_name: Name of the agent for logging
        Returns:
            Final calculation result with full pipeline history
        """
        # Add session_id and agent_name to all log records
        class ContextFilter(logging.Filter):
            def filter(self, record):
                record.session_id = session_id
                record.agent_name = agent_name
                return True
        self.logger.addFilter(ContextFilter())

        start_time = datetime.now()
        pipeline_result = {
            "success": False,
            "query": query,
            "context": context or {},
            "pipeline_steps": [],
            "final_result": None,
            "execution_time": None,
            "errors": []
        }
        
        try:
            self.logger.info(f"Starting calculation pipeline for query: {query[:100]}...", extra={'session_id': session_id, 'agent_name': agent_name})
            
            # Initialize pipeline context
            pipeline_context = {
                "query": query,
                "context": context or {},
                "extracted_data": None,
                "generated_code": None,
                "computation_result": None
            }
            
            # Execute pipeline in round-robin fashion
            for i, step in enumerate(self.pipeline):
                step_result = await self._execute_pipeline_step(step, pipeline_context, i, session_id, agent_name)
                pipeline_result["pipeline_steps"].append(step_result)
                
                # Check if step failed
                if not step_result["success"]:
                    pipeline_result["errors"].append(step_result["error"])
                    break
                
                # Update pipeline context for next step
                pipeline_context = self._update_pipeline_context(pipeline_context, step_result)
                
                # After each pipeline step, log accept or retry/fallback as appropriate
                self.log_decision_point(
                    session_id=session_id,
                    agent_name=agent_name,
                    step_id=f'step_{i}',
                    parent_step_id=None,
                    decision_type='accept' if step_result['success'] else 'retry',
                    reason='Pipeline step completed' if step_result['success'] else 'Step failed, retrying or aborting',
                    result_summary=str(step_result),
                    confidence=None,
                    parameters=step_result.get('input_data'),
                    llm_rationale=None
                )
            
            # Check if all steps completed successfully
            if len(pipeline_result["errors"]) == 0:
                pipeline_result["success"] = True
                pipeline_result["final_result"] = pipeline_context.get("computation_result")
            
            # Calculate execution time
            end_time = datetime.now()
            pipeline_result["execution_time"] = (end_time - start_time).total_seconds()
            
            # Log the execution
            self.execution_history.append(pipeline_result)
            
            # On finalization
            self.log_decision_point(
                session_id=session_id,
                agent_name=agent_name,
                step_id='finalize',
                parent_step_id=None,
                decision_type='finalize',
                reason='Pipeline completed',
                result_summary=str(pipeline_result),
                confidence=None,
                parameters=None,
                llm_rationale=None
            )
            
            return pipeline_result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}", extra={'session_id': session_id, 'agent_name': agent_name})
            pipeline_result["errors"].append({
                "step": "pipeline_orchestration",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return pipeline_result
    
    async def _execute_pipeline_step(self, step: AgentStep, context: Dict, step_index: int, session_id: int, agent_name: str) -> Dict[str, Any]:
        """
        Execute a single pipeline step with safety mechanisms.
        
        Args:
            step: The agent step to execute
            context: Current pipeline context
            step_index: Index of the current step
            session_id: Session identifier for logging
            agent_name: Name of the agent for logging
        Returns:
            Step execution result
        """
        step_result = {
            "step_name": step.name,
            "step_index": step_index,
            "success": False,
            "input_data": None,
            "output_data": None,
            "execution_time": None,
            "error": None,
            "validation_passed": False
        }
        
        start_time = datetime.now()
        
        try:
            # Step 1: Prepare input data
            input_data = self._prepare_step_input(step, context)
            step_result["input_data"] = input_data
            
            # Step 2: Validate input
            if not step.input_validator(input_data, context):
                raise ValueError(f"Input validation failed for {step.name}")
            
            # Step 3: Execute agent
            if step.name == "DataExtractor":
                output_data = await step.agent.on_message(TextMessage(content=input_data["query"], source="user"))
            elif step.name == "FormulaGenerator":
                output_data = await step.agent.process(input_data)
            elif step.name == "ComputationEngine":
                output_data = await step.agent.process(input_data)
            else:
                raise ValueError(f"Unknown agent step: {step.name}")
            
            step_result["output_data"] = output_data
            
            # Step 4: Validate output
            if not step.output_validator(output_data, context):
                raise ValueError(f"Output validation failed for {step.name}")
            
            step_result["validation_passed"] = True
            step_result["success"] = True
            
        except Exception as e:
            # Step 5: Handle errors
            error_info = step.error_handler(e, context, step_result)
            step_result["error"] = error_info
            
        finally:
            # Calculate execution time
            end_time = datetime.now()
            step_result["execution_time"] = (end_time - start_time).total_seconds()
        
        return step_result
    
    def _prepare_step_input(self, step: AgentStep, context: Dict) -> Dict[str, Any]:
        """Prepare input data for each step based on the pipeline context."""
        if step.name == "DataExtractor":
            return {
                "query": context["query"],
                "context": context["context"]
            }
        elif step.name == "FormulaGenerator":
            return {
                "operation": "resolve_and_generate_code",
                "query": context["query"],
                "extracted_parameters": context.get("extracted_data", {})
            }
        elif step.name == "ComputationEngine":
            return {
                "operation": "run_python_code",
                "code": context.get("generated_code", ""),
                "inputs": context.get("extracted_data", {})
            }
        else:
            return {}
    
    def _update_pipeline_context(self, context: Dict, step_result: Dict) -> Dict:
        """Update pipeline context based on step result."""
        if step_result["step_name"] == "DataExtractor" and step_result["success"]:
            # DataExtractor returns parameters as a list, convert to dict for easier handling
            parameters_list = step_result["output_data"].get("parameters", [])
            parameters_dict = {}
            for param in parameters_list:
                if isinstance(param, dict) and "name" in param and "value" in param:
                    parameters_dict[param["name"]] = param["value"]
            context["extracted_data"] = parameters_dict
        elif step_result["step_name"] == "FormulaGenerator" and step_result["success"]:
            context["generated_code"] = step_result["output_data"].get("data", {}).get("executable_code", "")
        elif step_result["step_name"] == "ComputationEngine" and step_result["success"]:
            context["computation_result"] = step_result["output_data"].get("results", {})
        
        return context
    
    # Safety Mechanisms - Input Validators
    def _validate_query_input(self, input_data: Dict, context: Dict) -> bool:
        """Validate input for DataExtractor."""
        if not input_data.get("query"):
            return False
        if len(input_data["query"].strip()) < 10:
            return False
        return True
    
    def _validate_formula_input(self, input_data: Dict, context: Dict) -> bool:
        """Validate input for FormulaGenerator."""
        if not input_data.get("query"):
            return False
        if not input_data.get("operation"):
            return False
        return True
    
    def _validate_computation_input(self, input_data: Dict, context: Dict) -> bool:
        """Validate input for ComputationEngine."""
        if not input_data.get("code"):
            return False
        if len(input_data["code"].strip()) < 10:
            return False
        return True
    
    # Safety Mechanisms - Output Validators
    def _validate_extracted_data(self, output_data: Dict, context: Dict) -> bool:
        """Validate output from DataExtractor."""
        if not output_data.get("success"):
            return False
        if not output_data.get("parameters"):
            return False
        if len(output_data["parameters"]) == 0:
            return False
        return True
    
    def _validate_generated_code(self, output_data: Dict, context: Dict) -> bool:
        """Validate output from FormulaGenerator."""
        if not output_data.get("success"):
            return False
        if not output_data.get("data", {}).get("executable_code"):
            return False
        if len(output_data["data"]["executable_code"].strip()) < 20:
            return False
        return True
    
    def _validate_computation_result(self, output_data: Dict, context: Dict) -> bool:
        """Validate output from ComputationEngine."""
        if not output_data.get("success"):
            return False
        if not output_data.get("results"):
            return False
        # Check if there's a result value (either 'result' or any numeric value)
        results = output_data["results"]
        if "result" in results:
            return True
        # If no 'result' key, check if there are any numeric values
        for value in results.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
        return False
    
    # Safety Mechanisms - Error Handlers
    def _handle_extraction_error(self, error: Exception, context: Dict, step_result: Dict) -> Dict:
        """Handle errors in DataExtractor."""
        return {
            "type": "extraction_error",
            "message": str(error),
            "suggestion": "Try rephrasing the query with more explicit parameter values",
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_generation_error(self, error: Exception, context: Dict, step_result: Dict) -> Dict:
        """Handle errors in FormulaGenerator."""
        return {
            "type": "generation_error", 
            "message": str(error),
            "suggestion": "Check if the query contains enough information for formula generation",
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_computation_error(self, error: Exception, context: Dict, step_result: Dict) -> Dict:
        """Handle errors in ComputationEngine."""
        return {
            "type": "computation_error",
            "message": str(error),
            "suggestion": "Verify that all required parameters have valid numeric values",
            "timestamp": datetime.now().isoformat()
        }
    
    def log_decision_point(self, session_id, agent_name, step_id, parent_step_id, decision_type, reason, result_summary, sub_prompt=None, stage='decision_point', confidence=None, parameters=None, llm_rationale=None, extra_context=None):
        log_data = {
            'session_id': session_id,
            'agent_name': agent_name,
            'stage': stage,
            'step_id': step_id,
            'parent_step_id': parent_step_id,
            'decision_type': decision_type,
            'decision_reason': reason,
            'decision_result': result_summary,
            'sub_prompt': sub_prompt,
            'confidence': confidence,
            'parameters': parameters,
            'llm_rationale': llm_rationale,
        }
        if extra_context:
            log_data.update(extra_context)
        self.logger.info(
            f"Decision: {decision_type} | Reason: {reason} | Result: {result_summary} | Confidence: {confidence} | Parameters: {parameters} | LLM Rationale: {llm_rationale}",
            extra=log_data
        )
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the current status of the calculation pipeline."""
        return {
            "pipeline_name": "Calculation Pipeline",
            "agents": [
                {"name": "DataExtractor", "description": "Extracts parameters from natural language queries"},
                {"name": "FormulaGenerator", "description": "Generates executable code for calculations"},
                {"name": "ComputationEngine", "description": "Executes calculations in a safe sandbox"}
            ],
            "total_executions": len(self.execution_history),
            "successful_executions": len([h for h in self.execution_history if h["success"]]),
            "last_execution": self.execution_history[-1] if self.execution_history else None
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict]:
        """Get recent execution history."""
        return self.execution_history[-limit:] if self.execution_history else []


# Convenience function for easy usage
async def calculate(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function to perform calculations using the orchestrated pipeline.
    
    Args:
        query: Natural language calculation query
        context: Additional context for the calculation
        
    Returns:
        Calculation result with full pipeline details
    """
    orchestrator = CalculationOrchestrator()
    return await orchestrator.process_calculation(query, context) 