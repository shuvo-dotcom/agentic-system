"""
Main application entry point for the Agentic System.
"""
import asyncio
import json
import logging
import re
from typing import Dict, Any

from core.simple_task_manager import SimpleTaskManager
from core.simple_base_agent import SimpleBaseAgent
from agents.plexos_csv_loader import PlexosCSVLoader
from agents.rag_indexer import RAGIndexer
from agents.llm_formula_resolver import LLMFormulaResolver # New import
from agents.calc_executor import CalcExecutor
from agents.qc_auditor import QCAuditor
from agents.exporter import Exporter

from config.settings import LOG_LEVEL


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/agentic_system.log")
        ]
    )


class AgenticSystem:
    """
    Main agentic system that orchestrates all agents.
    """
    
    def __init__(self):
        self.task_manager = SimpleTaskManager()
        self.agents = {}
        self.logger = logging.getLogger(__name__)
        self.name = "AgenticSystem" # Added this line
        
        # Initialize and register all agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize and register all specialist agents."""
        try:
            # Create agent instances
            self.agents = {
                "PlexosCSVLoader": PlexosCSVLoader(),
                "RAGIndexer": RAGIndexer(),
                "LLMFormulaResolver": LLMFormulaResolver(), # Replaced FormulaResolver
                "CalcExecutor": CalcExecutor(),
                "QCAuditor": QCAuditor(),
                "Exporter": Exporter()
            }
            
            # Register agents with task manager
            for agent_name, agent_instance in self.agents.items():
                self.task_manager.register_agent(agent_name, agent_instance)
            
            self.logger.info(f"Initialized {len(self.agents)} agents successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user query through the agentic system.
        
        Args:
            query: User query string
            context: Optional context information
            
        Returns:
            Processed result from the system
        """
        try:
            self.logger.info(f"Processing query: {query[:100]}...")
            
            # Step 1: Use LLMFormulaResolver to dynamically resolve formula, parameters, and generate code
            llm_resolver = self.agents["LLMFormulaResolver"]
            llm_resolution_result = await llm_resolver.process({"operation": "resolve_and_generate_code", "query": query})
            
            if not llm_resolution_result.get("success"):
                error_msg = llm_resolution_result.get("error", "Unknown error")
                return self.create_error_response(f"LLM formula resolution failed: {error_msg}")

            resolved_data = llm_resolution_result["data"]
            metric_name = resolved_data.get("metric_name", "Unknown Metric")
            formula_str = resolved_data.get("formula")
            parameters_info = resolved_data.get("parameters")
            executable_code = resolved_data.get("executable_code")

            if not executable_code:
                return self.create_error_response(f"LLM did not generate executable code for query: {query}")

            self.logger.info(f"Identified metric: {metric_name}")
            self.logger.info(f"Generated executable code: {executable_code[:200]}...")

            # Step 2: Execute the generated code using CalcExecutor
            calc_executor = self.agents["CalcExecutor"]
            calculation_result = await calc_executor.process({
                "operation": "run_python_code",
                "code": executable_code,
                "inputs": {}
            })

            if not calculation_result.get("success"):
                error_message = calculation_result.get("error", "Unknown calculation error")
                return self.create_error_response(f"Calculation failed: {error_message}")
            
            # Extract the final result from the executed code\'s output
            calculated_value = calculation_result.get("results", {}).get("result")
            if calculated_value is None:
                # Attempt to parse from output if not directly in results
                output_lines = calculation_result.get("output", "").strip().split('\n')
                if output_lines:
                    try:
                        # Look for a line that might contain the final result, e.g., \'result = 123.45\'
                        for line in reversed(output_lines):
                            if 'result =' in line:
                                calculated_value = float(line.split('=')[-1].strip())
                                break
                    except ValueError:
                        pass # Could not parse, continue with None

            if calculated_value is None:
                return self.create_error_response(f"Could not extract calculated value from code execution output.")

            # Step 3: Audit results using QCAuditor (if applicable, or adapt for dynamic)
            # For now, we\'ll just pass the calculated value directly
            audit_result = {"validation": {"overall_status": "Skipped (dynamic calculation)"}}

            # Step 4: Format and export results using Exporter
            exporter = self.agents["Exporter"]
            final_report = {
                "query": query,
                "metric": {"name": metric_name, "id": metric_name.lower().replace(" ", "_")},
                "formula_info": {"formula": formula_str, "parameters": parameters_info},
                "parameters_used": {k: v.get("value") for k, v in parameters_info.items()},
                "calculation_result": calculated_value,
                "audit_status": audit_result.get("validation", {}).get("overall_status"),
                "audit_details": audit_result.get("validation", {})
            }
            export_result = await exporter.process({"operation": "export_json", "data": final_report})

            query_success = export_result.get("success", False)
            self.logger.info(f"Query processing completed. Success: {query_success}")
            
            audit_status = audit_result.get("validation", {}).get("overall_status")
            return {
                "success": export_result.get("success", False),
                "data": {
                    "answer": f"The calculated {metric_name} is {calculated_value:.2f}. Audit status: {audit_status}.",
                    "report": export_result.get("content"),
                    "filename": export_result.get("filename")
                },
                "metadata": {
                    "processed_by": self.name,
                    "agents_involved": ["LLMFormulaResolver", "CalcExecutor", "QCAuditor", "Exporter"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": {
                    "message": str(e),
                    "type": "system_error"
                }
            }
    
    async def direct_agent_call(self, agent_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Directly call a specific agent (useful for testing).
        
        Args:
            agent_name: Name of the agent to call
            input_data: Input data for the agent
            
        Returns:
            Agent processing result
        """
        try:
            return await self.task_manager.process_direct_agent_call(agent_name, input_data)
        except Exception as e:
            self.logger.error(f"Error in direct agent call: {str(e)}")
            return {
                "success": False,
                "error": {
                    "message": str(e),
                    "type": "agent_error"
                }
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return self.task_manager.get_system_status()
    
    def get_task_history(self) -> list:
        """Get task processing history."""
        return self.task_manager.get_task_history()

    def create_error_response(self, message: str, error_type: str = "processing_error") -> Dict[str, Any]:
        """
        Helper to create a standardized error response.
        """
        return {
            "success": False,
            "error": {
                "message": message,
                "type": error_type
            }
        }


async def main():
    """
    Main function for testing the system.
    """
    setup_logging()
    print("Starting main function...")
    
    # Initialize the system
    system = AgenticSystem()
    print("AgenticSystem instance created.")
    
    print("Agentic System initialized successfully!")
    print(f"System status: {system.get_system_status()}")
    
    # Example queries for testing
    test_queries = [
        "Calculate the LCOE for a wind farm with CAPEX of $2000/kW, OPEX of $50/kW/year, capacity factor of 35%, and 20-year lifetime with 8% discount rate",
        "What is the Net Present Value of a project with initial investment of $100, cash flows of $20, $30, $40, $50 over 4 years, and a discount rate of 10%?",
        "What is the capacity factor of a solar plant that produced 500 MWh in a year, with a nameplate capacity of 2 MW?",
        "Calculate the internal rate of return for cash flows of -100, 20, 30, 40, 50."
    ]
    
    print("\nTesting with example queries:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Query: {query}")
        
        try:
            result = await system.process_query(query)
            
            if result.get("success"):
                print("✓ Query processed successfully")
                if "data" in result and "answer" in result["data"]:
                    answer_text = result["data"]["answer"]
                    print(f"Answer: {answer_text[:200]}...")
            else:
                print("✗ Query processing failed")
                error_message = result.get("error", {}).get("message", "Unknown error")
                print(f"Error: {error_message}")
                
        except Exception as e:
            print(f"✗ Exception occurred: {str(e)}")
    
    print(f"\nTask history: {len(system.get_task_history())} tasks processed")
    print("Main function finished.")


if __name__ == "__main__":
    asyncio.run(main())




