"""
Simplified Task Manager - Orchestrates agents without AutoGen dependencies.
"""
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from core.base_agent import BaseAgent


class SimpleTaskManager:
    """
    Simplified task manager that orchestrates agents without AutoGen dependencies.
    """
    
    def __init__(self):
        self.agents = {}
        self.task_history = []
        self.logger = logging.getLogger(__name__)
        
    def register_agent(self, name: str, agent: BaseAgent):
        """Register an agent with the task manager."""
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name}")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task through the appropriate agents.
        
        Args:
            input_data: Input data containing query and context
            
        Returns:
            Processing result
        """
        try:
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.info(f"Processing task {task_id}")
            
            query = input_data.get("query", "")
            context = input_data.get("context", {})
            
            # Simple task routing based on query content
            result = await self._route_and_process(query, context, task_id)
            
            # Store in history
            self.task_history.append({
                "task_id": task_id,
                "query": query,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing task: {str(e)}")
            return {
                "success": False,
                "error": {
                    "message": str(e),
                    "type": "processing_error"
                }
            }
    
    async def _route_and_process(self, query: str, context: Dict, task_id: str) -> Dict[str, Any]:
        """Route query to appropriate agents and process."""
        query_lower = query.lower()
        
        try:
            # Step 1: Identify required metric/calculation
            if "FormulaResolver" in self.agents:
                formula_result = await self.agents["FormulaResolver"].process({
                    "query": query,
                    "context": context
                })
                
                if not formula_result.get("success"):
                    return formula_result
                
                metric_info = formula_result.get("data", {})
            else:
                # Fallback metric identification
                metric_info = self._identify_metric_simple(query)
            
            # Step 2: Check if we need external data
            needs_data = any(term in query_lower for term in ["ireland", "country", "region", "latest", "current"])
            
            if needs_data and "DataHarvester" in self.agents:
                data_result = await self.agents["DataHarvester"].process({
                    "query": query,
                    "metric": metric_info.get("metric_id", ""),
                    "context": context
                })
                
                if data_result.get("success"):
                    context.update(data_result.get("data", {}))
            
            # Step 3: Perform calculation if we have a formula
            if metric_info.get("metric_id") and "CalcExecutor" in self.agents:
                calc_result = await self.agents["CalcExecutor"].process({
                    "formula": metric_info.get("formula", ""),
                    "parameters": context.get("parameters", {}),
                    "metric_type": metric_info.get("metric_id", "")
                })
                
                if calc_result.get("success"):
                    context["calculation_result"] = calc_result.get("data", {})
            
            # Step 4: Quality control
            if "QCAuditor" in self.agents:
                qc_result = await self.agents["QCAuditor"].process({
                    "calculation_data": context,
                    "query": query
                })
                
                if not qc_result.get("success") or qc_result.get("data", {}).get("audit_status") != "approved":
                    return {
                        "success": False,
                        "error": {
                            "message": "Quality control failed",
                            "details": qc_result
                        }
                    }
            
            # Step 5: Generate final answer
            final_answer = self._generate_answer(query, context, metric_info)
            
            # Step 6: Export if requested
            export_format = context.get("export_format", "json")
            if "Exporter" in self.agents:
                export_result = await self.agents["Exporter"].process({
                    "data": final_answer,
                    "format": export_format
                })
                
                if export_result.get("success"):
                    final_answer["export"] = export_result.get("data", {})
            
            return {
                "success": True,
                "data": final_answer,
                "task_id": task_id,
                "processing_steps": ["formula_resolution", "data_collection", "calculation", "qc_audit", "export"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "message": f"Processing failed: {str(e)}",
                    "type": "routing_error"
                }
            }
    
    def _identify_metric_simple(self, query: str) -> Dict[str, Any]:
        """Simple metric identification without FormulaResolver."""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["lcoe", "levelized cost", "cost of energy"]):
            return {
                "metric_id": "lcoe",
                "name": "Levelized Cost of Energy",
                "formula": "(CAPEX + sum(OPEX_t / (1 + discount_rate)^t)) / sum(energy_output_t / (1 + discount_rate)^t)"
            }
        elif any(term in query_lower for term in ["capacity factor", "cf", "utilization"]):
            return {
                "metric_id": "capacity_factor",
                "name": "Capacity Factor",
                "formula": "actual_energy_output / (nameplate_capacity * hours_in_period)"
            }
        elif any(term in query_lower for term in ["npv", "net present value"]):
            return {
                "metric_id": "npv",
                "name": "Net Present Value",
                "formula": "sum((cash_flow_t - investment_t) / (1 + discount_rate)^t)"
            }
        else:
            return {
                "metric_id": "general",
                "name": "General Query",
                "formula": None
            }
    
    def _generate_answer(self, query: str, context: Dict, metric_info: Dict) -> Dict[str, Any]:
        """Generate final answer based on processing results."""
        answer = {
            "query": query,
            "metric": metric_info.get("name", "Unknown"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add calculation result if available
        if "calculation_result" in context:
            calc_result = context["calculation_result"]
            answer["result"] = calc_result.get("result", "No result available")
            answer["calculation_details"] = calc_result
        
        # Add data sources if available
        if "data_sources" in context:
            answer["data_sources"] = context["data_sources"]
        
        # Generate text answer
        if metric_info.get("metric_id") == "lcoe" and "calculation_result" in context:
            result_value = context["calculation_result"].get("result", 0)
            answer["answer"] = f"The Levelized Cost of Energy (LCOE) is ${result_value:.2f}/MWh based on the provided parameters."
        elif metric_info.get("metric_id") == "capacity_factor" and "calculation_result" in context:
            result_value = context["calculation_result"].get("result", 0)
            answer["answer"] = f"The capacity factor is {result_value:.1%} based on the provided parameters."
        else:
            answer["answer"] = f"Analysis completed for: {query}. Please refer to the detailed results for more information."
        
        return answer
    
    async def process_direct_agent_call(self, agent_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Directly call a specific agent."""
        if agent_name not in self.agents:
            return {
                "success": False,
                "error": {
                    "message": f"Agent {agent_name} not found",
                    "type": "agent_not_found"
                }
            }
        
        try:
            return await self.agents[agent_name].process(input_data)
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "message": str(e),
                    "type": "agent_error"
                }
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "agents_registered": len(self.agents),
            "agent_names": list(self.agents.keys()),
            "tasks_processed": len(self.task_history),
            "status": "operational",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get task history."""
        return self.task_history

