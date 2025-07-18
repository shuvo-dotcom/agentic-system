"""
Task Manager - Central orchestrator for the agentic system.
"""
import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.base_agent import BaseAgent
from config.settings import OPENAI_API_KEY, OPENAI_MODEL


class TaskManager(BaseAgent):
    """
    Central task manager that orchestrates the entire agentic workflow.
    Parses user queries, decomposes them into atomic tasks, and coordinates specialist agents.
    """
    
    def __init__(self):
        super().__init__(
            name="TaskManager",
            description="Central orchestrator that parses user queries, decomposes them into atomic tasks, and coordinates specialist agents."
        )
        self.agents = {}
        self.task_history = []
        
    def register_agent(self, agent_name: str, agent_instance: BaseAgent):
        """Register a specialist agent with the task manager."""
        self.agents[agent_name] = agent_instance
        self.logger.info(f"Registered agent: {agent_name}")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method that handles user queries.
        
        Args:
            input_data: Dictionary containing 'query' and optional 'context'
            
        Returns:
            Final processed result
        """
        try:
            if not self.validate_input(input_data, ["query"]):
                return self.create_error_response("Missing required field: query")
            
            user_query = input_data["query"]
            context = input_data.get("context", {})
            
            self.log_activity("Processing user query", {"query": user_query})
            
            # Step 1: Parse and decompose the query
            task_plan = await self._decompose_query(user_query, context)
            if not task_plan["success"]:
                return task_plan
            
            # Step 2: Execute the task plan
            execution_result = await self._execute_task_plan(task_plan["data"])
            if not execution_result["success"]:
                return execution_result
            
            # Step 3: Merge and synthesize results
            final_result = await self._synthesize_results(execution_result["data"], user_query)
            
            # Store in task history
            self.task_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "task_plan": task_plan["data"],
                "result": final_result
            })
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return self.create_error_response(f"Processing failed: {str(e)}")
    
    async def _decompose_query(self, query: str, context: Dict) -> Dict[str, Any]:
        """
        Decompose user query into atomic tasks using LLM.
        """
        try:
            system_prompt = """
            You are a task decomposition expert. Given a user query, break it down into atomic tasks that can be executed by specialist agents.
            
            Available agents and their capabilities:
            - PlexusCSVLoader: Query Plexus API and ingest CSV data
            - DataHarvester: Pull data from external public sources (Eurostat, IEA, SEAI, ENTSO-E)
            - DocCrawler: Fetch and extract text from PDFs and documents
            - SchemaMapper: Harmonize data schemas and validate data quality
            - RAGIndexer: Build searchable vector and SQL indices
            - FormulaResolver: Identify required metrics and retrieve formulas
            - ParameterPlanner: Plan data retrieval for formula parameters
            - CalcExecutor: Execute numerical computations
            - InsightComposer: Generate user-friendly summaries
            - QCAuditor: Validate results for accuracy and consistency
            - Exporter: Format final output
            
            Return a JSON object with the following structure:
            {
                "tasks": [
                    {
                        "id": "task_1",
                        "agent": "AgentName",
                        "action": "specific_action",
                        "inputs": {"key": "value"},
                        "dependencies": ["task_id_1", "task_id_2"],
                        "priority": 1
                    }
                ],
                "execution_order": ["task_1", "task_2", "task_3"],
                "expected_output": "description of expected final output"
            }
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: {query}\nContext: {json.dumps(context)}")
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse the JSON response
            task_plan = json.loads(response.content)
            
            self.log_activity("Query decomposed", {"task_count": len(task_plan["tasks"])})
            
            return self.create_success_response(task_plan)
            
        except Exception as e:
            self.logger.error(f"Error decomposing query: {str(e)}")
            return self.create_error_response(f"Query decomposition failed: {str(e)}")
    
    async def _execute_task_plan(self, task_plan: Dict) -> Dict[str, Any]:
        """
        Execute the task plan by coordinating specialist agents.
        """
        try:
            tasks = task_plan["tasks"]
            execution_order = task_plan["execution_order"]
            results = {}
            
            self.log_activity("Starting task execution", {"task_count": len(tasks)})
            
            # Create task lookup
            task_lookup = {task["id"]: task for task in tasks}
            
            # Execute tasks in order, respecting dependencies
            for task_id in execution_order:
                task = task_lookup[task_id]
                agent_name = task["agent"]
                
                # Check if agent is available
                if agent_name not in self.agents:
                    self.logger.warning(f"Agent {agent_name} not available, skipping task {task_id}")
                    continue
                
                # Prepare inputs, including results from dependencies
                task_inputs = task["inputs"].copy()
                for dep_id in task.get("dependencies", []):
                    if dep_id in results:
                        task_inputs[f"dependency_{dep_id}"] = results[dep_id]
                
                # Execute task
                self.log_activity(f"Executing task {task_id}", {"agent": agent_name})
                
                try:
                    agent = self.agents[agent_name]
                    result = await agent.process(task_inputs)
                    results[task_id] = result
                    
                    if not result.get("success", False):
                        self.logger.error(f"Task {task_id} failed: {result.get('error', {}).get('message', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.error(f"Error executing task {task_id}: {str(e)}")
                    results[task_id] = self.create_error_response(f"Task execution failed: {str(e)}")
            
            return self.create_success_response(results)
            
        except Exception as e:
            self.logger.error(f"Error executing task plan: {str(e)}")
            return self.create_error_response(f"Task plan execution failed: {str(e)}")
    
    async def _synthesize_results(self, execution_results: Dict, original_query: str) -> Dict[str, Any]:
        """
        Synthesize results from all tasks into a final response.
        """
        try:
            system_prompt = """
            You are a result synthesizer. Given the results from various specialist agents, 
            create a comprehensive and user-friendly response to the original query.
            
            Focus on:
            1. Answering the user's question directly
            2. Providing relevant data and calculations
            3. Including appropriate citations and sources
            4. Highlighting any limitations or assumptions
            5. Formatting the response clearly
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                Original Query: {original_query}
                
                Agent Results:
                {json.dumps(execution_results, indent=2)}
                
                Please synthesize these results into a comprehensive response.
                """)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            self.log_activity("Results synthesized")
            
            return self.create_success_response({
                "answer": response.content,
                "query": original_query,
                "agent_results": execution_results,
                "synthesis_timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error synthesizing results: {str(e)}")
            return self.create_error_response(f"Result synthesis failed: {str(e)}")
    
    def get_task_history(self) -> List[Dict]:
        """Get the history of processed tasks."""
        return self.task_history
    
    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent names."""
        return list(self.agents.keys())

