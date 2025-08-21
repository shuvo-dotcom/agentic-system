import logging
from typing import List, Dict, Any, Optional
from agents.interaction_agent.prompt_decomposer import PromptDecomposerAgent
from agents.interaction_agent.human_simulator import HumanSimulator
from agents.time_series_agent.time_series_agent import DynamicTimeSeriesAgent
from agents.interaction_agent.prompt_type_detection_agent import PromptTypeDetectionAgent
from agents.pinpoint_value_agent import PinpointValueAgent
from config.settings import OPENAI_API_KEY, DELETE_LOG_FILE_AFTER_UPLOAD
import re
import os
import time
from agents.log_agent.log_handler import LogHandler
from agents.log_agent.config import get_log_agent_config
from datetime import datetime
import json

# Type patterns for dynamic detection (copied from previous integration logic)
TYPE_PATTERNS = {
    "int": [
        "how many years", "number of years", "years would you like", "enter a year", "years to forecast", "number of periods"
    ],
    "float": [
        "rate", "percentage", "cost", "factor", "value for .*cost", "value for .*rate", "discount", "efficiency", "capacity factor"
    ],
    "yes/no": [
        "yes/no", "do you want", "would you like", "confirm", "use default", "use llm-suggested defaults"
    ],
    "string": [
        "name", "description", "label", "enter a description", "type of", "specify"
    ]
}

def detect_type_from_prompt(prompt):
    for type_name, patterns in TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return type_name
    return None

def serialize_result(result):
    """Recursively serialize result object to dictionary for MongoDB storage"""
    if isinstance(result, dict):
        return {k: serialize_result(v) for k, v in result.items()}
    elif isinstance(result, list):
        return [serialize_result(v) for v in result]
    elif hasattr(result, 'to_dict'):
        return serialize_result(result.to_dict())
    elif hasattr(result, '__dict__'):
        return {k: serialize_result(v) for k, v in result.__dict__.items()}
    else:
        return result

class OrchestratorAgent:
    """
    High-level agent that orchestrates multi-entity, multi-metric, multi-time analysis.
    Decomposes complex prompts, invokes the interaction agent and time series agent for each atomic sub-prompt,
    aggregates results, and returns a comprehensive answer (table + summary).
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.decomposer = PromptDecomposerAgent(api_key=self.api_key)
        self.interaction_agent = HumanSimulator(api_key=self.api_key)
        self.time_series_agent = DynamicTimeSeriesAgent()
        self.pinpoint_value_agent = PinpointValueAgent(api_key=self.api_key)
        self.prompt_type_detector = PromptTypeDetectionAgent()
        self.logger = logging.getLogger(__name__)

    def log_decision_point(self, session_id, agent_name, step_id, parent_step_id, decision_type, reason, result_summary, sub_prompt=None, stage='decision_point', confidence=None, parameters=None, llm_rationale=None, extra_context=None, depth=None, parent_id=None):
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
        if depth is not None:
            log_data['depth'] = depth
        if parent_id is not None:
            log_data['parent_id'] = parent_id
        if extra_context:
            log_data.update(extra_context)
        self.logger.info(
            f"Decision: {decision_type} | Reason: {reason} | Result: {result_summary} | Confidence: {confidence} | Parameters: {parameters} | LLM Rationale: {llm_rationale}",
            extra=log_data
        )

    def log_agent_call(self, call_tree, node_id, agent_name, operation, parent_id, parameters=None, result=None):
        call_tree[node_id] = {
            'agent_name': agent_name,
            'operation': operation,
            'parent_id': parent_id,
            'parameters': parameters,
            'result': result
        }

    def log_decision_node_to_tree(self, call_tree, node_id, agent_name, decision_type, parent_id, reason, result_summary, sub_prompt=None, stage='decision_point', confidence=None, parameters=None, llm_rationale=None):
        call_tree[node_id] = {
            'agent_name': agent_name,
            'operation': f'decision_{decision_type}',
            'parent_id': parent_id,
            'parameters': parameters,
            'result': result_summary,
            'decision_type': decision_type,
            'reason': reason,
            'sub_prompt': sub_prompt,
            'stage': stage,
            'confidence': confidence,
            'llm_rationale': llm_rationale
        }

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Orchestrate the full analysis for a complex prompt.
        Returns a dict with all sub-results and a summary table.
        """
        # Set up session log file
        session_id = int(time.time())
        agent_name = 'OrchestratorAgent'
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f'session_{session_id}.log')
        print(f"[DEBUG] Log file path: {log_file_path}")
        print(f"[DEBUG] Session ID: {session_id}")
        file_handler = logging.FileHandler(log_file_path)
        # Add SafeFormatter to handle missing session_id/agent_name
        class SafeFormatter(logging.Formatter):
            def format(self, record):
                if not hasattr(record, 'session_id'):
                    record.session_id = 'N/A'
                if not hasattr(record, 'agent_name'):
                    record.agent_name = 'N/A'
                return super().format(record)
        formatter = SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - [session_id:%(session_id)s] [agent:%(agent_name)s] [step_id:%(step_id)s] [parent_id:%(parent_id)s] - %(message)s')
        file_handler.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        # Add session_id and agent_name to all log records
        class ContextFilter(logging.Filter):
            def filter(self, record):
                record.session_id = session_id
                record.agent_name = agent_name
                return True

        class DefaultFieldFilter(logging.Filter):
            def filter(self, record):
                if not hasattr(record, 'step_id'):
                    record.step_id = 'N/A'
                if not hasattr(record, 'parent_id'):
                    record.parent_id = 'N/A'
                return True

        logger.addFilter(ContextFilter())
        logger.addFilter(DefaultFieldFilter())
        print(f"[DEBUG] ContextFilter and DefaultFieldFilter added: session_id={session_id}, agent_name={agent_name}")

        # Initialize log handler and mongo_log_handler at the start
        config = get_log_agent_config()
        log_handler = LogHandler(
            mongo_uri=config.get('mongo_uri', ''),
            mongo_db=config.get('mongo_db', ''),
            mongo_collection=config.get('mongo_collection', ''),
            qdrant_url=config['qdrant_url'],
            openai_api_key=config['openai_api_key'],
            qdrant_collection=config.get('qdrant_collection', 'agent_logs')
        )
        mongo_log_handler = log_handler.mongo_logs

        step_id = 0
        call_tree = {}
        node_counter = 0
        node_ids = {}  # label -> step_id mapping for parent tracking
        def get_node_id(label):
            nonlocal node_counter
            node_counter += 1
            node_id = f'{label}_{node_counter}'
            node_ids[label] = node_id
            return node_id

        self.logger.info(f"OrchestratorAgent received prompt: {prompt}", extra={'session_id': session_id, 'agent_name': agent_name, 'stage': 'prompt_received', 'step_id': 'root', 'parent_step_id': '', 'parent_id': ''})
        sub_prompts = self.decomposer.decompose(prompt)
        results = []
        table = []
        parent_id = None
        last_decision_node_id = None
        print(f"[DEBUG] About to start sub-prompt loop. sub_prompts={sub_prompts}")
        for sub in sub_prompts:
            print(f"[DEBUG] Processing sub-prompt: {sub}")
            step_id += 1
            # For the first sub-prompt, parent_id is None; for others, parent_id is last_decision_node_id
            sub_prompt_parent_id = last_decision_node_id if last_decision_node_id else None
            sub_prompt_node_id = get_node_id('subprompt')
            
            # Detect prompt type
            prompt_type = self.prompt_type_detector.detect_type(sub)
            self.logger.info(f"Detected prompt type '{prompt_type}' for sub-prompt: {sub}")

            self.logger.info(f"OrchestratorAgent running sub-prompt: {sub}", extra={'session_id': session_id, 'agent_name': agent_name, 'stage': 'run_sub_prompt', 'step_id': sub_prompt_node_id, 'parent_step_id': str(sub_prompt_parent_id) if sub_prompt_parent_id else '', 'parent_id': str(sub_prompt_parent_id) if sub_prompt_parent_id else '', 'prompt_type': prompt_type})
            self.log_agent_call(call_tree, sub_prompt_node_id, agent_name, 'run_sub_prompt', str(sub_prompt_parent_id) if sub_prompt_parent_id else '', parameters={'sub_prompt': sub})
            import builtins
            original_input = builtins.input
            def simulated_input(agent_prompt):
                expected_type = detect_type_from_prompt(agent_prompt)
                response_obj = self.interaction_agent.generate_response(agent_prompt)
                response_text = response_obj.response
                if expected_type == "int":
                    match = re.search(r"\b\d+\b", response_text)
                    if match:
                        return match.group(0)
                elif expected_type == "float":
                    match = re.search(r"\b\d+\.\d+\b|\b\d+\b", response_text)
                    if match:
                        return match.group(0)
                elif expected_type == "yes/no":
                    match = re.search(r"yes|no", response_text, re.IGNORECASE)
                    if match:
                        return match.group(0).lower()
                return response_text
            builtins.input = simulated_input
            try:
                # Track sub-agent call in tree
                sub_agent_node_id = get_node_id('subagent')
                self.log_agent_call(call_tree, sub_agent_node_id, 'DynamicTimeSeriesAgent', 'analyze_query', sub_prompt_node_id, parameters={'sub_prompt': sub})
                # Log sub-agent node with correct parent_id
                self.logger.info(
                    f"DynamicTimeSeriesAgent analyzing sub-prompt: {sub}",
                    extra={
                        'session_id': session_id,
                        'agent_name': 'DynamicTimeSeriesAgent',
                        'stage': 'analyze_query',
                        'step_id': sub_agent_node_id,
                        'parent_step_id': sub_prompt_node_id,
                        'parent_id': sub_prompt_node_id
                    }
                )
                print(f"[DEBUG] Routing sub-prompt based on type: {prompt_type}")
                if prompt_type == 'time_series':
                    result = await self.time_series_agent.analyze_query(
                        sub,
                        session_id=str(session_id),
                        agent_name='DynamicTimeSeriesAgent',
                        call_tree=call_tree,
                        get_node_id=get_node_id
                    )
                elif prompt_type == 'pinpoint_value':
                    # Loop to fill missing parameters using interaction agent
                    filled_params = {}
                    def parse_param_value(param_name, param_info, response):
                        # Try to parse the response based on expected type
                        expected_type = param_info.get("type", "string")
                        if expected_type in ("int", "float"):
                            try:
                                val = float(response)
                                if expected_type == "int":
                                    return int(val)
                                return val
                            except Exception:
                                return None
                        elif expected_type == "yes/no":
                            if str(response).strip().lower() in ("yes", "no"):
                                return str(response).strip().lower()
                            return None
                        else:
                            return response.strip() if isinstance(response, str) else response

                    while True:
                        # Build an updated prompt with filled parameter values
                        if filled_params:
                            given_str = "; ".join(f"{k}={v}" for k, v in filled_params.items() if v is not None)
                            updated_prompt = f"{sub}\nGiven: {given_str}"
                        else:
                            updated_prompt = sub
                        result = await self.pinpoint_value_agent.analyze_query(
                            updated_prompt,
                            session_id=str(session_id),
                            agent_name='PinpointValueAgent',
                            call_tree=call_tree,
                            get_node_id=get_node_id,
                            extracted_params=filled_params if filled_params else None
                        )
                        # If an error occurred during parameter collection, abort immediately
                        if isinstance(result, dict) and result.get("error"):
                            self.logger.error(f"Aborting due to missing parameter: {result['error']}")
                            return result
                        if isinstance(result, dict) and result.get("missing_parameters"):
                            self.logger.warning(f"Missing parameters detected: {result['missing_parameters']}")
                            
                            # First, try to resolve parameters using PostgreSQL data with self-healing capability
                            if hasattr(self, 'postgres_provider') and hasattr(self, '_fetch_postgres_data'):
                                self.logger.info("[Postgres Integration] Attempting to resolve parameters using PostgreSQL data with self-healing")
                                
                                # Generate a specific parameter query
                                param_query = f"What are the values for {', '.join(result['missing_parameters'])} in the context of {sub}?"
                                
                                # Fetch PostgreSQL data with self-healing enabled
                                postgres_result = await self._fetch_postgres_data(
                                    param_query, 
                                    session_id=str(session_id)
                                )
                                
                                # Process the parameters if data was successfully retrieved
                                if postgres_result and postgres_result.get("status") == "success":
                                    self.logger.info(f"[Postgres Integration] Successfully retrieved parameter data")
                                    
                                    # Extract parameters from PostgreSQL data
                                    postgres_data = postgres_result.get("data", {})
                                    postgres_params = {}
                                    
                                    # Check if we have PostgreSQL data to work with
                                    if postgres_data and isinstance(postgres_data, (dict, list)):
                                        # Convert to a list if it's a dictionary
                                        if isinstance(postgres_data, dict):
                                            postgres_data_list = [postgres_data]
                                        else:
                                            postgres_data_list = postgres_data
                                            
                                        # Log the available data for debugging
                                        self.logger.info(f"[Postgres Integration] Available data fields: {list(postgres_data_list[0].keys()) if postgres_data_list else 'None'}")
                                        
                                        # Use LLM to smartly extract parameters from the data
                                        params_extraction_prompt = f"""
                                        Extract values for these parameters from the PostgreSQL data:
                                        Required parameters: {result['missing_parameters']}
                                        
                                        Available data:
                                        {json.dumps(postgres_data_list[:5], indent=2)}
                                        
                                        Original query context:
                                        {sub}
                                        
                                        Return a JSON object mapping each parameter to the appropriate value from the data.
                                        If a parameter is not available in the data, leave it as null.
                                        For example: {{"param_name": extracted_value, "param2": null}}
                                        
                                        Extraction rules:
                                        - For energy_output_t: Look for values like "total_generation", "annual_energy_output", etc.
                                        - For n: Look for values like "operational_reactors", "number_of_units", etc.
                                        - Prefer actual numeric values over null/placeholder values
                                        - Include units in the parameter value when appropriate (e.g., "1000 GWh")
                                        - Be smart about interpreting the data in the context of the query
                                        """
                                        
                                        extraction_messages = [
                                            {"role": "system", "content": "You are an expert data analyst who extracts parameter values from database results. Return only valid JSON."},
                                            {"role": "user", "content": params_extraction_prompt}
                                        ]
                                        
                                        try:
                                            from utils.llm_provider import get_llm_response
                                            extraction_response = get_llm_response(extraction_messages, temperature=0.1)
                                            
                                            # Clean the response - remove markdown code blocks if present
                                            cleaned_response = extraction_response.strip()
                                            if cleaned_response.startswith('```json'):
                                                cleaned_response = cleaned_response[7:]  # Remove ```json
                                            if cleaned_response.startswith('```'):
                                                cleaned_response = cleaned_response[3:]   # Remove ```
                                            if cleaned_response.endswith('```'):
                                                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
                                            
                                            # Parse the extracted parameters
                                            postgres_params = json.loads(cleaned_response)
                                            
                                            # Log the extracted parameters
                                            self.logger.info(f"[Postgres Integration] Extracted parameters: {postgres_params}")
                                            
                                            # Filter out null values
                                            postgres_params = {k: v for k, v in postgres_params.items() if v is not None}
                                            
                                            # Update the filled parameters with extracted values
                                            for param, value in postgres_params.items():
                                                if param in result["missing_parameters"]:
                                                    self.logger.info(f"[Postgres Integration] Using value '{value}' for parameter '{param}' from PostgreSQL data")
                                                    filled_params[param] = value
                                                    
                                                    # Check for direct parameters from the Postgres response
                                                    if postgres_result.get("parameters") and param in postgres_result["parameters"]:
                                                        direct_value = postgres_result["parameters"][param]
                                                        if direct_value is not None:
                                                            self.logger.info(f"[Postgres Integration] Overriding with direct value '{direct_value}' for parameter '{param}'")
                                                            filled_params[param] = direct_value
                                                            
                                                    # Also check in the data.parameters path (used by the postgres_data_provider)
                                                    if postgres_result.get("data") and isinstance(postgres_result["data"], dict) and "parameters" in postgres_result["data"]:
                                                        data_params = postgres_result["data"]["parameters"]
                                                        if param in data_params and data_params[param] is not None:
                                                            self.logger.info(f"[Postgres Integration] Using data.parameters value '{data_params[param]}' for parameter '{param}'")
                                                            filled_params[param] = data_params[param]
                                                    
                                                    result["missing_parameters"].remove(param)
                                        except Exception as e:
                                            self.logger.error(f"[Postgres Integration] Failed to extract parameters: {str(e)}")
                                
                            # Second, try to resolve parameters using CSV data if available
                            if hasattr(self, '_resolve_parameters_from_csv'):
                                self.logger.info("[CSV Integration] Attempting to resolve parameters using CSV data")
                                csv_resolved_params = await self._resolve_parameters_from_csv(
                                    sub, result["missing_parameters"], result.get("parameters", {})
                                )
                                for param_name, param_value in csv_resolved_params.items():
                                    if param_name in result["missing_parameters"]:
                                        filled_params[param_name] = param_value
                                        self.logger.info(f"[CSV Integration] Resolved '{param_name}' = '{param_value}' from CSV data")
                                        # Remove from missing parameters list
                                        result["missing_parameters"] = [p for p in result["missing_parameters"] if p != param_name]
                            
                            # Use interaction agent to collect remaining missing values
                            required_params_info = result.get("parameters", {})
                            for param in result["missing_parameters"]:
                                param_info = {}
                                # Try to get param_info from the latest result if available
                                if param in required_params_info:
                                    param_info = {"description": "", "type": "string"}
                                    # Try to infer type from description if not present
                                    if isinstance(required_params_info, dict):
                                        param_info = required_params_info.get(param, param_info)
                                prompt_text = f"Please provide a value for '{param}'"
                                if "description" in param_info and param_info["description"]:
                                    prompt_text += f" ({param_info['description']})"
                                prompt_text += ":"
                                max_retries = 3
                                for attempt in range(max_retries):
                                    # Use LLM direct value response if available
                                    if hasattr(self.interaction_agent, "generate_parameter_value_response"):
                                        user_value = self.interaction_agent.generate_parameter_value_response(param, param_info).response
                                    else:
                                        user_value = self.interaction_agent.generate_response(prompt_text).response
                                    parsed_value = parse_param_value(param, param_info, user_value)
                                    if parsed_value is not None:
                                        filled_params[param] = parsed_value
                                        break
                                    else:
                                        prompt_text = f"Invalid value for '{param}'. Please enter a valid {param_info.get('type', 'value')}:"
                                else:
                                    # Fallback: prompt user directly via input()
                                    self.logger.warning(f"LLM failed to provide a valid value for '{param}'. Falling back to manual user input.")
                                    for attempt in range(2):
                                        try:
                                            user_value = input(f"[USER INPUT REQUIRED] Please enter a value for '{param}': ")
                                        except Exception as e:
                                            self.logger.error(f"Input error for '{param}': {e}")
                                            user_value = ""
                                        parsed_value = parse_param_value(param, param_info, user_value)
                                        if parsed_value is not None:
                                            filled_params[param] = parsed_value
                                            break
                                        else:
                                            print(f"Invalid value for '{param}'. Please enter a valid {param_info.get('type', 'value')}.")
                                    else:
                                        # Abort if still invalid after retries
                                        error_msg = (
                                            f"Failed to obtain valid values for required parameters after LLM and user input attempts.\n"
                                            f"Please provide the following values in your next request:\n"
                                            f"  - {param} (type: {param_info.get('type', 'value')}, description: {param_info.get('description', '')})"
                                        )
                                        self.logger.error(error_msg)
                                        print(error_msg)
                                        return {
                                            "error": error_msg,
                                            "missing_parameter": param,
                                            "sub_prompt": sub
                                        }
                            # Continue loop to re-run with filled_params
                        else:
                            break
                    # If an error occurred during parameter collection, abort immediately
                    if isinstance(result, dict) and result.get("error"):
                        self.logger.error(f"Aborting due to missing parameter: {result['error']}")
                        return result
                    # Check for missing parameters in pinpoint_value_agent result
                    if isinstance(result, dict) and result.get("missing_parameters"):
                        self.logger.warning(f"Missing parameters detected: {result['missing_parameters']}")
                        # Return early with missing parameters info
                        return {
                            "missing_parameters": result["missing_parameters"],
                            "message": result.get("message", "Missing required parameters."),
                            "sub_prompt": sub
                        }
                else:
                    # Fallback to time_series_agent for other types for now
                    print(f"[DEBUG] Fallback to DynamicTimeSeriesAgent for prompt type: {prompt_type}")
                    result = await self.time_series_agent.analyze_query(
                        sub,
                        session_id=session_id,
                        agent_name='DynamicTimeSeriesAgent',
                        call_tree=call_tree,
                        get_node_id=get_node_id
                    )
                print(f"[DEBUG] Result from DynamicTimeSeriesAgent: {result}")
                self.log_agent_call(call_tree, sub_agent_node_id, 'DynamicTimeSeriesAgent', 'analyze_query', sub_prompt_node_id, parameters={'sub_prompt': sub}, result=str(result))
                results.append({"sub_prompt": sub, "result": result})
                summary = result.get('summary') if isinstance(result, dict) else getattr(result, 'summary', None)
                if summary:
                    table.append({"sub_prompt": sub, **summary})
                # Log decision point and add to call_tree
                decision_node_id = get_node_id('decision')
                self.log_decision_point(
                    session_id=session_id,
                    agent_name=agent_name,
                    step_id=decision_node_id,
                    parent_step_id=sub_agent_node_id,
                    decision_type='accept',
                    reason='Sub-agent result accepted as valid for sub-prompt',
                    result_summary=str(summary) if summary else str(result),
                    sub_prompt=sub,
                    stage='decision_point',
                    confidence=getattr(result, 'confidence', None),
                    parameters=None,
                    llm_rationale=None,
                    depth=2,
                    parent_id=sub_agent_node_id
                )
                self.logger.info(
                    f"Decision: accept | Reason: Sub-agent result accepted as valid for sub-prompt | Result: {str(summary) if summary else str(result)} | Confidence: {getattr(result, 'confidence', None)} | Parameters: None | LLM Rationale: None",
                    extra={
                        'session_id': session_id,
                        'agent_name': agent_name,
                        'stage': 'decision_point',
                        'step_id': decision_node_id,
                        'parent_step_id': sub_agent_node_id,
                        'parent_id': sub_agent_node_id,
                        'decision_type': 'accept'
                    }
                )
                self.log_decision_node_to_tree(
                    call_tree,
                    decision_node_id,
                    agent_name,
                    'accept',
                    sub_prompt_node_id,
                    'Sub-agent result accepted as valid for sub-prompt',
                    str(summary) if summary else str(result),
                    sub_prompt=sub,
                    stage='decision_point',
                    confidence=getattr(result, 'confidence', None),
                    parameters=None,
                    llm_rationale=None
                )
                mongo_log_handler.insert_one({
                    "session_id": session_id,
                    "agent_name": "OrchestratorAgent",
                    "checkpoint": "sub_prompt_result",
                    "sub_prompt": sub,
                    "result": serialize_result(result),
                    "timestamp": datetime.utcnow().isoformat(),
                    "stage": "sub_prompt_result",
                    "step_id": sub_prompt_node_id,
                    "parent_step_id": parent_id
                })
            finally:
                builtins.input = original_input
        print(f"[DEBUG] Finished sub-prompt loop. Results: {results}")
        result = {
            "original_prompt": prompt,
            "sub_prompts": sub_prompts,
            "results": results,
            "table": table,
            "summary": self._summarize_table(table)
        }
        # Log final result to MongoDB
        final_node_id = get_node_id('finalize')
        mongo_log_handler.insert_one({
            "session_id": session_id,
            "agent_name": "OrchestratorAgent",
            "checkpoint": "final_result",
            "result": serialize_result(result),
            "timestamp": datetime.utcnow().isoformat(),
            "stage": "final_result",
            "step_id": final_node_id,
            "parent_step_id": None
        })
        self.log_decision_point(
            session_id=session_id,
            agent_name=agent_name,
            step_id=final_node_id,
            parent_step_id=None,
            decision_type='finalize',
            reason='All sub-prompts processed and results aggregated',
            result_summary=str(result['summary']),
            sub_prompt=None,
            stage='decision_point',
            depth=1,
            parent_id=None
        )
        self.logger.info(
            f"Decision: finalize | Reason: All sub-prompts processed and results aggregated | Result: {str(result['summary'])} | Confidence: None | Parameters: None | LLM Rationale: None",
            extra={
                'session_id': session_id,
                'agent_name': agent_name,
                'stage': 'decision_point',
                'step_id': final_node_id,
                'parent_step_id': last_decision_node_id if 'last_decision_node_id' in locals() else '',
                'parent_id': last_decision_node_id if 'last_decision_node_id' in locals() else '',
                'decision_type': 'finalize'
            }
        )
        self.log_decision_node_to_tree(
            call_tree,
            final_node_id,
            agent_name,
            'finalize',
            None,
            'All sub-prompts processed and results aggregated',
            str(result['summary']),
            sub_prompt=None,
            stage='decision_point'
        )
        print(f"[DEBUG] About to log final result to MongoDB and log call_tree.")
        print(f"[DEBUG] Logging call_tree: {json.dumps(call_tree)}")
        # Remove file handler and flush logs to Qdrant
        print(f"[DEBUG] Removing file handler and closing log file.")
        logger.removeHandler(file_handler)
        file_handler.close()
        print(f"[DEBUG] Finished logging. Log file exists: {os.path.exists(log_file_path)}")
        print("[DEBUG] About to call upload_logs_from_file.")
        log_handler.upload_logs_from_file(log_file_path)
        print("[DEBUG] Called upload_logs_from_file.")
        # Delete the log file after upload if configured
        if DELETE_LOG_FILE_AFTER_UPLOAD:
            try:
                os.remove(log_file_path)
                print(f"[DEBUG] Deleted log file: {log_file_path}")
            except Exception as e:
                print(f"[DEBUG] Failed to delete log file: {log_file_path}. Error: {e}")
        return result

    def _summarize_table(self, table: List[Dict[str, Any]]) -> str:
        if not table:
            return "No results to summarize."
        # Simple summary: list all sub-prompts and their main result
        lines = ["Summary of Results:"]
        for row in table:
            main_val = next((v for k, v in row.items() if k not in ("sub_prompt",)), None)
            lines.append(f"- {row['sub_prompt']}: {main_val}")
        return "\n".join(lines)

if __name__ == "__main__":
    import asyncio
    orchestrator = OrchestratorAgent()
    prompt = "what is the load factor of a generator for a year"
    print("[DEBUG] Running orchestrator main entry point.")
    asyncio.run(orchestrator.analyze(prompt))
