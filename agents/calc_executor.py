"""
Calculation Executor Agent - Runs numeric computations using OpenAI Code Interpreter.
"""
import json
import ast
import math
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import traceback
import sys
from io import StringIO
import re
import scipy
import scipy.optimize
import statistics
import sympy
import random
import logging

from core.simple_base_agent import SimpleBaseAgent


class CalcExecutor(SimpleBaseAgent):
    """
    Agent responsible for executing numeric computations using Python sandbox
    and OpenAI Code Interpreter capabilities.
    """
    
    def __init__(self):
        # Define tools for calculation operations
        tools = [
            self.execute_formula,
            self.validate_inputs,
            self.run_python_code,
            self.calculate_financial_metrics
        ]
        
        super().__init__(
            name="CalcExecutor",
            description="Executes numeric computations for energy calculations including LCOE, NPV, IRR, and other financial/technical metrics. Uses Python sandbox and mathematical libraries.",
            tools=tools
        )
        
        # Safe execution environment
        self.safe_globals = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
                'chr': chr, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
                'filter': filter, 'float': float, 'format': format, 'hex': hex,
                'int': int, 'len': len, 'list': list, 'map': map, 'max': max,
                'min': min, 'oct': oct, 'ord': ord, 'pow': pow, 'range': range,
                'reversed': reversed, 'round': round, 'set': set, 'slice': slice,
                'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
                'type': type, 'zip': zip
            },
            'math': math,
            'np': np,
            'pd': pd,
            'datetime': datetime,
            'scipy': scipy,
            'optimize': scipy.optimize,
            'statistics': statistics,
            'sympy': sympy,
            'random': random
        }
    
    def execute_formula(self, formula: str, parameters: Dict[str, Union[float, List[float]]],
                       formula_type: str = "standard", session_id: int = None, agent_name: str = 'CalcExecutor') -> Dict[str, Any]:
        """
        Execute a mathematical formula with given parameters.
        
        Args:
            formula: Mathematical formula as string
            parameters: Dictionary of parameter values
            formula_type: Type of formula (standard, time_series, iterative)
            session_id: Session identifier for logging
            agent_name: Name of the agent for logging
        Returns:
            Calculation results and logs
        """
        # Add session_id and agent_name to all log records
        class ContextFilter(logging.Filter):
            def filter(self, record):
                record.session_id = session_id
                record.agent_name = agent_name
                return True
        self.logger.addFilter(ContextFilter())

        try:
            self.logger.info(f"Executing formula: {formula[:50]}...", extra={'session_id': session_id, 'agent_name': agent_name})
            
            # Prepare execution environment
            execution_env = self.safe_globals.copy()
            execution_env.update({k: [v] if not isinstance(v, (list, np.ndarray)) else v for k, v in parameters.items()})            
            # Handle different formula types
            if formula_type == "time_series":
                result = self._execute_time_series_formula(formula, parameters, execution_env)
            elif formula_type == "iterative":
                result = self._execute_iterative_formula(formula, parameters, execution_env)
            else:
                result = self._execute_standard_formula(formula, parameters, execution_env)
            
            # After formula execution, log accept or error/fallback as appropriate
            try:
                self.log_decision_point(
                    session_id=session_id,
                    agent_name=agent_name,
                    step_id='execute_formula',
                    parent_step_id=None,
                    decision_type='accept',
                    reason='Formula executed successfully',
                    result_summary=str(result),
                    confidence=None,
                    parameters=parameters,
                    llm_rationale=None
                )
                return {
                    "success": True,
                    "result": result,
                    "formula": formula,
                    "parameters_used": parameters,
                    "formula_type": formula_type
                }
            except Exception as e:
                self.log_decision_point(
                    session_id=session_id,
                    agent_name=agent_name,
                    step_id='execute_formula_error',
                    parent_step_id=None,
                    decision_type='retry',
                    reason=f'Formula execution failed: {str(e)}',
                    result_summary=None,
                    confidence=None,
                    parameters=parameters,
                    llm_rationale=None
                )
                return {"error": f"Formula execution failed: {str(e)}"}
            
        except Exception as e:
            return {"error": f"Formula execution failed: {str(e)}"}
    
    def validate_inputs(self, parameters: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input parameters against requirements.
        
        Args:
            parameters: Input parameters
            requirements: Parameter requirements
        
        Returns:
            Validation results
        """
        try:
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "processed_parameters": {}
            }
            
            for param_name, param_req in requirements.items():
                if param_name not in parameters:
                    if param_req.get("required", True):
                        validation_results["valid"] = False
                        validation_results["errors"].append("Missing required parameter: {}".format(param_name))
                    elif "default" in param_req:
                        validation_results["processed_parameters"][param_name] = param_req["default"]
                        validation_results["warnings"].append("Using default value for {}: {}".format(param_name, param_req["default"]))
                    continue
                
                value = parameters[param_name]
                
                # Type validation
                expected_type = param_req.get("type", "float")
                if expected_type == "float" and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except ValueError:
                        validation_results["valid"] = False
                        validation_results["errors"].append("Parameter {} must be numeric".format(param_name))
                        continue
                
                # Range validation
                if "min_value" in param_req and value < param_req["min_value"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("Parameter {} below minimum: {}".format(param_name, param_req["min_value"]))
                
                if "max_value" in param_req and value > param_req["max_value"]:
                    validation_results["valid"] = False
                    validation_results["errors"].append("Parameter {} above maximum: {}".format(param_name, param_req["max_value"]))
                
                validation_results["processed_parameters"][param_name] = value
            
            return {
                "success": True,
                "validation": validation_results
            }
            
        except Exception as e:
            return {"error": f"Input validation failed: {str(e)}"}
    
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
    
    def _strip_unnecessary_imports(self, code: str) -> str:
        """
        Remove all import statements from the code string.
        Only pre-imported modules are available in the execution environment.
        """
        # Remove any line that starts with 'import' or 'from ... import ...'
        return re.sub(r"^\s*(import\s+\w+|from\s+\w+(?:\.\w+)*\s+import\s+.*)$", "", code, flags=re.MULTILINE)
    
    def run_python_code(self, code: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute Python code in a safe environment.
        
        Args:
            code: Python code to execute
            inputs: Input variables for the code
        
        Returns:
            Execution results
        """
        try:
            # Strip all import statements
            code = self._strip_unnecessary_imports(code)
            # Prepare execution environment
            execution_env = self.safe_globals.copy()
            if inputs:
                execution_env.update(inputs)
            
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Execute code
            try:
                # Compile and execute
                compiled_code = compile(code, 
                                        '<string>', 
                                        'exec')
                try:
                    exec(compiled_code, execution_env)
                except NameError as ne:
                    # Check if the error is due to a missing module
                    missing = str(ne).split("'")
                    if len(missing) > 1:
                        missing_mod = missing[1]
                        available_mods = ', '.join([k for k in self.safe_globals.keys() if k != '__builtins__'])
                        return {
                            "error": f"Module '{missing_mod}' is not available. Only these modules are allowed: {available_mods}.",
                            "code_executed": code
                        }
                    else:
                        raise
                # Get output
                output = captured_output.getvalue()
                # Extract result variables (variables that don't start with _)
                results = {k: v for k, v in execution_env.items() 
                          if not k.startswith('_') and k not in self.safe_globals}
                return {
                    "success": True,
                    "results": results,
                    "output": output,
                    "code_executed": code
                }
            finally:
                sys.stdout = old_stdout
        except Exception as e:
            return {"error": f"Code execution failed: {str(e)}", "traceback": traceback.format_exc()}
    
    def calculate_financial_metrics(self, cash_flows: List[float], initial_investment: float = None,
                                   discount_rate: float = 0.1) -> Dict[str, Any]:
        """
        Calculate common financial metrics (NPV, IRR, Payback Period).
        
        Args:
            cash_flows: List of cash flows over time
            initial_investment: Initial investment (if not included in cash_flows[0])
            discount_rate: Discount rate for NPV calculation
        
        Returns:
            Financial metrics results
        """
        try:
            # Prepare cash flow array
            if initial_investment is not None:
                cash_flows = [-initial_investment] + list(cash_flows)
            
            cash_flows = np.array(cash_flows)
            
            # Calculate NPV
            periods = np.arange(len(cash_flows))
            discount_factors = (1 + discount_rate) ** periods
            npv = np.sum(cash_flows / discount_factors)
            
            # Calculate IRR using Newton-Raphson method
            irr = self._calculate_irr(cash_flows)
            
            # Calculate Payback Period
            cumulative_cash_flow = np.cumsum(cash_flows)
            payback_period = None
            for i, cum_cf in enumerate(cumulative_cash_flow):
                if cum_cf >= 0:
                    if i == 0:
                        payback_period = 0
                    else:
                        # Linear interpolation for fractional year
                        prev_cum_cf = cumulative_cash_flow[i-1]
                        payback_period = i - 1 + abs(prev_cum_cf) / cash_flows[i]
                    break
            
            # Calculate Profitability Index
            pv_future_cash_flows = np.sum(cash_flows[1:] / discount_factors[1:])
            profitability_index = pv_future_cash_flows / abs(cash_flows[0]) if cash_flows[0] < 0 else None
            
            return {
                "success": True,
                "metrics": {
                    "npv": float(npv),
                    "irr": float(irr) if irr is not None else None,
                    "payback_period": float(payback_period) if payback_period is not None else None,
                    "profitability_index": float(profitability_index) if profitability_index is not None else None
                },
                "cash_flows": cash_flows.tolist(),
                "discount_rate": discount_rate
            }
            
        except Exception as e:
            return {"error": f"Financial metrics calculation failed: {str(e)}"}
    
    def _execute_standard_formula(self, formula: str, parameters: Dict, execution_env: Dict) -> float:
        """
        Execute a standard mathematical formula.
        """
        # Replace '^' with '**' for power operations
        formula_parsed = formula.replace('^', '**')
        
        # Evaluate the formula using parameters directly from execution_env
        # Ensure all parameters are available in the execution environment
        for k, v in parameters.items():
            if k not in execution_env:
                execution_env[k] = v
        result = eval(formula_parsed, execution_env)
        return float(result)
    
    def _execute_time_series_formula(self, formula: str, parameters: Dict, execution_env: Dict) -> List[float]:
        """
        Execute a formula that operates on time series data.
        """
        results = []
        
        # Determine the length of time series
        time_series_length = 1
        for param_value in parameters.values():
            if isinstance(param_value, list):
                time_series_length = max(time_series_length, len(param_value))
        
        # Execute formula for each time period
        for t in range(time_series_length):
            period_params = {}
            for param_name, param_value in parameters.items():
                if isinstance(param_value, list):
                    period_params[param_name] = param_value[t] if t < len(param_value) else param_value[-1]
                else:
                    period_params[param_name] = param_value
            
            # Replace '^' with '**' for power operations
            formula_parsed = formula.replace('^', '**')
            
            # Update execution environment with period-specific parameters
            period_execution_env = execution_env.copy()
            period_execution_env.update(period_params)
            period_execution_env["i"] = t # Add current time index to environment

            result = eval(formula_parsed, period_execution_env)
            results.append(float(result))


        
        return results
    
    def _execute_iterative_formula(self, formula: str, parameters: Dict, execution_env: Dict) -> float:
        """
        Execute a formula that requires iterative calculation (like IRR).
        """
        # This is a simplified implementation
        # In practice, you'd implement specific iterative algorithms
        
        if "IRR" in formula or "irr" in formula:
            cash_flows = parameters.get("cash_flows", [])
            if cash_flows:
                return self._calculate_irr(np.array(cash_flows))
        
        # Fallback to standard execution
        return self._execute_standard_formula(formula, parameters, execution_env)
    
    def _calculate_irr(self, cash_flows: np.ndarray, max_iterations: int = 100, tolerance: float = 1e-6) -> Optional[float]:
        """
        Calculate Internal Rate of Return using Newton-Raphson method.
        """
        try:
            # Initial guess
            irr = 0.1
            
            for _ in range(max_iterations):
                # Calculate NPV and its derivative
                periods = np.arange(len(cash_flows))
                
                # NPV function
                npv = np.sum(cash_flows / (1 + irr) ** periods)
                
                # Derivative of NPV
                dnpv = np.sum(-periods * cash_flows / (1 + irr) ** (periods + 1))
                
                # Newton-Raphson update
                if abs(dnpv) < tolerance:
                    break
                
                irr_new = irr - npv / dnpv
                
                if abs(irr_new - irr) < tolerance:
                    return irr_new
                
                irr = irr_new
            
            return irr
            
        except:
            return None
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process calculation execution request.
        
        Args:
            input_data: Dictionary containing formula, parameters, and execution type
            
        Returns:
            Dictionary with calculation results and logs
        """
        try:
            self.log_activity("Starting calculation execution", input_data)
            
            operation = input_data.get("operation")
            
            if operation == "execute_formula":
                return self.execute_formula(input_data.get("formula"), input_data.get("parameters"), input_data.get("formula_type", "standard"), input_data.get("session_id"), input_data.get("agent_name", "CalcExecutor"))
            elif operation == "validate_inputs":
                return self.validate_inputs(input_data.get("parameters"), input_data.get("requirements"))
            elif operation == "run_python_code":
                return self.run_python_code(input_data.get("code"), input_data.get("inputs"))
            elif operation == "calculate_financial_metrics":
                return self.calculate_financial_metrics(input_data.get("cash_flows"), input_data.get("initial_investment"), input_data.get("discount_rate"))
            else:
                return self.create_error_response(f"Unsupported calculation operation: {operation}")
            
        except Exception as e:
            self.logger.error(f"Error in calculation execution: {str(e)}")
            return self.create_error_response(f"Calculation execution failed: {str(e)}")
    
    def get_supported_functions(self) -> Dict[str, Any]:
        """
        Get list of supported mathematical functions and operations.
        """
        return {
            "mathematical_functions": [
                "abs", "pow", "sqrt", "exp", "log", "log10",
                "sin", "cos", "tan", "asin", "acos", "atan",
                "sinh", "cosh", "tanh", "ceil", "floor", "round"
            ],
            "numpy_functions": [
                "sum", "mean", "std", "var", "min", "max",
                "cumsum", "cumprod", "diff", "gradient"
            ],
            "financial_functions": [
                "npv", "irr", "payback_period", "profitability_index"
            ],
            "supported_formula_types": [
                "standard", "time_series", "iterative"
            ]
        }


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Run CalcExecutor independently.")
    parser.add_argument('--code', type=str, help='Python code to execute')
    parser.add_argument('--inputs', type=str, help='JSON string or path to JSON file with input variables', default=None)
    parser.add_argument('--json', type=str, help='Path to a JSON file with {"code": ..., "inputs": ...}')
    args = parser.parse_args()

    # Load code and inputs
    if args.json:
        with open(args.json, 'r') as f:
            data = json.load(f)
        code = data.get('code')
        inputs = data.get('inputs', None)
    else:
        code = args.code
        inputs = args.inputs
        if inputs:
            try:
                # Try to load as JSON string
                if inputs.endswith('.json'):
                    with open(inputs, 'r') as f:
                        inputs = json.load(f)
                else:
                    inputs = json.loads(inputs)
            except Exception:
                print("Failed to parse inputs. Provide a JSON string or path to a JSON file.")
                inputs = None

    if not code:
        print("You must provide code to execute via --code or --json.")
        exit(1)

    agent = CalcExecutor()
    result = agent.run_python_code(code, inputs)
    print(json.dumps(result, indent=2, default=str))
