"""
Main application entry point for the Agentic System.
"""
import os
import sys
import logging
import json
import asyncio
from datetime import datetime
from config.settings import OPENAI_API_KEY, OPENAI_MODEL
import textwrap
import re

# Initialize LM Studio as the default LLM provider
from utils.llm_provider import initialize_llm_provider
initialize_llm_provider()

# Import agents
from agents.llm_formula_resolver import LLMFormulaResolver
from agents.calc_executor import CalcExecutor
from agents.autogen_parser_agent import AutoGenParserAgent
from agents.messages import TextMessage
from agents.llm_parsing_manager import ParsingManagerAgent
from agents.llm_parsing_worker import ParsingWorkerAgent

# Setup logging
LOG_FILE = "logs/agentic_system.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def sanitize_and_fix_code(code: str, parameters: dict) -> str:
    """
    Improved code sanitization that properly handles parameter assignments
    """
    lines = code.splitlines()
    sanitized_lines = []
    defined_variables = set()
    used_variables = set()
    
    # First pass: identify what variables are used
    for line in lines:
        # Find variable usage (simple heuristic)
        import re
        vars_in_line = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', line)
        for var in vars_in_line:
            if var not in ['def', 'if', 'else', 'for', 'while', 'try', 'except', 'import', 'from', 'return', 'pass', 'raise', 'print', 'input', 'float', 'int', 'str', 'list', 'dict', 'True', 'False', 'None']:
                used_variables.add(var)
    
    # Second pass: process lines and fix issues
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
            
        stripped = line.strip()
        
        # Skip problematic patterns
        if (stripped.startswith('input(') or 
            'input(' in stripped or 
            stripped.startswith('raise ') or
            stripped.startswith('print(') or
            stripped.startswith('try:') or
            stripped.startswith('except')):
            continue
            
        # Handle variable definitions
        if '=' in line and not line.strip().startswith('def '):
            var_name = line.split('=')[0].strip()
            defined_variables.add(var_name)
            
        sanitized_lines.append(line)
    
    # Third pass: add missing variable definitions
    missing_vars = used_variables - defined_variables
    parameter_assignments = []
    
    for var in missing_vars:
        # Try to find a matching parameter
        param_value = None
        for param_key, param_info in parameters.items():
            # Convert parameter names to valid variable names
            clean_param_name = param_key.lower().replace(' ', '_').replace('-', '_')
            if clean_param_name == var.lower() or param_key.lower().replace(' ', '_') == var.lower():
                param_value = param_info.get('value')
                break
        
        if param_value is not None:
            parameter_assignments.append(f"{var} = {param_value}")
        else:
            # Use smart defaults from constants module
            from config.constants import get_smart_default
            default_value = get_smart_default(var)
            parameter_assignments.append(f"{var} = {default_value}  # dynamic default")
    
    # Combine parameter assignments with the sanitized code
    if parameter_assignments:
        sanitized_lines = parameter_assignments + [''] + sanitized_lines
    
    return '\n'.join(sanitized_lines)

def log_agent_step(agent_name, step, input_data, output_data):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "step": step,
        "input": input_data,
        "output": output_data
    }
    logging.info(json.dumps(log_entry))

async def process_query(query):
    # 1. Query Parsing
    parser_agent = AutoGenParserAgent()
    parser_result = await parser_agent.on_message(TextMessage(content=query, source="user"))
    log_agent_step("AutoGenParserAgent", "parse", {"query": query}, parser_result)
    if not parser_result.get("success"):
        return {"success": False, "error": "Query parsing failed", "details": parser_result}
    parameters = parser_result["parameters"]

    # 2. Code Generation
    llm_resolver = LLMFormulaResolver()
    llm_input = {"operation": "resolve_and_generate_code", "query": query}
    llm_result = await llm_resolver.process(llm_input)
    log_agent_step("LLMFormulaResolver", "resolve_and_generate_code", llm_input, llm_result)
    if not llm_result.get("success"):
        return {"success": False, "error": "Code generation failed", "details": llm_result}
    resolved_data = llm_result.get("data") or llm_result
    executable_code = resolved_data.get("executable_code")
    if not executable_code:
        return {"success": False, "error": "No executable code generated", "details": llm_result}

    # Dedent and strip code to avoid indentation errors
    executable_code = textwrap.dedent(executable_code).strip()
    
    # Improved code sanitization
    executable_code = sanitize_and_fix_code(executable_code, resolved_data.get("parameters", {}))
    log_agent_step("Main", "sanitize_code", {"reason": "Applied improved sanitization"}, {"code": executable_code})
    # Remove any remaining 'if' lines not followed by an indented block
    lines = executable_code.splitlines()
    cleaned_lines = []
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        if line.strip().startswith('if '):
            # If next line is not indented, skip this line
            if i + 1 >= len(lines) or (lines[i + 1] and not lines[i + 1].startswith((' ', '\t'))):
                continue
        cleaned_lines.append(line)
    executable_code = '\n'.join(cleaned_lines)
    # Remove any sys.exit lines
    executable_code = '\n'.join([line for line in executable_code.splitlines() if 'sys.exit' not in line])
    # Remove all lines that are just 'if' statements with no indented block, and remove empty lines
    executable_code = '\n'.join([line for line in executable_code.splitlines() if not (line.strip().startswith('if ') and line.strip().endswith(':')) and line.strip()])
    # Fully dedent all lines to remove any remaining indentation
    executable_code = '\n'.join([line.lstrip() for line in executable_code.splitlines()])
    # Remove any while lines
    executable_code = '\n'.join([line for line in executable_code.splitlines() if not line.strip().startswith('while ')])
    # Fix concatenated assignments (e.g., 'CAPEX = 0OPEX_t = 0')
    executable_code = re.sub(r'(= 0)([A-Za-z_][A-Za-z0-9_]*\s*=)', r'\1\n\2', executable_code)
    # Remove any 'return' lines
    executable_code = '\n'.join([line for line in executable_code.splitlines() if not line.strip().startswith('return')])
    # Replace any assignment of None to a variable with 0 (generic, not hardcoded)
    executable_code = re.sub(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*None', r'\1 = 0', executable_code)

    # Insert 'pass' into any function definition with no indented block
    def insert_pass_in_empty_functions(code):
        lines = code.splitlines()
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith('def ') and (i+1 == len(lines) or not (lines[i+1].startswith(' ') or lines[i+1].startswith('\t'))):
                new_lines.append(line)
                new_lines.append('    pass')
                i += 1
            else:
                new_lines.append(line)
                i += 1
        return '\n'.join(new_lines)
    executable_code = insert_pass_in_empty_functions(executable_code)

    # 3. Code Execution using CalcExecutor
    calc_executor = CalcExecutor()
    
    # Prepare inputs with extracted parameters and dynamic constants
    execution_inputs = {}
    
    # Add extracted parameters from parser
    if parameters:
        for param_name, param_info in parameters.items():
            if isinstance(param_info, dict) and 'value' in param_info:
                execution_inputs[param_name] = param_info['value']
            else:
                execution_inputs[param_name] = param_info
    
    # Add common dynamic constants that might be needed
    try:
        from config.constants import TimeConstants, EnergyDefaults, FinancialDefaults
        execution_inputs.update({
            'HOURS_PER_YEAR': TimeConstants.HOURS_PER_YEAR,
            'DAYS_PER_YEAR': TimeConstants.DAYS_PER_YEAR,
            'MONTHS_PER_YEAR': TimeConstants.MONTHS_PER_YEAR,
            'DEFAULT_CAPACITY_MW': EnergyDefaults.DEFAULT_CAPACITY_MW,
            'DEFAULT_EFFICIENCY': EnergyDefaults.DEFAULT_EFFICIENCY,
            'DEFAULT_CAPACITY_FACTOR': EnergyDefaults.DEFAULT_CAPACITY_FACTOR,
            'DEFAULT_DISCOUNT_RATE': FinancialDefaults.DEFAULT_DISCOUNT_RATE
        })
    except ImportError:
        pass
    
    calc_input = {"operation": "run_python_code", "code": executable_code, "inputs": execution_inputs}
    calc_result = await calc_executor.process(calc_input)
    log_agent_step("CalcExecutor", "run_python_code", calc_input, calc_result)
    if not calc_result.get("success"):
        return {"success": False, "error": "Code execution failed", "details": calc_result}

    return {
        "success": True,
        "parser_result": parser_result,
        "llm_result": llm_result,
        "calc_result": calc_result
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agentic System: Query to Answer Pipeline")
    parser.add_argument('--query', type=str, help='Natural language query to process')
    parser.add_argument('--llm-parse', action='store_true', help='Use LLM ParsingManagerAgent for parameter extraction only')
    args = parser.parse_args()

    if not args.query:
        print("You must provide a query string via --query.")
        sys.exit(1)

    if args.llm_parse:
        # Use LLM ParsingManagerAgent for parameter extraction
        # Step 1: LLM-driven parameter name extraction
        param_names = asyncio.run(ParsingWorkerAgent.extract_param_names(args.query))
        # Filter out 'result', empty, and duplicates
        param_names = [p for p in param_names if p and p.lower() != 'result']
        param_names = list(dict.fromkeys(param_names))  # remove duplicates, preserve order
        print("[DEBUG] Discovered parameter names:", param_names)
        manager = ParsingManagerAgent()
        results = asyncio.run(manager.extract_all(args.query, param_names))
        print("LLM ParsingManagerAgent results:")
        print(json.dumps(results, indent=2, default=str))
        sys.exit(0)

    result = asyncio.run(process_query(args.query))
    print(json.dumps(result, indent=2, default=str))




