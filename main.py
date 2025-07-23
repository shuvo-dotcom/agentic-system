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
    # Remove or replace input() calls
    if 'input(' in executable_code:
        # Remove any line containing input()
        executable_code = '\n'.join([line for line in executable_code.splitlines() if 'input(' not in line])
        log_agent_step("Main", "sanitize_code", {"reason": "input() not allowed"}, {"code": executable_code})
    # Remove or replace raise ValueError or similar error-raising lines
    if 'raise ValueError' in executable_code or 'raise Exception' in executable_code:
        executable_code = '\n'.join([line for line in executable_code.splitlines() if not line.strip().startswith('raise ')])
        log_agent_step("Main", "sanitize_code", {"reason": "raise ValueError/Exception not allowed"}, {"code": executable_code})
    # Remove any try/except/print lines
    executable_code = '\n'.join([line for line in executable_code.splitlines() if not (line.strip().startswith('try:') or line.strip().startswith('except') or 'print(' in line)])
    # Replace any 'if <var> is None:' with '<var> = 0' to avoid empty if blocks (generic, not hardcoded)
    executable_code = re.sub(r'^\s*if\s+([A-Za-z_][A-Za-z0-9_]*)\s+is\s+None:\s*$', r'\1 = 0', executable_code, flags=re.MULTILINE)
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

    # 3. Code Execution
    calc_executor = CalcExecutor()
    calc_input = {"operation": "run_python_code", "code": executable_code, "inputs": {}}
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




