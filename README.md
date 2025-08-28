# Agentic System for Energy Analysis

A comprehensive multi-agent system built with AutoGen and OpenAI for energy sector analysis, calculations, and reporting. The system specializes in Plexos energy modeling data processing, dynamic calculations, and quality-assured outputs.

## Overview

This agentic system orchestrates multiple specialized agents to handle complex energy analysis workflows. It processes queries related to energy calculations (LCOE, capacity factors, NPV, etc.), retrieves data from various sources including Plexos databases, performs calculations with quality control, and exports results in multiple formats.

## System Architecture

The system follows a multi-agent architecture with the following key components:

### Core Components

- **Task Manager**: Central orchestrator that routes queries to appropriate agents
- **Base Agent**: Abstract base class providing common functionality for all agents
- **Configuration**: Centralized settings and environment management

### Specialized Agents

1. **PlexosCSVLoader**: Loads and processes Plexos energy modeling CSV files
2. **PostgresDataProvider**: Connects to external PostgreSQL endpoints for real-time data retrieval with self-healing query capabilities
3. **DataHarvester**: Retrieves data from external sources and APIs
4. **RAGIndexer**: Builds searchable vector and SQL indices for knowledge retrieval
5. **FormulaResolver**: Identifies required metrics and retrieves canonical formulas
6. **CalcExecutor**: Executes numeric computations using Python sandbox
7. **QCAuditor**: Validates results for accuracy, unit consistency, and citations
8. **Exporter**: Formats and exports final answers in various formats

## Features

### Energy Calculations
- Levelized Cost of Energy (LCOE)
- Capacity Factor calculations
- Net Present Value (NPV) and Internal Rate of Return (IRR)
- Payback Period analysis
- Emission Factor calculations
- Load Factor analysis

### Data Integration
- Plexos database integration
- PostgreSQL data provider for precise data retrieval with self-healing queries
- Progressive constraint relaxation for robust data retrieval
- External API data harvesting
- Vector database for semantic search
- SQL database for structured queries
- Hybrid search capabilities

### Quality Assurance
- Numerical validation and range checking
- Unit consistency verification
- Citation link validation
- Comprehensive audit reporting

### Export Formats
- JSON with formatting options
- Markdown documentation
- CSV for spreadsheet applications
- XML structured format
- HTML with styling options
- PDF reports (with additional setup)

## Installation

### Prerequisites

- Python 3.11+
- OpenAI API key
- Required Python packages (see requirements.txt)

### Setup

1. Clone or download the system files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy the `.env.example` file to `.env`
   ```bash
   cp .env.example .env
   ```
   - Edit the `.env` file to add your OpenAI API key and other configuration settings:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export OPENAI_API_BASE="https://api.openai.com/v1"  # Optional
   ```

4. Configure database connections (optional):
   ```bash
   export PLEXOS_DB_PATH="/path/to/plexos/database"
   export DATABASE_URL="sqlite:///./data.db"
   export VECTOR_DB_PATH="./vector_db"
   ```

5. Create logs directory:
   ```bash
   mkdir -p logs
   ```

## Usage

### Basic Usage

```python
import asyncio
from main import AgenticSystem

# Initialize the system
system = AgenticSystem()

# Process a query
result = await system.process_query(
    "Calculate the LCOE for a wind farm with CAPEX of $2000/kW, "
    "OPEX of $50/kW/year, capacity factor of 35%, and 20-year lifetime "
    "with 8% discount rate"
)

print(result)
```

### Direct Agent Calls

```python
# Call a specific agent directly
result = await system.direct_agent_call("CalcExecutor", {
    "formula": "CAPEX / (capacity_factor * 8760 * lifetime)",
    "parameters": {
        "CAPEX": 2000,
        "capacity_factor": 0.35,
        "lifetime": 20
    }
})
```

### System Status

```python
# Get system status
status = system.get_system_status()
print(f"Agents registered: {status['agents_registered']}")
print(f"Tasks processed: {status['tasks_processed']}")

# Get task history
history = system.get_task_history()
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_API_BASE` | OpenAI API base URL | https://api.openai.com/v1 |
| `OPENAI_MODEL` | OpenAI model to use | gpt-4 |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | text-embedding-ada-002 |
| `DATABASE_URL` | SQL database URL | sqlite:///./data.db |
| `VECTOR_DB_PATH` | Vector database path | ./vector_db |
| `PLEXOS_DB_PATH` | Plexos database path | "" |
| `LOG_LEVEL` | Logging level | INFO |
| `MAX_RETRIES` | Max API retries | 3 |
| `TIMEOUT` | Request timeout | 30 |

### Agent Configuration

Each agent can be configured through the settings file or environment variables. Key configuration options include:

- API endpoints and credentials
- Database connection strings
- Calculation parameters and validation ranges
- Export format options
- Quality control thresholds

## API Reference

### AgenticSystem Class

#### Methods

- `process_query(query: str, context: Dict = None) -> Dict`: Process a user query
- `direct_agent_call(agent_name: str, input_data: Dict) -> Dict`: Call specific agent
- `get_system_status() -> Dict`: Get system status information
- `get_task_history() -> List`: Get processing history

### Agent Base Classes

#### SimpleBaseAgent

Base class for all agents providing:
- Standardized input/output handling
- Error handling and logging
- OpenAI API integration
- Activity tracking

#### Key Methods

- `process(input_data: Dict) -> Dict`: Main processing method
- `log_activity(activity: str, data: Dict)`: Log agent activity
- `create_error_response(message: str) -> Dict`: Create error response
- `create_success_response(data: Dict) -> Dict`: Create success response

## Examples

### LCOE Calculation

```python
query = """
Calculate the LCOE for a solar PV system with:
- CAPEX: $1500/kW
- OPEX: $25/kW/year
- Capacity factor: 22%
- Lifetime: 25 years
- Discount rate: 6%
"""

result = await system.process_query(query)
print(f"LCOE: {result['data']['answer']}")
```

### Capacity Factor Analysis

```python
query = """
What is the typical capacity factor for wind energy in Ireland?
Include data sources and validation.
"""

result = await system.process_query(query, {
    "export_format": "markdown",
    "include_sources": True
})
```

### Multi-step Analysis

```python
# Step 1: Load Plexos data
plexos_result = await system.direct_agent_call("PlexosCSVLoader", {
    "file_path": "/path/to/plexos_output.csv",
    "data_type": "generation"
})

# Step 2: Calculate metrics
calc_result = await system.direct_agent_call("CalcExecutor", {
    "formula": "sum(generation) / sum(capacity * hours)",
    "parameters": plexos_result["data"]["processed_data"]
})

# Step 3: Quality control
qc_result = await system.direct_agent_call("QCAuditor", {
    "calculation_data": {
        "inputs": plexos_result["data"],
        "results": calc_result["data"],
        "calculation_type": "capacity_factor"
    }
})
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenAI API Errors**: Check API key and quota
   ```bash
   export OPENAI_API_KEY="your-valid-key"
   ```

3. **Database Connection Issues**: Verify database paths and permissions
   ```bash
   # Check file permissions
   ls -la /path/to/database
   ```

4. **Memory Issues**: For large datasets, consider chunking data processing

### Logging

The system provides comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Check logs
tail -f logs/agentic_system.log
```

### Performance Optimization

- Use direct agent calls for simple operations
- Implement caching for repeated calculations
- Optimize database queries and indices
- Consider parallel processing for independent calculations

## Development

### Adding New Agents

1. Create agent class inheriting from `SimpleBaseAgent`
2. Implement the `process` method
3. Register with task manager
4. Add routing logic if needed

Example:

```python
from core.simple_base_agent import SimpleBaseAgent

class CustomAgent(SimpleBaseAgent):
    def __init__(self):
        super().__init__(
            name="CustomAgent",
            description="Custom agent for specific calculations"
        )
    
    async def process(self, input_data):
        # Implementation here
        return self.create_success_response({"result": "processed"})
```

### Testing

```python
# Run main test suite
python main.py

# Test individual agents
python -c "
import asyncio
from agents.calc_executor import CalcExecutor

async def test():
    agent = CalcExecutor()
    result = await agent.process({
        'formula': '2 + 2',
        'parameters': {}
    })
    print(result)

asyncio.run(test())
"
```

### Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive error handling
3. Include logging for debugging
4. Write tests for new functionality
5. Update documentation

## License

This system is provided as-is for educational and research purposes. Please ensure compliance with OpenAI's usage policies and any applicable licenses for dependencies.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error details
3. Verify configuration and dependencies
4. Test with simplified examples

## Changelog

### Version 1.0
- Initial release with core agent functionality
- Support for energy calculations and Plexos integration
- Quality control and multi-format export capabilities
- Comprehensive documentation and examples

