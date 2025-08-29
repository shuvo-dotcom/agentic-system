# Introduction to Agentic System

The Agentic System is a comprehensive multi-agent framework designed specifically for energy sector analysis, calculations, and reporting. Built with AutoGen and leveraging OpenAI's powerful models, the system specializes in processing complex energy-related queries with precision and flexibility.

## What is Agentic System?

Agentic System is a modular, multi-agent architecture that processes complex energy analysis queries through specialized agents. Each agent has a specific responsibility and can be called independently or as part of a coordinated workflow.

The system excels at:

- Processing structured and unstructured data from multiple sources
- Performing complex calculations with automatic formula resolution
- Quality control and validation of results
- Flexible output formatting for various stakeholder needs

## System Architecture

The system follows a multi-tier architecture:

1. **User Interface Layer**: Handles user input and output formatting
2. **Task Manager**: Central orchestrator that routes queries to appropriate agents
3. **Agent Layer**: Specialized agents performing specific functions
4. **Data Layer**: Connects to various data sources and storage systems
5. **External Services**: Integration with APIs and external systems

## Key Components

### Core Components

- **Task Manager**: Central orchestrator that routes queries to appropriate agents
- **Base Agent**: Abstract base class providing common functionality for all agents
- **Configuration**: Centralized settings and environment management

### Specialized Agents

1. **PlexosCSVLoader**: Loads and processes Plexos energy modeling CSV files
2. **PostgresDataProvider**: Connects to external PostgreSQL endpoints for data retrieval
3. **DataHarvester**: Retrieves data from external sources and APIs
4. **RAGIndexer**: Builds searchable vector and SQL indices for knowledge retrieval
5. **FormulaResolver**: Identifies required metrics and retrieves canonical formulas
6. **CalcExecutor**: Executes numeric computations using Python sandbox
7. **QCAuditor**: Validates results for accuracy, unit consistency, and citations
8. **Exporter**: Formats and exports final answers in various formats

## Use Cases

The Agentic System is designed to tackle a wide range of energy sector analytical challenges:

- **Energy Cost Analysis**: Calculating LCOE, NPV, and IRR for energy projects
- **Data-Driven Decision Making**: Processing complex datasets for insights
- **Scenario Planning**: Modeling different energy scenarios with variable inputs
- **Quality Assurance**: Validating calculations and ensuring accuracy
- **Report Generation**: Creating standardized or custom reports from analyses

## Next Steps

To get started with the Agentic System:

1. Follow the [Installation Guide](installation.md) to set up your environment
2. Complete the [Quickstart Tutorial](quickstart.md) to run your first analysis
3. Explore the [Configuration Options](../configuration/index.md) to customize the system
