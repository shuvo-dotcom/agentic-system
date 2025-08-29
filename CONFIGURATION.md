# Agentic System Configuration Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
   - [Environment Variables](#environment-variables)
   - [Configuration Files](#configuration-files)
3. [LLM Provider Configuration](#llm-provider-configuration)
   - [OpenAI Integration](#openai-integration)
   - [Alternative LLM Providers](#alternative-llm-providers)
   - [Model Parameters](#model-parameters)
4. [Database Configuration](#database-configuration)
   - [PostgreSQL Setup](#postgresql-setup)
   - [Vector Database (Qdrant)](#vector-database-qdrant)
5. [Logging Configuration](#logging-configuration)
   - [Log Levels](#log-levels)
   - [Log Storage](#log-storage)
   - [Langfuse Integration](#langfuse-integration)
6. [Agent Customization](#agent-customization)
   - [Agent Parameters](#agent-parameters)
   - [Agent Communication](#agent-communication)
   - [Adding Custom Agents](#adding-custom-agents)
7. [Performance Tuning](#performance-tuning)
   - [Memory Management](#memory-management)
   - [Concurrency Settings](#concurrency-settings)
8. [Troubleshooting](#troubleshooting)
   - [Common Issues](#common-issues)
   - [Debugging](#debugging)
9. [Security Best Practices](#security-best-practices)
   - [API Key Management](#api-key-management)
   - [Access Control](#access-control)

## Introduction

The Agentic System is a multi-agent framework designed for energy sector analysis, calculations, and reporting. This configuration manual provides instructions for setting up and customizing the system to suit your specific needs.

The system follows a modular design with multiple specialized agents working together. Proper configuration ensures optimal performance, security, and functionality.

## Environment Setup

### Environment Variables

The system uses environment variables for sensitive configuration parameters. This approach enhances security by keeping sensitive information out of source code.

1. **Create a `.env` file**: Copy the provided `.env.example` file to create your own configuration:

```bash
cp .env.example .env
```

2. **Required Environment Variables**:

| Variable | Description | Example Value |
|----------|-------------|--------------|
| `OPENAI_API_KEY` | Your OpenAI API key | sk-abc123... |
| `OPENAI_MODEL` | The OpenAI model to use | gpt-4-turbo |
| `OPENAI_PROJECT_ID` | Project ID for Nohm integrations | proj_abc123... |

3. **Optional Environment Variables**:

| Variable | Description | Default Value |
|----------|-------------|--------------|
| `OPENAI_EMBEDDING_MODEL` | Model for embeddings | text-embedding-ada-002 |
| `DATABASE_URL` | Database connection string | sqlite:///./temp_data.db |
| `VECTOR_DB_PATH` | Path to vector database | ./temp_vector_db |
| `LANGFUSE_HOST` | Langfuse monitoring host | http://localhost:3001 |

### Configuration Files

The system uses several configuration files to control behavior:

1. **LLM Settings** (`config/llm_settings.json`):
   - Configures language model providers
   - Specifies models and endpoints

2. **Log Agent Settings** (`config/log_agent_settings.json`):
   - Controls logging behavior
   - Configures vector storage

3. **Constants** (`config/constants.py`):
   - System-wide constants
   - Default values for calculations

## LLM Provider Configuration

### OpenAI Integration

The system uses OpenAI as the primary language model provider. Configure OpenAI integration in `config/llm_settings.json`:

```json
{
  "provider": "openai",
  "openai": {
    "api_key": "YOUR_API_KEY",
    "model": "gpt-4-turbo",
    "api_base": "https://api.openai.com/v1",
    "project_id": "YOUR_PROJECT_ID"
  }
}
```

**Note**: It is recommended to use environment variables instead of hardcoding API keys in configuration files.

### Alternative LLM Providers

The system supports additional LLM providers:

1. **Anthropic (Claude)**:

```json
{
  "provider": "anthropic",
  "anthropic": {
    "api_key": "YOUR_ANTHROPIC_KEY",
    "model": "claude-3-opus-20240229",
    "api_base": "https://api.anthropic.com/v1"
  }
}
```

2. **LM Studio** (for local model deployment):

```json
{
  "provider": "lmstudio",
  "lmstudio": {
    "model": "google/gemma-3-1b",
    "api_base": "http://localhost:1234/v1"
  }
}
```

### Model Parameters

Common model parameters can be adjusted in `config/llm_settings.json` or when calling the LLM provider:

| Parameter | Description | Default Value |
|-----------|-------------|--------------|
| `temperature` | Controls randomness (0-2) | 0.7 |
| `max_tokens` | Maximum output length | 2048 |
| `json_mode` | Forces JSON output format | false |

## Database Configuration

### PostgreSQL Setup

For PostgreSQL integration, configure the connection details in the environment:

```
DATABASE_URL=postgresql://username:password@localhost:5432/database_name
```

Alternative: Configure in `config/postgres_settings.json`:

```json
{
  "host": "localhost",
  "port": 5432,
  "username": "username",
  "password": "password",
  "database": "database_name"
}
```

### Vector Database (Qdrant)

The system uses Qdrant for vector storage. Configure in `config/log_agent_settings.json`:

```json
{
  "qdrant_url": "http://localhost:6333",
  "qdrant_collection": "agent_logs"
}
```

For external Qdrant instances, update the URL and add authentication if required.

## Logging Configuration

### Log Levels

Control logging verbosity in `config/settings.py`:

```python
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Log Storage

Logs are stored in the `logs` directory. Configure log rotation parameters in `config/settings.py`:

```python
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5
```

### Langfuse Integration

The system integrates with Langfuse for LLM monitoring. Configure in the `.env` file:

```
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PUBLIC_KEY=pk-lf-local-development-key
LANGFUSE_SECRET_KEY=sk-lf-local-development-key
```

For production, use your Langfuse cloud keys instead of local development keys.

## Agent Customization

### Agent Parameters

Each agent has specific parameters that can be adjusted. Common parameters are found in their respective files in the `agents` directory.

Example: `LLMFormulaResolver` parameters in `agents/llm_formula_resolver.py`:

```python
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
```

### Agent Communication

Configure agent communication patterns in `agents/messages.py`. This controls how agents exchange information.

### Adding Custom Agents

To add a custom agent:

1. Create a new file in the `agents` directory
2. Inherit from `core.base_agent.BaseAgent`
3. Implement the required methods:
   - `process(self, input_data)`
   - `_validate_input(self, input_data)`
   - `_process_valid_input(self, input_data)`

Example template:

```python
from core.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("MyCustomAgent")
        
    def _validate_input(self, input_data):
        # Implement validation logic
        return True, None
        
    def _process_valid_input(self, input_data):
        # Implement processing logic
        return {"success": True, "data": {}}
```

## Performance Tuning

### Memory Management

For memory-intensive operations, adjust Python memory limits in `start.sh`:

```bash
export PYTHONMEMORYPERCENT=80
```

### Concurrency Settings

Control concurrency in `config/settings.py`:

```python
MAX_CONCURRENT_TASKS = 5
WORKER_THREADS = 3
```

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Error: "Authentication failed"
   - Solution: Verify your API key in the `.env` file

2. **Model Not Found**:
   - Error: "Model 'gpt-4-turbo' not found"
   - Solution: Check your OpenAI account for model access

3. **Database Connection Failures**:
   - Error: "Could not connect to database"
   - Solution: Verify connection string and credentials

### Debugging

Enable debug mode for more detailed logs:

```bash
python main.py --query "your query" --debug
```

Check logs in the `logs` directory:
- `agentic_system.log` - Main application logs
- `server.log` - API server logs
- Session logs - Individual query session logs

## Security Best Practices

### API Key Management

1. **Environment Variables**: Always use environment variables for API keys
2. **Key Rotation**: Regularly rotate API keys
3. **Limited Scope**: Use keys with limited permissions when possible

### Access Control

1. **Server Security**: If exposing the API, use proper authentication
2. **Network Isolation**: Run in a secure network environment
3. **Input Validation**: Validate all user inputs before processing

---

This configuration manual provides guidance for setting up and customizing the Agentic System. For additional assistance or to report issues, please refer to the project documentation or submit an issue on the repository.
