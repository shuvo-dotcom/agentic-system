# Configuration Overview

This section covers the various configuration options available in the Agentic System. Proper configuration is essential for optimal performance and integration with your specific environment.

## Configuration Methods

The Agentic System supports multiple configuration methods, with the following precedence order (highest to lowest):

1. Environment variables
2. `.env` file
3. Configuration files in the `config/` directory
4. Default values

## Core Configuration Files

### `config/settings.py`

Contains the central configuration logic and default values for the entire system:

```python
# Example settings.py structure
from pathlib import Path
import os
import json

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Load settings from environment variables or default values
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
```

### `config/llm_settings.json`

JSON configuration file for LLM-specific settings:

```json
{
  "default_model": "gpt-4",
  "temperature": 0.1,
  "max_tokens": 4000,
  "models": {
    "gpt-4": {
      "system_message": "You are an expert energy analyst assistant..."
    },
    "gpt-3.5-turbo": {
      "system_message": "You are a helpful energy assistant..."
    }
  }
}
```

### `config/postgres_settings.json`

Configuration for PostgreSQL database connections:

```json
{
  "host": "localhost",
  "port": 5432,
  "database": "energy_data",
  "user": "postgres",
  "password": "",
  "ssl_mode": "prefer",
  "timeout": 30
}
```

## Environment Variables

Key environment variables that can be set:

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

## Agent-Specific Configuration

Each agent has its own configuration parameters that can be adjusted:

- [LLM Settings](llm.md) - Configuration for language models
- [Database Settings](database.md) - Database connection parameters
- [Calculation Settings](calculations.md) - Parameters for the calculation engine
- [Logging Settings](logging.md) - Log levels and formatting options

## Configuration Helper

The system includes a `config_helper.py` utility for simplified configuration management:

```python
from config_helper import get_config

# Load specific configuration
postgres_config = get_config("postgres")
openai_config = get_config("openai")

# Access configuration values
database_name = postgres_config["database"]
api_key = openai_config["api_key"]
```

## Advanced Configuration

For advanced users, the system supports:

- **Dynamic configuration**: Updating configuration at runtime
- **Profile-based configuration**: Loading different settings for development, testing, and production
- **Encrypted secrets**: Storing sensitive configuration values securely

See the [Advanced Configuration](advanced.md) page for more details.
