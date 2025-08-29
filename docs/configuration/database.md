# Database Configuration

This document covers the configuration options for database connections in the Agentic System.

## Overview

The Agentic System uses multiple database technologies to handle different types of data:

1. **Relational Databases** (PostgreSQL/SQLite): For structured data storage
2. **Vector Databases**: For semantic search and embedding storage
3. **File-Based Storage**: For CSV files and raw data

## PostgreSQL Configuration

PostgreSQL is used for structured data queries, especially through the PostgresDataProvider agent. Configuration is stored in `config/postgres_settings.json`.

### Basic Configuration

```json
{
  "host": "localhost",
  "port": 5432,
  "database": "energy_data",
  "user": "postgres",
  "password": "",
  "ssl_mode": "prefer",
  "timeout": 30,
  "connection_pool": {
    "min_size": 1,
    "max_size": 10
  },
  "application_name": "agentic_system"
}
```

### Docker Configuration

A separate configuration file `postgres_settings.docker.json` is provided for Docker environments:

```json
{
  "host": "postgres",
  "port": 5432,
  "database": "energy_data",
  "user": "postgres",
  "password": "postgres",
  "ssl_mode": "disable"
}
```

### Environment Variables

PostgreSQL settings can be overridden with environment variables:

| Variable | Description | Maps To |
|----------|-------------|---------|
| `POSTGRES_HOST` | Database host | `host` |
| `POSTGRES_PORT` | Database port | `port` |
| `POSTGRES_DB` | Database name | `database` |
| `POSTGRES_USER` | Username | `user` |
| `POSTGRES_PASSWORD` | Password | `password` |
| `POSTGRES_SSL_MODE` | SSL mode | `ssl_mode` |

### Connection String

Alternatively, the entire connection can be specified via the `DATABASE_URL` environment variable:

```
DATABASE_URL=postgresql://username:password@localhost:5432/energy_data
```

## Vector Database Configuration

The system uses a vector database for semantic search. Configuration is typically in environment variables:

```
VECTOR_DB_PATH=./data/vector_db
VECTOR_DB_TYPE=chromadb
```

### ChromaDB Configuration

For ChromaDB-specific settings:

```json
{
  "persist_directory": "./data/vector_db",
  "collection_name": "energy_documents",
  "embedding_function": "openai",
  "metadata_fields": ["source", "date", "author", "category"]
}
```

### Custom Vector Database Settings

For advanced vector database configurations:

```json
"vector_db": {
  "provider": "chromadb",
  "settings": {
    "persist_directory": "./data/vector_db",
    "collection_name": "energy_documents"
  },
  "embedding": {
    "model": "text-embedding-ada-002",
    "dimensions": 1536,
    "batch_size": 100
  }
}
```

## SQLite Configuration

For simpler deployments, SQLite is used as a lightweight database:

```
DATABASE_URL=sqlite:///./data.db
```

Additional SQLite configuration:

```json
"sqlite": {
  "filename": "./data.db",
  "journal_mode": "WAL",
  "synchronous": "NORMAL",
  "foreign_keys": true,
  "timeout": 30
}
```

## Self-Healing Query Configuration

Configuration for the self-healing query system used by the PostgresDataProvider:

```json
"self_healing": {
  "enabled": true,
  "max_fallback_attempts": 3,
  "strategies": ["time_relaxation", "geography_expansion", "metric_simplification", "general_relaxation"],
  "relaxation_factors": {
    "time": {
      "day_expansion": 7,
      "month_expansion": 1,
      "year_expansion": 1
    },
    "geography": ["state", "region", "country", "global"]
  }
}
```

## Connection Pooling

For high-performance applications, connection pooling is configured:

```json
"connection_pool": {
  "min_size": 5,
  "max_size": 20,
  "max_idle": 300,
  "max_queries": 50000,
  "setup": ["SET application_name TO 'agentic_system'"]
}
```

## Security Best Practices

- Store passwords and sensitive information in environment variables, not in config files
- Use connection string URLs with limited user privileges
- Enable SSL for all production connections
- Implement connection timeouts to prevent resource exhaustion

## Troubleshooting

Common database configuration issues:

| Issue | Solution |
|-------|----------|
| Connection refused | Check host, port, and firewall settings |
| Authentication failed | Verify username and password |
| Database not found | Confirm database name and create if necessary |
| Connection pool exhaustion | Increase `max_size` or check for connection leaks |
| Timeout errors | Adjust `timeout` setting or check query performance |
