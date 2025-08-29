# LLM Configuration

This document details the configuration options for language models in the Agentic System.

## Overview

The Agentic System relies heavily on Large Language Models (LLMs) for natural language understanding, code generation, formula creation, and other tasks. Configuring these models properly is essential for system performance and cost management.

## Configuration File

LLM settings are primarily stored in `config/llm_settings.json`. This file contains model selections, parameters, and prompt templates.

### Sample Configuration

```json
{
  "default_model": "gpt-4",
  "temperature": 0.1,
  "max_tokens": 4000,
  "request_timeout": 60,
  "retry_count": 3,
  "backoff_factor": 2,
  "models": {
    "gpt-4": {
      "system_message": "You are an expert energy analyst assistant specialized in energy calculations and data analysis. You have deep knowledge of energy economics, renewable energy systems, and power markets. You can perform detailed calculations including LCOE, NPV, IRR, capacity factors, and other energy-related metrics.",
      "temperature": 0.1,
      "max_tokens": 4000
    },
    "gpt-3.5-turbo": {
      "system_message": "You are a helpful energy assistant that can perform calculations and answer questions about energy systems.",
      "temperature": 0.2,
      "max_tokens": 2048
    }
  },
  "embedding_model": "text-embedding-ada-002"
}
```

## Key Configuration Parameters

### Global Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `default_model` | The default model to use when not specified | gpt-4 |
| `temperature` | Global default temperature setting | 0.1 |
| `max_tokens` | Global default max tokens setting | 4000 |
| `request_timeout` | Timeout for API requests in seconds | 60 |
| `retry_count` | Number of retries for failed requests | 3 |
| `backoff_factor` | Exponential backoff factor for retries | 2 |

### Model-Specific Settings

Each model can have its own configuration that overrides the global settings:

| Parameter | Description |
|-----------|-------------|
| `system_message` | System prompt that defines the model's role and capabilities |
| `temperature` | Controls randomness (0.0-2.0, lower is more deterministic) |
| `max_tokens` | Maximum tokens in the response |
| `top_p` | Nucleus sampling parameter (optional) |
| `frequency_penalty` | Penalty for token frequency (optional) |
| `presence_penalty` | Penalty for token presence (optional) |

### Embedding Model

| Parameter | Description | Default |
|-----------|-------------|---------|
| `embedding_model` | Model used for creating vector embeddings | text-embedding-ada-002 |

## Agent-Specific Prompts

Different agents use specialized prompts stored in the configuration:

### Formula Resolver Prompts

```json
"formula_resolver": {
  "system_prompt": "You are an expert in energy calculations and formulas...",
  "metric_identification_prompt": "Identify the primary energy metric being requested in the following query...",
  "formula_creation_prompt": "Create the appropriate formula for calculating {metric}..."
}
```

### QC Auditor Prompts

```json
"qc_auditor": {
  "system_prompt": "You are a meticulous quality control expert for energy calculations...",
  "validation_prompt": "Validate the following calculation results for accuracy..."
}
```

## Environment Variables

LLM settings can be overridden with environment variables:

| Variable | Description | Maps To |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required for authentication |
| `OPENAI_API_BASE` | API base URL | Used for custom endpoints |
| `OPENAI_MODEL` | Model override | `default_model` |
| `OPENAI_TEMPERATURE` | Temperature override | `temperature` |
| `OPENAI_MAX_TOKENS` | Max tokens override | `max_tokens` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model override | `embedding_model` |

## Cost Management

The system includes features to manage API costs:

- **Caching**: Responses are cached to avoid duplicate API calls
- **Batching**: Multiple embedding requests are batched
- **Token counting**: Preprocessing to estimate token usage
- **Model selection**: Automatic downgrading to cheaper models for simpler tasks

## Advanced Configuration

### Model Fallbacks

The system can be configured to fall back to alternative models when the primary model is unavailable:

```json
"fallbacks": {
  "gpt-4": ["gpt-4-0613", "gpt-3.5-turbo"],
  "gpt-3.5-turbo": ["gpt-3.5-turbo-0613"]
}
```

### Custom Endpoints

For users running local models or using alternative providers:

```json
"custom_endpoints": {
  "local-llama": {
    "api_base": "http://localhost:8000/v1",
    "api_type": "open_ai",
    "api_version": null
  }
}
```

## Troubleshooting

Common LLM configuration issues and their solutions:

| Issue | Solution |
|-------|----------|
| API key errors | Check `OPENAI_API_KEY` is set correctly |
| Timeout errors | Increase `request_timeout` value |
| Context length errors | Reduce input size or switch to a model with larger context |
| High costs | Lower temperature, implement more caching, or use cheaper models |
