# FAQs

## General Questions

### What is the Agentic System?

The Agentic System is a comprehensive multi-agent framework designed specifically for energy sector analysis, calculations, and reporting. It uses specialized agents to process complex queries, retrieve data, perform calculations, and generate reports.

### What can I use the Agentic System for?

The system excels at:
- Energy calculations (LCOE, NPV, IRR, etc.)
- Data analysis and processing
- Formula resolution and validation
- Quality control and reporting
- Multi-format exports

### Is the system open source?

Yes, the Agentic System is open source and available under the MIT license. You can use, modify, and distribute it according to the terms of this license.

## Technical Questions

### Which Python versions are supported?

The Agentic System requires Python 3.11 or higher. This requirement exists due to our use of newer language features and dependencies that require this version.

### Can I use a different LLM provider than OpenAI?

The system is designed with OpenAI's models as the default, but it can be configured to work with other providers that implement a compatible API. Check the [LLM Configuration](../configuration/llm.md) document for details.

### How do I contribute to the project?

Contributions are welcome! Please check our GitHub repository for contribution guidelines. Typical ways to contribute include submitting bug reports, feature requests, documentation improvements, and code contributions through pull requests.

## Installation Questions

### I'm getting a "No module named 'openai'" error, how do I fix it?

This usually means the OpenAI Python library wasn't properly installed. Try installing it manually:

```bash
pip install openai
```

### Do I need a GPU to run the system?

No, a GPU is not required as the system uses cloud-based LLMs by default. However, if you configure it to use local models, a GPU might significantly improve performance.

### Can I run the system in Docker?

Yes, we provide Docker configurations. See the Docker setup instructions in the [Installation Guide](../getting-started/installation.md#docker-installation).

## Usage Questions

### How do I customize the system for my specific needs?

The system is highly configurable through:
- Configuration files in the `config/` directory
- Environment variables
- Direct code customization

See the [Configuration](../configuration/index.md) section for details.

### Can the system handle large datasets?

Yes, but with some considerations:
- Large datasets may require more memory
- Processing time will increase with data size
- Consider using chunked processing for very large datasets
- Vector databases have size limits to consider

### How accurate are the calculations?

The system includes:
- Precise formula implementation
- Quality control validation
- Unit consistency checks
- Range validation

However, results are only as accurate as the input data and formulas provided. Always validate critical calculations using multiple methods.

## Troubleshooting Questions

### Why is the system timing out on large calculations?

Try:
1. Increasing the timeout settings in the configuration
2. Breaking large calculations into smaller parts
3. Optimizing the query to reduce data volume
4. Increasing system resources (memory/CPU)

### How do I debug agent failures?

1. Check the logs in the `logs/` directory
2. Set the log level to DEBUG in your configuration
3. Try calling the specific agent directly with simpler inputs
4. Check the error details in the response

### Why isn't my PostgreSQL connection working?

Common issues include:
- Incorrect credentials in the configuration
- Database server not running
- Network/firewall issues
- Missing database or tables
- Insufficient permissions

See the [Database Configuration](../configuration/database.md) document for troubleshooting steps.
