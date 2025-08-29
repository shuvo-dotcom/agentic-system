# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Agentic System.

## Installation Issues

### API Key Configuration

**Problem**: System fails with authentication errors
```
OpenAIError: Authentication failed. Please check your API key.
```

**Solution**:
1. Verify your API key is correctly set in the `.env` file
2. Check for extra whitespace around the key
3. Ensure the API key is active in your OpenAI account
4. Try exporting the key directly in your terminal:
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```

### Python Version Incompatibility

**Problem**: Installation fails with package compatibility errors
```
ERROR: Package requires Python>=3.11 but you have Python 3.9.5
```

**Solution**:
1. Check your Python version: `python --version`
2. Install Python 3.11 or higher
3. Create a new virtual environment with the correct Python version:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

### Missing Dependencies

**Problem**: Import errors when running the system
```
ImportError: No module named 'chromadb'
```

**Solution**:
1. Ensure you've installed all dependencies: `pip install -r requirements.txt`
2. Check for any OS-specific dependencies that might need separate installation
3. If using a virtual environment, verify it's activated
4. For specific package errors, try installing individually: `pip install chromadb`

## Configuration Issues

### Missing Configuration Files

**Problem**: System cannot find configuration files
```
FileNotFoundError: [Errno 2] No such file or directory: '.../config/llm_settings.json'
```

**Solution**:
1. Verify all configuration files exist in the `config/` directory
2. Create missing files based on example templates
3. Check file permissions
4. Ensure paths are correct for your OS

### Database Connection Errors

**Problem**: Unable to connect to PostgreSQL database
```
psycopg2.OperationalError: could not connect to server: Connection refused
```

**Solution**:
1. Verify database server is running
2. Check connection settings in `config/postgres_settings.json`
3. Ensure the database exists and user has appropriate permissions
4. Try connecting with a direct client like `psql` to verify credentials
5. Check for firewall or network restrictions

## Runtime Issues

### LLM Timeouts

**Problem**: LLM API calls time out
```
TimeoutError: OpenAI API request timed out: (read timeout=60)
```

**Solution**:
1. Increase the timeout setting in configuration
2. Check your internet connection
3. Verify the OpenAI service status
4. Consider reducing token length in requests
5. Implement retry logic for intermittent failures

### Memory Issues

**Problem**: System crashes with memory errors
```
MemoryError: Unable to allocate array with shape (50000, 1536)
```

**Solution**:
1. Reduce batch sizes for embedding operations
2. Process large datasets in smaller chunks
3. Increase system memory or use swap space
4. Close unused applications to free memory
5. Consider cloud deployment for memory-intensive operations

### Calculation Errors

**Problem**: Numerical errors in calculations
```
ValueError: Math domain error in calculation
```

**Solution**:
1. Check input parameters for invalid values (negative values, zeros)
2. Verify formulas in the FormulaResolver
3. Add validation for edge cases
4. Implement proper error handling for mathematical operations
5. Use the QCAuditor agent to validate results

## Agent-Specific Issues

### PlexosCSVLoader Issues

**Problem**: Cannot parse CSV files
```
Error: Unable to parse CSV file: Unexpected delimiter found
```

**Solution**:
1. Check CSV file format and encoding
2. Verify delimiter settings match the file format
3. Inspect the file for corruption or special characters
4. Try preprocessing the file with standard tools like `pandas`

### PostgresDataProvider Issues

**Problem**: Self-healing queries fail to find data
```
Error: Failed to retrieve data after multiple fallback attempts
```

**Solution**:
1. Check the data actually exists in the database
2. Increase `max_fallback_attempts` in configuration
3. Examine self-healing metadata for specific failure points
4. Add more relaxation strategies
5. Verify database indexes for performance

### RAGIndexer Issues

**Problem**: Vector search returns irrelevant results
```
Warning: Vector search similarity scores below threshold
```

**Solution**:
1. Rebuild the vector index with updated embeddings
2. Check embedding model configuration
3. Verify document preprocessing steps
4. Adjust similarity thresholds
5. Consider hybrid search with keyword filtering

## Logging and Debugging

### Enabling Debug Logs

To get more detailed logs for troubleshooting:

1. Set the log level in your `.env` file:
   ```
   LOG_LEVEL=DEBUG
   ```

2. Check the logs directory for detailed output:
   ```bash
   tail -f logs/agentic_system.log
   ```

3. Filter logs for specific components:
   ```bash
   grep "PostgresDataProvider" logs/agentic_system.log
   ```

### Common Log Patterns

Look for these patterns in logs to identify issues:

- `ERROR` messages indicate critical failures
- `WARNING` messages suggest potential issues that didn't cause failure
- Messages with `retry attempt #` indicate API retries
- Timing information with `processing_time_ms` helps identify performance bottlenecks

## Getting Help

If you continue to experience issues:

1. Check the GitHub repository issues section
2. Search for similar problems in the documentation
3. Run the system diagnostics:
   ```bash
   python -m diagnostics.system_check
   ```
4. Collect relevant logs and configuration for support requests
5. Provide a minimal reproducible example when asking for help
