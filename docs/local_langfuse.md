# Local Langfuse Setup for Monitoring

This guide explains how to set up and use a local Langfuse instance for monitoring your agentic system.

## What is Langfuse?

Langfuse is an open-source observability and analytics platform for LLM applications. It helps you:
- Track prompt and response history
- Monitor model usage and performance
- Debug and analyze your application's behavior

## Running Langfuse Locally

The agentic system is already configured to work with a local Langfuse instance. To use it:

### 1. Start Local Langfuse

```bash
# Start the local Langfuse instance
./start_local_langfuse.sh
```

This starts:
- Langfuse server on port 3001
- PostgreSQL database on port 5432
- ClickHouse database on port 8123

### 2. Access the Dashboard

Open [http://localhost:3001/dashboard](http://localhost:3001/dashboard) in your browser.

When you first access the dashboard, you'll need to create an account.

### 3. Run Your Application

Your application is already configured to use the local Langfuse instance. The configuration in `.env` file includes:

```
LANGFUSE_SECRET_KEY=sk-lf-local-development-key
LANGFUSE_PUBLIC_KEY=pk-lf-local-development-key
LANGFUSE_HOST=http://localhost:3001
ENABLE_LANGFUSE=true
```

### 4. Test the Connection

You can test the connection to your local Langfuse instance:

```bash
python test_local_langfuse.py
```

### 5. Stop Langfuse

When you're done, stop the local Langfuse instance:

```bash
./stop_local_langfuse.sh
```

## Switching Between Local and Cloud

To switch between local and cloud Langfuse:

### For Cloud Langfuse

Update your `.env` file:

```
LANGFUSE_SECRET_KEY=sk-lf-your-cloud-secret-key
LANGFUSE_PUBLIC_KEY=pk-lf-your-cloud-public-key
LANGFUSE_HOST=https://cloud.langfuse.com
ENABLE_LANGFUSE=true
```

### For Local Langfuse

Update your `.env` file:

```
LANGFUSE_SECRET_KEY=sk-lf-local-development-key
LANGFUSE_PUBLIC_KEY=pk-lf-local-development-key
LANGFUSE_HOST=http://localhost:3001
ENABLE_LANGFUSE=true
```

## Data Persistence

Your local Langfuse data is stored in Docker volumes and persists between restarts. To completely remove the data:

```bash
docker-compose -f docker-compose.langfuse.yaml down -v
```

## Additional Information

- [Langfuse Documentation](https://langfuse.com/docs)
- [Docker Compose Configuration](./docker-compose.langfuse.yaml)
