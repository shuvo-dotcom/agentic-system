#!/bin/bash

# Stop local Langfuse instance
echo "🛑 Stopping local Langfuse instance..."
docker-compose -f docker-compose.langfuse.yaml down

# Check if containers were stopped
if [ $? -eq 0 ]; then
    echo "✅ Langfuse has been stopped."
    echo "💾 Your data is preserved in Docker volumes for the next run."
    echo "   To completely remove the data, use:"
    echo "   docker-compose -f docker-compose.langfuse.yaml down -v"
else
    echo "❌ There was an issue stopping Langfuse."
fi
