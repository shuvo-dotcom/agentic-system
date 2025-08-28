#!/bin/bash

# Start local Langfuse instance and wait for it to be ready
echo "🚀 Starting local Langfuse instance..."
docker-compose -f docker-compose.langfuse.yaml up -d

echo "⏳ Waiting for Langfuse to start..."
sleep 5

# Check if Langfuse is responding
echo "🔍 Checking Langfuse status..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/api/public/health

if [ $? -eq 0 ]; then
    echo "✅ Langfuse is running locally at http://localhost:3001"
    echo "📊 Open http://localhost:3001/dashboard in your browser to view the dashboard"
    echo ""
    echo "🔑 Authentication:"
    echo "   - Default login credentials will be created on first use"
    echo "   - Or use the Sign Up option to create a new account"
    echo ""
    echo "📝 In your application, Langfuse is configured with:"
    echo "   - LANGFUSE_HOST=http://localhost:3001"
    echo "   - LANGFUSE_PUBLIC_KEY=pk-lf-local-development-key"
    echo "   - LANGFUSE_SECRET_KEY=sk-lf-local-development-key"
else
    echo "❌ There was an issue starting Langfuse. Check docker logs:"
    echo "docker-compose -f docker-compose.langfuse.yaml logs"
fi
