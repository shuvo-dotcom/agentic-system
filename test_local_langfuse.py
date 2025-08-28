#!/usr/bin/env python
"""
Test script to verify local Langfuse connection
"""
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env file

import os
import time
from langfuse import Langfuse

# Print environment configuration
print("Langfuse Environment Variables:")
print(f"LANGFUSE_PUBLIC_KEY: {os.getenv('LANGFUSE_PUBLIC_KEY')}")
print(f"LANGFUSE_SECRET_KEY: {'*' * 10 + os.getenv('LANGFUSE_SECRET_KEY')[-4:] if os.getenv('LANGFUSE_SECRET_KEY') else 'Not set'}")
print(f"LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST')}")
print(f"ENABLE_LANGFUSE: {os.getenv('ENABLE_LANGFUSE')}")

try:
    print("\nConnecting to Langfuse...")
    lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )
    print("✅ Langfuse client initialized successfully")
    
    print("\nCreating a test trace...")
    trace = lf.trace(
        name="local-test-trace",
        user_id="test-user",
        metadata={
            "env": "local",
            "test": True
        }
    )
    print(f"✅ Trace created with ID: {trace.id}")
    
    print("\nCreating a generation within the trace...")
    generation = trace.generation(
        name="test-generation",
        model="gpt-4o-mini",
        prompt="Tell me about nuclear generation in Belgium",
        completion="Belgium has multiple nuclear power plants that contribute significantly to its energy mix.",
        metadata={
            "temperature": 0.7,
            "max_tokens": 150
        }
    )
    print(f"✅ Generation created with ID: {generation.id}")
    
    print("\nCreating a span within the trace...")
    span = trace.span(
        name="data-processing",
        metadata={
            "processed_items": 42
        }
    )
    time.sleep(1)  # Simulate some processing time
    span.end()
    print(f"✅ Span created and ended with ID: {span.id}")
    
    print("\nAdding score to the trace...")
    trace.score(
        name="accuracy",
        value=0.95,
        comment="High accuracy response"
    )
    print("✅ Score added to trace")
    
    print("\nFinalizing trace...")
    trace.update(status="success")
    print("✅ Trace finalized")
    
    print("\n✅ All Langfuse operations completed successfully!")
    print(f"🔍 You can view this trace at: {os.getenv('LANGFUSE_HOST', 'http://localhost:3001')}/trace/{trace.id}")
    
except Exception as e:
    import traceback
    print(f"\n❌ Error: {str(e)}")
    print(traceback.format_exc())
