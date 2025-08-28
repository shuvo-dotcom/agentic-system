"""
Helper script to test Langfuse connectivity
"""
import os
from dotenv import load_dotenv
load_dotenv()

print("Langfuse Environment Variables:")
print(f"LANGFUSE_PUBLIC_KEY: {'Set' if os.getenv('LANGFUSE_PUBLIC_KEY') else 'Not set'}")
print(f"LANGFUSE_SECRET_KEY: {'Set' if os.getenv('LANGFUSE_SECRET_KEY') else 'Not set'}")
print(f"LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST', 'Not set')}")
print(f"ENABLE_LANGFUSE: {os.getenv('ENABLE_LANGFUSE', 'Not set')}")

try:
    from langfuse import Langfuse
    print("\nTrying to initialize Langfuse client...")
    lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )
    print("✅ Langfuse client initialized successfully")
    
    print("\nTrying to create a test trace...")
    trace = lf.trace(name="test-trace", user_id="test-user")
    print(f"✅ Trace created with ID: {trace.id}")
    
    print("\nTrying to create a generation...")
    generation = trace.generation(
        name="test-generation",
        model="test-model",
        prompt="This is a test prompt",
        completion="This is a test completion"
    )
    print(f"✅ Generation created with ID: {generation.id}")
    
    print("\nFinalizing trace...")
    trace.update(status="success")
    print("✅ Trace finalized")
    
    print("\n✅ All Langfuse operations completed successfully!")
    
except Exception as e:
    import traceback
    print(f"\n❌ Error: {str(e)}")
    print(traceback.format_exc())
