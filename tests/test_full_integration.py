import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_full_integration():
    """
    Test the full integration with the correct project ID (proj_hq4gfL5gbCQvZEKgV4PNLQz0)
    """
    # Import utility functions
    from utils.llm_provider import get_llm_response, _setup_langfuse_env
    
    # Load environment variables
    _setup_langfuse_env()
    
    # Set the correct project ID
    os.environ["OPENAI_PROJECT_ID"] = "proj_hq4gfL5gbCQvZEKgV4PNLQz0"
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    print(f"\n=== Full Integration Test ===")
    print(f"API Key: {api_key[:10]}...{api_key[-5:]}")
    print(f"Project ID: {os.environ.get('OPENAI_PROJECT_ID')}")
    
    # Test 1: Simple query with system prompt
    try:
        print("\nTest 1: Simple query with system prompt")
        response = get_llm_response(
            prompt="Write a very short hello world in Python",
            system_prompt="You are a helpful assistant who writes concise code."
        )
        
        print("✅ SUCCESS: Simple query with system prompt")
        print("-"*50)
        print(response)
        print("-"*50)
    except Exception as e:
        print(f"❌ ERROR: Simple query with system prompt - {str(e)}")
    
    # Test 2: Query with messages
    try:
        print("\nTest 2: Query with messages array")
        response = get_llm_response(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a one-line Python script to print the current date"}
            ]
        )
        
        print("✅ SUCCESS: Query with messages array")
        print("-"*50)
        print(response)
        print("-"*50)
    except Exception as e:
        print(f"❌ ERROR: Query with messages array - {str(e)}")
    
    print("\n✅ Tests completed")

if __name__ == "__main__":
    test_full_integration()
