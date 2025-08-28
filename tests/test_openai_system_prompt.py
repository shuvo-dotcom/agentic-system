import os
import sys
import json
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_openai_system_prompt():
    """
    Test the OpenAI integration with system_prompt parameter
    """
    from utils.llm_provider import get_llm_response
    
    print("\n=== Testing OpenAI with system_prompt parameter ===")
    
    # First, ensure we're using the OpenAI provider
    llm_settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_settings.json')
    with open(llm_settings_path, 'r') as f:
        config = json.load(f)
    
    original_provider = config.get('provider', 'openai')
    if original_provider != 'openai':
        print(f"ℹ️ Temporarily switching provider from {original_provider} to openai for this test")
        config['provider'] = 'openai'
        with open(llm_settings_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    try:
        # Test with system_prompt parameter
        try:
            response = await get_llm_response(
                prompt="Write a short hello world program in Python",
                max_tokens=150,
                temperature=0.3,
                system_prompt="You are a helpful coding assistant."
            )
            print("✅ Success with system_prompt parameter!")
            print("✅ Response:")
            print("="*50)
            print(response)
            print("="*50)
            return True
        except Exception as e:
            print(f"❌ Error with system_prompt parameter: {str(e)}")
            
            # Alternate test method with messages array
            print("\nℹ️ Trying alternative approach with messages array instead of system_prompt")
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful coding assistant."},
                    {"role": "user", "content": "Write a short hello world program in Python"}
                ]
                response = await get_llm_response(
                    messages=messages,
                    max_tokens=150,
                    temperature=0.3
                )
                print("✅ Success with messages array!")
                print("✅ Response:")
                print("="*50)
                print(response)
                print("="*50)
                return True
            except Exception as e2:
                print(f"❌ Error with messages array: {str(e2)}")
                return False
    finally:
        # Restore original provider if we changed it
        if original_provider != 'openai':
            config['provider'] = original_provider
            with open(llm_settings_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"ℹ️ Restored original provider: {original_provider}")

if __name__ == "__main__":
    success = asyncio.run(test_openai_system_prompt())
    sys.exit(0 if success else 1)
