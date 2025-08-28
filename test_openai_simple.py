"""
Test script to verify the OpenAI integration with project-scoped key
"""
import os
import sys
import json
from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_PROJECT_ID

def test_openai_integration():
    """Test OpenAI integration with project-scoped key"""
    print(f"Testing OpenAI integration with:")
    print(f"API Key: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-5:]}")
    print(f"Project ID: {OPENAI_PROJECT_ID}")
    print(f"Model: {OPENAI_MODEL}")
    
    # Create OpenAI client with project header
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://api.openai.com/v1",
        default_headers={"OpenAI-Project": OPENAI_PROJECT_ID}
    )
    
    try:
        print("\nSending test request to OpenAI API...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a very short hello world in Python"}
            ],
            max_tokens=50
        )
        
        print("\n✅ SUCCESS! Response:")
        print("-"*50)
        print(response.choices[0].message.content)
        print("-"*50)
        return True
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openai_integration()
    sys.exit(0 if success else 1)
