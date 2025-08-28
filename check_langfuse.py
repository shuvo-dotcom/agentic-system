"""
Simple test script to verify Langfuse connectivity
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3001")

print(f"Testing connection to Langfuse at: {langfuse_host}")

try:
    # Test the health endpoint
    response = requests.get(f"{langfuse_host}/api/public/health")
    print(f"Health endpoint status code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ Langfuse is running and accessible!")
    else:
        print("❌ Langfuse health check failed")
        
except Exception as e:
    print(f"❌ Error connecting to Langfuse: {str(e)}")

print("\nVerifying environment configuration:")
print(f"LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST')}")
print(f"LANGFUSE_PUBLIC_KEY: {os.getenv('LANGFUSE_PUBLIC_KEY')}")
print(f"LANGFUSE_SECRET_KEY: {'*' * 10 + os.getenv('LANGFUSE_SECRET_KEY')[-4:] if os.getenv('LANGFUSE_SECRET_KEY') else 'Not set'}")
print(f"ENABLE_LANGFUSE: {os.getenv('ENABLE_LANGFUSE')}")
