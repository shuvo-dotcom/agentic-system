import os
import sys
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_openai_api_key():
    """
    Check the OpenAI API key configuration and provide guidance on how to fix issues
    """
    print("\n=== OpenAI API Key Configuration Check ===")
    
    # Load environment variables from .env
    from utils.llm_provider import _setup_langfuse_env
    _setup_langfuse_env()
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY", None)
    if not api_key:
        print("❌ No OpenAI API key found in environment variables")
        print("ℹ️ Please add your OpenAI API key to the .env file:")
        print("   OPENAI_API_KEY=sk-...")
        return False
    
    # Check API key format
    is_project_key = api_key.startswith("sk-proj-")
    if is_project_key:
        print("⚠️ Project-scoped API key detected (sk-proj-*)")
        print("ℹ️ This key format is usually for specialized OpenAI projects")
        print("ℹ️ For standard OpenAI API use, you typically need a key starting with 'sk-' (not 'sk-proj-')")
        print("ℹ️ You can get a standard OpenAI API key at: https://platform.openai.com/api-keys")
    else:
        print("✅ Standard OpenAI API key format detected")
    
    # Check the base URL configuration
    llm_settings_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_settings.json')
    with open(llm_settings_path, 'r') as f:
        config = json.load(f)
    
    base_url = config.get('openai', {}).get('api_base', 'https://api.openai.com/v1')
    print(f"🌐 API Base URL: {base_url}")
    
    if is_project_key and base_url == "https://api.openai.com/v1":
        print("⚠️ You are using a project-scoped API key with the standard OpenAI API endpoint")
        print("ℹ️ This may not work unless your project key is configured for the standard API")
        
        custom_url = input("Would you like to specify a custom API base URL for your project key? (y/n): ")
        if custom_url.lower() == 'y':
            new_url = input("Enter the API base URL for your project: ")
            if new_url:
                config['openai']['api_base'] = new_url
                with open(llm_settings_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"✅ Updated API base URL to: {new_url}")
    
    # Check the current provider
    provider = config.get('provider', 'openai')
    print(f"🔧 Current provider: {provider}")
    
    if provider != 'openai':
        change_provider = input("Switch to OpenAI provider? (y/n): ")
        if change_provider.lower() == 'y':
            config['provider'] = 'openai'
            with open(llm_settings_path, 'w') as f:
                json.dump(config, f, indent=2)
            print("✅ Set OpenAI as the current provider")
    
    print("\n=== Recommendations ===")
    if is_project_key:
        print("1. If you continue to experience authentication errors:")
        print("   - Verify if your project-scoped API key is valid")
        print("   - Check if you need a custom API base URL for your project")
        print("   - Consider getting a standard OpenAI API key (sk-*) from https://platform.openai.com/api-keys")
    else:
        print("1. Your API key appears to be in the correct format")
        print("   - Ensure the key is valid and has not expired")
        print("   - Verify you have sufficient credits in your OpenAI account")
    
    print("2. If you're experiencing issues with the 'system_prompt' parameter:")
    print("   - We've updated the code to handle it properly")
    print("   - It will now be converted to a system message in the messages array")
    
    print("\n=== Configuration Check Complete ===")
    return True

if __name__ == "__main__":
    check_openai_api_key()
