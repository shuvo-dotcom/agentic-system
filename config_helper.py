#!/usr/bin/env python3
"""
LLM Provider Configuration Helper
"""

import sys
import os
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.llm_provider import (
    get_available_providers, 
    switch_provider, 
    get_current_model, 
    load_llm_config,
    LLM_CONFIG_PATH
)

def show_current_config():
    """Display current LLM configuration"""
    config = load_llm_config()
    print("\n=== Current LLM Configuration ===")
    print(f"Active Provider: {config.get('provider', 'Not set')}")
    print(f"Available Providers: {get_available_providers()}")
    
    try:
        print(f"Current Model: {get_current_model()}")
    except Exception as e:
        print(f"Current Model: Error - {e}")
    
    print("\n=== Provider Details ===")
    for provider in ['openai', 'anthropic', 'lmstudio']:
        if provider in config:
            print(f"\n{provider.upper()}:")
            provider_config = config[provider]
            for key, value in provider_config.items():
                if 'api_key' in key.lower():
                    # Mask API keys for security
                    masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                    print(f"  {key}: {masked_value}")
                else:
                    print(f"  {key}: {value}")

def update_lmstudio_config():
    """Update LM Studio configuration"""
    config = load_llm_config()
    
    print("\n=== Update LM Studio Configuration ===")
    current_base = config.get('lmstudio', {}).get('api_base', 'http://localhost:1234/v1')
    current_model = config.get('lmstudio', {}).get('model', 'model-name')
    
    print(f"Current API Base: {current_base}")
    print(f"Current Model: {current_model}")
    
    new_base = input(f"Enter new API base URL (press Enter to keep '{current_base}'): ").strip()
    if not new_base:
        new_base = current_base
        
    new_model = input(f"Enter new model name (press Enter to keep '{current_model}'): ").strip()
    if not new_model:
        new_model = current_model
    
    # Update configuration
    if 'lmstudio' not in config:
        config['lmstudio'] = {}
    
    config['lmstudio']['api_base'] = new_base
    config['lmstudio']['model'] = new_model
    
    # Save configuration
    with open(LLM_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ LM Studio configuration updated!")
    print(f"API Base: {new_base}")
    print(f"Model: {new_model}")

def main():
    """Main configuration menu"""
    while True:
        print("\n" + "="*50)
        print("LLM Provider Configuration Helper")
        print("="*50)
        print("1. Show current configuration")
        print("2. Switch provider")
        print("3. Update LM Studio configuration")
        print("4. Test current provider")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            show_current_config()
            
        elif choice == '2':
            providers = get_available_providers()
            if not providers:
                print("❌ No providers available!")
                continue
                
            print(f"\nAvailable providers: {providers}")
            provider = input("Enter provider name: ").strip().lower()
            
            try:
                switch_provider(provider)
                print(f"✅ Switched to {provider}")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif choice == '3':
            update_lmstudio_config()
            
        elif choice == '4':
            try:
                from utils.llm_provider import get_llm_response
                messages = [{"role": "user", "content": "Hello! Please respond with a short greeting."}]
                response = get_llm_response(messages, max_tokens=50)
                print(f"\n✅ Test successful! Response: {response}")
            except Exception as e:
                print(f"❌ Test failed: {e}")
                
        elif choice == '5':
            print("Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
