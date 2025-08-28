import json
import os
from typing import List, Dict, Any, Optional

LLM_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_settings.json')

# Feature flag to enable/disable Langfuse
ENABLE_LANGFUSE = os.getenv("ENABLE_LANGFUSE", "true").lower() == "true"

def load_llm_config():
    with open(LLM_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Use the provider specified in the config file
    if 'provider' not in config:
        config['provider'] = 'openai'  # Default to OpenAI if not specified
    
    return config

def _setup_langfuse_env():
    """Setup Langfuse environment variables from .env file"""
    # Load environment variables from .env file
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

def _build_nohm_api_url(project_id=None):
    """
    Build the appropriate API URL for Nohm project-scoped API keys
    """
    # Get project ID from env variable if not provided
    if not project_id:
        # First check environment variable
        project_id = os.getenv("OPENAI_PROJECT_ID", "")
        
        # If not in env, check config
        if not project_id:
            config = load_llm_config()
            openai_conf = config.get('openai', {})
            project_id = openai_conf.get('project_id', "")
    
    # Get API base URL from config
    config = load_llm_config()
    openai_conf = config.get('openai', {})
    configured_base_url = openai_conf.get('api_base')
    
    # If a base URL is explicitly set in the config, use that
    if configured_base_url:
        print(f"ℹ️ Using configured API base URL: {configured_base_url}")
        return configured_base_url
    
    # If we have a project ID, build the project URL
    if project_id and project_id.startswith("proj_"):
        # For Nohm projects, we'll use the standard OpenAI API endpoint
        # The project ID is included in the API key authentication
        base_url = "https://api.openai.com/v1"
        print(f"ℹ️ Using standard OpenAI endpoint for Nohm project: {base_url}")
        print(f"ℹ️ Project ID: {project_id}")
        return base_url
    
    # Default to standard OpenAI API URL
    return "https://api.openai.com/v1"

def get_openai_client():
    """
    Get OpenAI client with optional Langfuse integration
    Returns: OpenAI client instance
    """
    config = load_llm_config()
    openai_conf = config['openai']
    
    # Load environment variables
    _setup_langfuse_env()
    
    # Get API key from environment or config, but prioritize environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Fall back to config if environment variable is not set
    if not api_key:
        api_key = openai_conf.get('api_key', "YOUR_API_KEY")
        print("⚠️ Warning: Using API key from config file. For security, consider using environment variables.")
    
    # Check if this is a Nohm project-scoped API key
    is_project_key = api_key.startswith("sk-proj-")
    project_id = os.getenv("OPENAI_PROJECT_ID", openai_conf.get('project_id', ""))
    
    # Default headers
    default_headers = {}
    
    if is_project_key:
        print("🔑 Nohm project-scoped API key detected (sk-proj-*)")
        if project_id:
            print(f"🔑 Project ID: {project_id}")
            # Add the project header for Nohm project-scoped keys
            default_headers["OpenAI-Project"] = project_id
        
        # For Nohm project keys, we need the proper project endpoint
        base_url = _build_nohm_api_url(project_id)
        
        # Use Langfuse with Nohm project keys as requested by user
        # Check for runtime environment variable
        enable_langfuse = os.getenv("ENABLE_LANGFUSE", "").lower()
        if enable_langfuse == "":
            # No runtime variable, use the default
            use_langfuse = ENABLE_LANGFUSE
        else:
            # Use the runtime variable value
            use_langfuse = enable_langfuse == "true"
        
        # For Nohm projects, update the API base URL in config if needed
        if project_id and project_id == "proj_Wq7kwcaJaYn1IktsKKcAZISZ":
            print(f"🔑 Using specific configuration for Nohm project: {project_id}")
    else:
        # For standard OpenAI keys, use the configured base URL
        base_url = openai_conf.get('api_base', 'https://api.openai.com/v1')
        # Check for runtime environment variable (takes precedence)
        enable_langfuse = os.getenv("ENABLE_LANGFUSE", "").lower()
        if enable_langfuse == "":
            # No runtime variable, use the default
            use_langfuse = ENABLE_LANGFUSE
        else:
            # Use the runtime variable value
            use_langfuse = enable_langfuse == "true"
    
    if use_langfuse:
        # Try to use Langfuse OpenAI for tracing and monitoring
        try:
            from langfuse.openai import OpenAI
            client = OpenAI(
                api_key=api_key, 
                base_url=base_url,
                default_headers=default_headers
            )
            print("✅ Langfuse monitoring enabled")
            
            # Try to create a simple trace to verify Langfuse is working
            try:
                from utils.langfuse_logger import safe_trace
                test_trace = safe_trace("langfuse_test", metadata={"test": "verify_connection"})
                if test_trace:
                    print("✅ Langfuse connection verified")
                else:
                    print("⚠️ Langfuse trace creation failed")
            except Exception as e:
                print(f"⚠️ Langfuse verification failed: {str(e)}")
        except Exception as e:
            print(f"⚠️ Langfuse integration failed: {str(e)}. Falling back to standard OpenAI.")
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key, 
                base_url=base_url,
                default_headers=default_headers
            )
    else:
        # Use standard OpenAI client without Langfuse
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key, 
                base_url=base_url,
                default_headers=default_headers
            )
            print("ℹ️ Langfuse monitoring disabled")
            if is_project_key:
                print(f"ℹ️ Using Nohm project ID: {os.getenv('OPENAI_PROJECT_ID', 'Not specified')}")
        except Exception as e:
            print(f"⚠️ Error initializing OpenAI client: {str(e)}")
            raise
    return client

def get_embedding_client():
    """
    Get OpenAI client specifically for embeddings with Langfuse integration
    Returns: OpenAI client instance
    """
    return get_openai_client()

def create_embeddings(texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
    """
    Create embeddings using OpenAI with Langfuse integration
    """
    client = get_embedding_client()
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [embedding.embedding for embedding in response.data]

def ensure_lmstudio_default():
    """
    Ensure LM Studio is set as the default provider
    """
    config = load_llm_config()
    if config.get('provider') != 'lmstudio':
        config['provider'] = 'lmstudio'
        with open(LLM_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Set LM Studio as default provider")
    return config

def get_provider_status():
    """
    Get the current provider status and configuration
    Returns: Dict with provider information
    """
    config = load_llm_config()
    current_provider = config.get('provider', 'not_set')
    
    status = {
        'current_provider': current_provider,
        'is_lmstudio_default': current_provider == 'lmstudio',
        'available_providers': get_available_providers(),
        'current_model': None,
        'endpoint': None
    }
    
    try:
        status['current_model'] = get_current_model()
        if current_provider == 'lmstudio':
            status['endpoint'] = config['lmstudio'].get('api_base')
    except Exception as e:
        status['error'] = str(e)
    
    return status

def print_provider_status():
    """
    Print the current provider status in a user-friendly format
    """
    status = get_provider_status()
    
    print("\n" + "="*50)
    print("🔍 LLM Provider Status")
    print("="*50)
    
    if status['is_lmstudio_default']:
        print("✅ LM Studio is the default provider")
    else:
        print(f"⚠️  Current provider: {status['current_provider']} (not LM Studio)")
    
    print(f"📱 Current Model: {status.get('current_model', 'Unknown')}")
    
    if status.get('endpoint'):
        print(f"🌐 Endpoint: {status['endpoint']}")
    
    print(f"🎯 Available Providers: {', '.join(status['available_providers'])}")
    
    if status.get('error'):
        print(f"❌ Error: {status['error']}")
    
    print("="*50 + "\n")

def initialize_lmstudio_as_default():
    """
    Initialize the LLM provider with LM Studio as the enforced default.
    Call this function at the start of your application to ensure 
    LM Studio is always used throughout the project.
    """
    print("🚀 Initializing LLM Provider with LM Studio as default...")
    config = ensure_lmstudio_default()
    
    # Verify LM Studio configuration
    if 'lmstudio' not in config:
        raise ValueError("❌ LM Studio configuration not found in llm_settings.json")
    
    if not config['lmstudio'].get('api_base'):
        raise ValueError("❌ LM Studio API base URL not configured")
    
    if not config['lmstudio'].get('model'):
        raise ValueError("❌ LM Studio model not configured")
    
    print(f"✅ LM Studio initialized successfully!")
    print(f"   Model: {config['lmstudio']['model']}")
    print(f"   Endpoint: {config['lmstudio']['api_base']}")
    print("🎯 All LLM operations will now use LM Studio by default.\n")
    
    return config

def initialize_llm_provider(project_id=None):
    """
    Initialize the LLM provider based on the configured provider in settings.
    Respects the provider setting in llm_settings.json
    
    Args:
        project_id: Optional project ID for project-scoped API keys
    """
    # Check Langfuse status early
    enable_langfuse = os.getenv("ENABLE_LANGFUSE", "").lower()
    use_langfuse = enable_langfuse == "true" if enable_langfuse else ENABLE_LANGFUSE
    
    # Check if we're using local or cloud Langfuse
    if use_langfuse:
        # Force reload environment variables to ensure we get the latest values
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3001")
        is_local = "localhost" in langfuse_host or "127.0.0.1" in langfuse_host
        mode = "local" if is_local else "cloud"
        print(f"ℹ️ Langfuse monitoring enabled ({mode} instance at {langfuse_host})")
    else:
        print("ℹ️ Langfuse monitoring disabled")
    config = load_llm_config()
    provider = config.get('provider', 'openai').lower()
    
    # Set project ID in environment if provided
    if project_id:
        os.environ["OPENAI_PROJECT_ID"] = project_id
    
    print(f"🚀 Initializing LLM Provider with {provider.upper()} as configured provider...")
    
    if provider == 'openai':
        openai_conf = config.get('openai', {})
        if not openai_conf.get('api_key'):
            raise ValueError("❌ OpenAI API key not configured")
        print(f"✅ OpenAI initialized successfully!")
        print(f"   Model: {openai_conf.get('model', 'gpt-4')}")
        print(f"   Endpoint: {openai_conf.get('api_base', 'https://api.openai.com/v1')}")
        print("🎯 All LLM operations will now use OpenAI by default.\n")
        
    elif provider == 'lmstudio':
        lmstudio_conf = config.get('lmstudio', {})
        if not lmstudio_conf.get('api_base'):
            raise ValueError("❌ LM Studio API base URL not configured")
        print(f"✅ LM Studio initialized successfully!")
        print(f"   Model: {lmstudio_conf.get('model', 'Unknown')}")
        print(f"   Endpoint: {lmstudio_conf['api_base']}")
        print("🎯 All LLM operations will now use LM Studio by default.\n")
        
    elif provider == 'anthropic':
        anthropic_conf = config.get('anthropic', {})
        if not anthropic_conf.get('api_key'):
            raise ValueError("❌ Anthropic API key not configured")
        print(f"✅ Anthropic initialized successfully!")
        print(f"   Model: {anthropic_conf.get('model', 'claude-3-opus-20240229')}")
        print(f"   Endpoint: {anthropic_conf.get('api_base', 'https://api.anthropic.com/v1')}")
        print("🎯 All LLM operations will now use Anthropic by default.\n")
        
    else:
        raise ValueError(f"❌ Unsupported provider: {provider}. Supported providers: openai, lmstudio, anthropic")
    
    return config

def get_lmstudio_client():
    """
    Get LM Studio client (OpenAI-compatible)
    Returns: OpenAI client instance configured for LM Studio
    """
    config = load_llm_config()
    lmstudio_conf = config['lmstudio']
    
    from openai import OpenAI
    client = OpenAI(
        api_key="lm-studio",  # LM Studio doesn't require a real API key
        base_url=lmstudio_conf['api_base']
    )
    return client

def get_current_llm_client():
    """
    Get the current LLM client based on the configured provider
    Returns: Client instance for the current provider
    """
    config = load_llm_config()
    provider = config.get('provider', 'openai').lower()
    
    if provider == 'openai':
        return get_openai_client()
    elif provider == 'lmstudio':
        return get_lmstudio_client()
    elif provider == 'anthropic':
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Please install with 'pip install anthropic'")
        anthropic_conf = config['anthropic']
        return anthropic.Anthropic(
            api_key=anthropic_conf['api_key'],
            base_url=anthropic_conf.get('api_base')
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

def get_current_model():
    """
    Get the current model name based on the configured provider
    Returns: Model name string
    """
    config = load_llm_config()
    provider = config.get('provider', 'openai').lower()
    
    if provider == 'openai':
        return config['openai']['model']
    elif provider == 'lmstudio':
        return config['lmstudio']['model']
    elif provider == 'anthropic':
        return config['anthropic']['model']
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

def get_available_providers():
    """
    Get list of available providers from config
    Returns: List of provider names
    """
    config = load_llm_config()
    providers = []
    
    if 'openai' in config and config['openai'].get('api_key'):
        providers.append('openai')
    if 'anthropic' in config and config['anthropic'].get('api_key'):
        providers.append('anthropic')
    if 'lmstudio' in config and config['lmstudio'].get('api_base'):
        providers.append('lmstudio')
    
    return providers

def switch_provider(provider_name: str):
    """
    Switch the active LLM provider
    Args:
        provider_name: Name of the provider to switch to ('openai', 'anthropic', 'lmstudio')
    """
    config = load_llm_config()
    available_providers = get_available_providers()
    
    if provider_name not in available_providers:
        raise ValueError(f"Provider '{provider_name}' not available. Available providers: {available_providers}")
    
    config['provider'] = provider_name
    
    # Save the updated config
    with open(LLM_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Switched to provider: {provider_name}")
    print(f"Current model: {get_current_model()}")

def get_llm_response(messages: List[Dict[str, str]] = None, prompt: str = None, **kwargs) -> str:
    """
    messages: list of dicts (OpenAI/Anthropic format)
    prompt: alternative to messages - a string prompt
    kwargs: extra params (e.g., temperature, system_prompt)
    Returns: response text
    
    Note: This function uses the configured provider from llm_settings.json
    """
    config = load_llm_config()
    provider = config.get('provider', 'openai').lower()  # Default to openai
    
    # Handle system_prompt parameter for Nohm project compatibility
    system_prompt = kwargs.pop("system_prompt", None)
    
    # Check if we're using a project-scoped API key (Nohm)
    api_key = os.getenv("OPENAI_API_KEY", config.get('openai', {}).get('api_key', ''))
    is_project_key = api_key.startswith("sk-proj-")
    
    if system_prompt and messages is None:
        messages = [{"role": "system", "content": system_prompt}]
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
            prompt = None  # Set prompt to None as we've included it in messages
    elif system_prompt and isinstance(messages, list):
        # Insert system prompt at the beginning if it's not already there
        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
    
    # Handle string prompt if not already handled
    if prompt is not None:
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages.append({"role": "user", "content": prompt})
    
    # Ensure we have valid messages
    if messages is None:
        messages = []
    
    print(f"🔧 Using LLM provider: {provider.upper()}")
    
    try:
        if provider == 'openai':
            client = get_openai_client()
            model = config['openai']['model']
            
            # Set default temperature to 0.7 for OpenAI calls
            openai_kwargs = kwargs.copy()
            if 'temperature' not in openai_kwargs:
                openai_kwargs['temperature'] = 0.7
                
            # Clean up kwargs to prevent invalid parameters
            for invalid_param in ['system_prompt', 'langfuse_trace']:
                if invalid_param in openai_kwargs:
                    openai_kwargs.pop(invalid_param)
            
            # Make API call with proper error handling
            try:
                print(f"🤖 Model: {model}")
                
                # Add response_format parameter for JSON mode if requested
                if 'json_mode' in openai_kwargs and openai_kwargs.pop('json_mode', False):
                    openai_kwargs['response_format'] = {"type": "json_object"}
                    # According to OpenAI API, we need to have 'json' in the message content
                    if messages and isinstance(messages, list) and len(messages) > 0:
                        last_msg = messages[-1]
                        if "content" in last_msg and not "json" in last_msg["content"].lower():
                            last_msg["content"] += " Please provide the response in JSON format."
                            messages[-1] = last_msg
                elif 'response_format' in openai_kwargs and isinstance(openai_kwargs['response_format'], dict) and openai_kwargs['response_format'].get('type') == 'json_object':
                    # Keep the response_format as is, but ensure 'json' is in the messages
                    if messages and isinstance(messages, list) and len(messages) > 0:
                        last_msg = messages[-1]
                        if "content" in last_msg and not "json" in last_msg["content"].lower():
                            last_msg["content"] += " Please provide the response in JSON format."
                            messages[-1] = last_msg
                    
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **openai_kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"❌ OpenAI API error: {str(e)}")
                if "Invalid URL" in str(e) and "proj_" in str(e):
                    print("⚠️ This appears to be an issue with the Nohm project URL format.")
                    print("ℹ️ Trying with corrected project URL format...")
                    
                    # Try with a different URL format as a fallback
                    try:
                        # Use standard OpenAI API directly
                        base_url = "https://api.openai.com/v1"
                        client._base_url = base_url
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            **openai_kwargs
                        )
                        print("✅ Success with standard OpenAI API endpoint!")
                        return response.choices[0].message.content
                    except Exception as e2:
                        print(f"❌ Second attempt also failed: {str(e2)}")
                        raise e2
                else:
                    raise e
            
        elif provider == 'lmstudio':
            client = get_lmstudio_client()
            model = config['lmstudio']['model']
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
            
        elif provider == 'anthropic':
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic package not installed. Please install with 'pip install anthropic'")
            anthropic_conf = config['anthropic']
            client = anthropic.Anthropic(
                api_key=anthropic_conf['api_key'],
                base_url=anthropic_conf.get('api_base')
            )
            model = anthropic_conf['model']
            response = client.messages.create(
                model=model,
                max_tokens=kwargs.get('max_tokens', 1024),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7)
            )
            return response.content[0].text if hasattr(response, 'content') else response.completion
            
    except Exception as e:
        print(f"❌ Error with {provider}: {str(e)}")
        if provider != 'lmstudio':
            print("🔄 Falling back to LM Studio...")
            try:
                client = get_lmstudio_client()
                model = config['lmstudio']['model']
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as fallback_error:
                print(f"❌ LM Studio fallback also failed: {str(fallback_error)}")
                raise fallback_error
        else:
            raise e
    
    raise ValueError(f"Unknown LLM provider: {provider}")

def chat_completion(messages: List[Dict[str, str]], **kwargs):
    """
    Direct chat completion with full response object
    Uses the configured provider from llm_settings.json
    """
    config = load_llm_config()
    provider = config.get('provider', 'openai').lower()  # Default to openai

    print(f"🔧 Using LLM provider: {provider.upper()}")
    
    try:
        if provider == 'openai':
            client = get_openai_client()
            model = config['openai']['model']
            # Set default temperature to 1.0 for OpenAI calls
            openai_kwargs = kwargs.copy()
            if 'temperature' not in openai_kwargs:
                openai_kwargs['temperature'] = 1.0
                
            # Add response_format parameter for JSON mode if requested
            if 'response_format' in kwargs and isinstance(kwargs['response_format'], dict) and kwargs['response_format'].get('type') == 'json_object':
                openai_kwargs['response_format'] = {"type": "json_object"}
                # According to OpenAI API, we need to have 'json' in the message content
                if messages and isinstance(messages, list) and len(messages) > 0:
                    last_msg = messages[-1].copy() if isinstance(messages[-1], dict) else {"role": "user", "content": str(messages[-1])}
                    if "content" in last_msg and not "json" in last_msg["content"].lower():
                        last_msg["content"] += " Please provide the response in JSON format."
                        messages[-1] = last_msg
                
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **openai_kwargs
            )
            
        elif provider == 'lmstudio':
            client = get_lmstudio_client()
            model = config['lmstudio']['model']
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
        elif provider == 'openai':
            # This is a duplicate code block that should be removed in a future refactoring
            client = get_openai_client()
            model = config['openai']['model']
            # Set default temperature to 1.0 for OpenAI calls
            openai_kwargs = kwargs.copy()
            if 'temperature' not in openai_kwargs:
                openai_kwargs['temperature'] = 0.0
                
            # Add response_format parameter for JSON mode if requested
            if 'response_format' in kwargs and isinstance(kwargs['response_format'], dict) and kwargs['response_format'].get('type') == 'json_object':
                openai_kwargs['response_format'] = {"type": "json_object"}
                # According to OpenAI API, we need to have 'json' in the message content
                if messages and isinstance(messages, list) and len(messages) > 0:
                    last_msg = messages[-1].copy() if isinstance(messages[-1], dict) else {"role": "user", "content": str(messages[-1])}
                    if "content" in last_msg and not "json" in last_msg["content"].lower():
                        last_msg["content"] += " Please provide the response in JSON format."
                        messages[-1] = last_msg
                
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **openai_kwargs
            )
        else:
            # Fallback to LM Studio for any other provider
            print(f"🔄 Falling back to LM Studio for provider: {provider}")
            client = get_lmstudio_client()
            model = config['lmstudio']['model']
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
    except Exception as e:
        print(f"❌ Error with {provider}: {str(e)}")
        if provider != 'lmstudio':
            print("🔄 Falling back to LM Studio...")
            client = get_lmstudio_client()
            model = config['lmstudio']['model']
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
        else:
            raise e
