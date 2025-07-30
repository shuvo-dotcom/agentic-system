import json
import os
from typing import List, Dict, Any, Optional

LLM_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_settings.json')

def load_llm_config():
    with open(LLM_CONFIG_PATH, 'r') as f:
        return json.load(f)

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

def get_openai_client():
    """
    Get OpenAI client with Langfuse integration
    Returns: OpenAI client instance
    """
    config = load_llm_config()
    openai_conf = config['openai']
    
    # Setup Langfuse environment variables
    _setup_langfuse_env()
    
    # Always use Langfuse OpenAI for tracing and monitoring
    from langfuse.openai import OpenAI
    client = OpenAI(
        api_key=openai_conf['api_key'], 
        base_url=openai_conf.get('api_base', 'https://api.openai.com/v1')
    )
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

def get_llm_response(messages: List[Dict[str, str]], **kwargs) -> str:
    """
    messages: list of dicts (OpenAI/Anthropic format)
    kwargs: extra params (e.g., temperature)
    Returns: response text
    """
    config = load_llm_config()
    provider = config.get('provider', 'openai').lower()
    
    if provider == 'openai':
        client = get_openai_client()
        model = config['openai']['model']
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
        # Anthropic expects a single string prompt, not OpenAI-style messages
        prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
        response = client.messages.create(
            model=model,
            max_tokens=kwargs.get('max_tokens', 1024),
            messages=messages,
            temperature=kwargs.get('temperature', 0.7)
        )
        return response.content[0].text if hasattr(response, 'content') else response.completion
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

def chat_completion(messages: List[Dict[str, str]], **kwargs):
    """
    Direct chat completion with full response object
    """
    client = get_openai_client()
    config = load_llm_config()
    model = config['openai']['model']
    return client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
