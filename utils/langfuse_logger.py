from langfuse import Langfuse
import traceback

# Singleton Langfuse client
_lf = None

def get_langfuse():
    global _lf
    if _lf is None:
        try:
            import os
            # Use environment variables with fallback to the hardcoded values
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-local-development-key")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-local-development-key")
            host = os.getenv("LANGFUSE_HOST", "http://localhost:3001")
            
            is_local = "localhost" in host or "127.0.0.1" in host
            
            _lf = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            
            if is_local:
                print(f"[Langfuse] 🔄 Using local Langfuse instance at {host}")
            else:
                print(f"[Langfuse] 🌐 Connected to cloud Langfuse at {host}")
        except Exception as e:
            print(f"[Langfuse] ❌ Failed to initialize: {e}")
            _lf = None
    return _lf

def safe_trace(name, user_id=None, **kwargs):
    try:
        lf = get_langfuse()
        if lf is None:
            return None
        trace = lf.trace(name=name, user_id=user_id, **kwargs)
        return trace
    except Exception as e:
        print(f"[Langfuse] Trace error: {e}\n{traceback.format_exc()}")
        return None

def safe_span(trace, name, **kwargs):
    try:
        if trace is None:
            return None
        span = trace.span(name=name, **kwargs)
        return span
    except Exception as e:
        print(f"[Langfuse] Span error: {e}\n{traceback.format_exc()}")
        return None

def safe_generation(trace, name, input=None, output=None, **kwargs):
    try:
        if trace is None:
            return None
        gen = trace.generation(name=name, input=input, output=output, **kwargs)
        return gen
    except Exception as e:
        print(f"[Langfuse] Generation error: {e}\n{traceback.format_exc()}")
        return None

def safe_log_metadata(trace, metadata: dict):
    try:
        if trace is not None and hasattr(trace, "update_metadata"):
            trace.update_metadata(metadata)
    except Exception as e:
        print(f"[Langfuse] Metadata error: {e}\n{traceback.format_exc()}")

def safe_trace_finish(trace):
    try:
        if trace is not None and hasattr(trace, "finish"):
            trace.finish()
    except Exception as e:
        print(f"[Langfuse] Trace finish error: {e}\n{traceback.format_exc()}")
