from langfuse import Langfuse
import traceback

# Singleton Langfuse client
_lf = None

def get_langfuse():
    global _lf
    if _lf is None:
        try:
            _lf = Langfuse(
                public_key="pk-lf-18da7bf3-d09d-4737-b866-cbdb41cfcc2c",
                secret_key="sk-lf-9be76db0-8a1a-4662-89d0-e9a00460e10c",
                host="http://localhost:3001"
            )
        except Exception as e:
            print(f"[Langfuse] Failed to initialize: {e}")
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
