"""
JSON serialization utilities for dataclasses used in the system
"""
import json
from typing import Any
from agents.messages import TextMessage, Reset, UploadForCodeInterpreter, UploadForFileSearch

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles serialization of special classes
    like TextMessage, Reset, etc.
    """
    def default(self, obj):
        # Handle TextMessage
        if isinstance(obj, TextMessage):
            return {
                "__type__": "TextMessage",
                "content": obj.content,
                "source": obj.source
            }
        
        # Handle Reset
        elif isinstance(obj, Reset):
            return {
                "__type__": "Reset"
            }
        
        # Handle UploadForCodeInterpreter
        elif isinstance(obj, UploadForCodeInterpreter):
            return {
                "__type__": "UploadForCodeInterpreter",
                "file_path": obj.file_path
            }
        
        # Handle UploadForFileSearch
        elif isinstance(obj, UploadForFileSearch):
            return {
                "__type__": "UploadForFileSearch",
                "file_path": obj.file_path,
                "vector_store_id": obj.vector_store_id
            }
            
        # Let the base class handle other types
        return super().default(obj)

def custom_json_dumps(obj: Any) -> str:
    """
    Serialize an object to a JSON string, handling custom types
    """
    return json.dumps(obj, cls=CustomJSONEncoder)

def custom_object_hook(obj: dict) -> Any:
    """
    Deserialize a JSON object, handling custom types
    """
    if "__type__" in obj:
        obj_type = obj["__type__"]
        
        if obj_type == "TextMessage":
            return TextMessage(
                content=obj.get("content", ""),
                source=obj.get("source", "")
            )
        
        elif obj_type == "Reset":
            return Reset()
        
        elif obj_type == "UploadForCodeInterpreter":
            return UploadForCodeInterpreter(
                file_path=obj.get("file_path", "")
            )
        
        elif obj_type == "UploadForFileSearch":
            return UploadForFileSearch(
                file_path=obj.get("file_path", ""),
                vector_store_id=obj.get("vector_store_id", "")
            )
    
    return obj

def custom_json_loads(json_str: str) -> Any:
    """
    Deserialize a JSON string, handling custom types
    """
    return json.loads(json_str, object_hook=custom_object_hook)
