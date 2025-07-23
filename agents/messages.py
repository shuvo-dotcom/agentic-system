from dataclasses import dataclass

@dataclass
class TextMessage:
    content: str
    source: str

@dataclass
class Reset:
    pass

@dataclass
class UploadForCodeInterpreter:
    file_path: str

@dataclass
class UploadForFileSearch:
    file_path: str
    vector_store_id: str 