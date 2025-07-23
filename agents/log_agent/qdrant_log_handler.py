import logging
from .vector_store import QdrantVectorStore
import uuid

class QdrantLogHandler(logging.Handler):
    def __init__(self, qdrant_url, openai_api_key, collection_name='agent_logs'):
        super().__init__()
        self.vector_store = QdrantVectorStore(qdrant_url, openai_api_key, collection_name=collection_name)

    def emit(self, record):
        log_id = str(uuid.uuid4())
        message = self.format(record)
        embedding = self.vector_store.embed_text(message)
        if self.formatter:
            asctime = self.formatter.formatTime(record)
        else:
            import datetime
            asctime = datetime.datetime.fromtimestamp(record.created).isoformat()
        payload = {
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'asctime': asctime,
            'pathname': record.pathname,
            'lineno': record.lineno,
            'funcName': record.funcName,
        }
        self.vector_store.upsert_log(log_id, message, embedding, metadata=payload) 