import logging
from typing import Any, Dict, Optional
from .mongo_handler import MongoHandler
from .postgres_handler import PostgresHandler
from .redis_handler import RedisHandler
from .pinecone_handler import PineconeHandler

logger = logging.getLogger(__name__)

class DatabaseInteractionAgent:
    def __init__(self, db_type: str, connection_uri: str, db_name: Optional[str] = None):
        self.db_type = db_type.lower()
        self.connection_uri = connection_uri
        self.db_name = db_name
        self.handler = self._get_handler()

    def _get_handler(self):
        if self.db_type == 'mongo':
            return MongoHandler(self.connection_uri, self.db_name)
        elif self.db_type == 'postgres':
            return PostgresHandler(self.connection_uri, self.db_name)
        elif self.db_type == 'redis':
            return RedisHandler(self.connection_uri, self.db_name)
        elif self.db_type == 'pinecone':
            return PineconeHandler(self.connection_uri, self.db_name)
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

    def execute(self, operation: str, collection_or_table: str, data: Optional[Dict] = None, query: Optional[Dict] = None, update: Optional[Dict] = None, session_id: int = None, agent_name: str = 'DatabaseInteractionAgent') -> Any:
        logger.info(f"DatabaseInteractionAgent executing {operation} on {collection_or_table}", extra={'session_id': session_id, 'agent_name': agent_name})
        return self.handler.execute(operation, collection_or_table, data, query, update)

    def close(self, session_id: int = None, agent_name: str = 'DatabaseInteractionAgent'):
        logger.info("DatabaseInteractionAgent closing connection", extra={'session_id': session_id, 'agent_name': agent_name})
        self.handler.close() 