from pymongo import MongoClient
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class MongoHandler:
    def __init__(self, connection_uri: str, db_name: Optional[str] = None):
        self.client = MongoClient(connection_uri)
        self.db = self.client.get_database(db_name) if db_name else self.client.get_default_database()

    def execute(self, operation: str, collection: str, data: Optional[Dict] = None, query: Optional[Dict] = None, update: Optional[Dict] = None) -> Any:
        col = self.db[collection]
        operation = operation.lower()
        if operation == 'insert':
            result = col.insert_one(data)
            logger.info(f"Inserted into {collection}: {data}")
            return result.inserted_id
        elif operation == 'find':
            result = list(col.find(query or {}))
            logger.info(f"Queried {collection} with {query}, found {len(result)} records.")
            return result
        elif operation == 'update':
            result = col.update_many(query, {'$set': update})
            logger.info(f"Updated {result.modified_count} records in {collection} where {query} with {update}")
            return result.modified_count
        elif operation == 'delete':
            result = col.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} records from {collection} where {query}")
            return result.deleted_count
        else:
            raise ValueError(f"Unsupported operation for MongoDB: {operation}")

    def close(self):
        self.client.close()
        logger.info("Closed MongoDB connection.") 