import psycopg2
import psycopg2.extras
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class PostgresHandler:
    def __init__(self, connection_uri: str, db_name: Optional[str] = None):
        self.conn = psycopg2.connect(connection_uri)

    def execute(self, operation: str, table: str, data: Optional[Dict] = None, query: Optional[Dict] = None, update: Optional[Dict] = None) -> Any:
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        operation = operation.lower()
        if operation == 'insert':
            keys = ','.join(data.keys())
            values = ','.join(['%s'] * len(data))
            sql = f"INSERT INTO {table} ({keys}) VALUES ({values}) RETURNING *;"
            cur.execute(sql, list(data.values()))
            self.conn.commit()
            result = cur.fetchone()
            logger.info(f"Inserted into {table}: {data}")
            return result
        elif operation == 'find':
            where = ''
            vals = []
            if query:
                where = 'WHERE ' + ' AND '.join([f"{k}=%s" for k in query.keys()])
                vals = list(query.values())
            sql = f"SELECT * FROM {table} {where};"
            cur.execute(sql, vals)
            result = cur.fetchall()
            logger.info(f"Queried {table} with {query}, found {len(result)} records.")
            return result
        elif operation == 'update':
            set_clause = ', '.join([f"{k}=%s" for k in update.keys()])
            where = ''
            vals = list(update.values())
            if query:
                where = 'WHERE ' + ' AND '.join([f"{k}=%s" for k in query.keys()])
                vals += list(query.values())
            sql = f"UPDATE {table} SET {set_clause} {where} RETURNING *;"
            cur.execute(sql, vals)
            self.conn.commit()
            result = cur.fetchall()
            logger.info(f"Updated records in {table} where {query} with {update}")
            return result
        elif operation == 'delete':
            where = ''
            vals = []
            if query:
                where = 'WHERE ' + ' AND '.join([f"{k}=%s" for k in query.keys()])
                vals = list(query.values())
            sql = f"DELETE FROM {table} {where} RETURNING *;"
            cur.execute(sql, vals)
            self.conn.commit()
            result = cur.fetchall()
            logger.info(f"Deleted records from {table} where {query}")
            return result
        else:
            raise ValueError(f"Unsupported operation for PostgreSQL: {operation}")

    def close(self):
        self.conn.close()
        logger.info("Closed PostgreSQL connection.") 