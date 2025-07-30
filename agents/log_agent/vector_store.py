from utils.llm_provider import create_embeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from typing import List, Dict, Any, Optional
import hashlib

class QdrantVectorStore:
    def __init__(self, qdrant_url: str, openai_api_key: str, collection_name: str = 'agent_logs'):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key
        self.vector_size = 3072
        self._ensure_collection()

    def _ensure_collection(self):
        if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE
                )
            )

    def embed_text(self, text: str) -> List[float]:
        embeddings = create_embeddings([text], model="text-embedding-3-large")
        return embeddings[0]

    def upsert_log(self, log_id: str, message: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None):
        payload = metadata.copy() if metadata else {}
        payload['message'] = message
        
        # Add debug information about payload size
        payload_size = len(str(payload))
        print(f"[DEBUG] Upserting log {log_id[:8]}... with payload size: {payload_size} bytes")
        
        # Ensure no data truncation by checking key fields
        if 'full_message' in payload:
            print(f"[DEBUG] Full message length: {len(payload['full_message'])} chars")
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=log_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            print(f"[DEBUG] Successfully upserted log {log_id[:8]}...")
        except Exception as e:
            print(f"[ERROR] Failed to upsert log {log_id}: {e}")
            # Try with truncated message if it's too large
            if len(message) > 10000:  # Truncate very large messages
                payload['message'] = message[:10000] + "... [TRUNCATED]"
                payload['message_truncated'] = True
                payload['original_length'] = len(message)
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        qmodels.PointStruct(
                            id=log_id,
                            vector=embedding,
                            payload=payload
                        )
                    ]
                )
                print(f"[DEBUG] Upserted log {log_id[:8]}... with truncated message")
            else:
                raise

    def verify_log_storage(self, log_id: str) -> Dict[str, Any]:
        """Verify that a log was stored correctly and return its details."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[log_id]
            )
            if result:
                point = result[0]
                payload = point.payload
                return {
                    'stored': True,
                    'log_id': log_id,
                    'message_length': len(payload.get('message', '')),
                    'full_message_length': len(payload.get('full_message', '')),
                    'metadata_keys': list(payload.keys()),
                    'timestamp': payload.get('timestamp', ''),
                    'session_id': payload.get('session_id', ''),
                    'agent_name': payload.get('agent_name', ''),
                    'has_tree_info': 'tree_info' in payload
                }
            else:
                return {'stored': False, 'log_id': log_id}
        except Exception as e:
            return {'stored': False, 'log_id': log_id, 'error': str(e)}

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored logs."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            count = collection_info.points_count
            return {
                'total_logs': count,
                'collection_name': self.collection_name,
                'vector_size': self.vector_size
            }
        except Exception as e:
            return {'error': str(e)}

    def semantic_search(self, query: str, top_k: int = 5, depth_filter: Optional[int] = None, parent_id: Optional[str] = None, step_id: Optional[str] = None, parent_step_id: Optional[str] = None) -> List[Dict[str, Any]]:
        embedding = self.embed_text(query)
        
        # Build filter conditions
        filter_conditions = []
        if depth_filter is not None:
            filter_conditions.append(
                qmodels.FieldCondition(
                    key="tree_info.depth",
                    match=qmodels.MatchValue(value=depth_filter)
                )
            )
        if parent_id:
            filter_conditions.append(
                qmodels.FieldCondition(
                    key="tree_info.parent_id",
                    match=qmodels.MatchValue(value=parent_id)
                )
            )
        if step_id:
            filter_conditions.append(
                qmodels.FieldCondition(
                    key="tree_info.step_id",
                    match=qmodels.MatchValue(value=step_id)
                )
            )
        if parent_step_id:
            filter_conditions.append(
                qmodels.FieldCondition(
                    key="tree_info.parent_step_id",
                    match=qmodels.MatchValue(value=parent_step_id)
                )
            )
        
        search_params = {
            'collection_name': self.collection_name,
            'query_vector': embedding,
            'limit': top_k
        }
        
        if filter_conditions:
            search_params['query_filter'] = qmodels.Filter(
                must=filter_conditions
            )
        
        results = self.client.search(**search_params)
        return [{**r.payload, '_qdrant_id': r.id} for r in results]
    
    def get_children(self, log_id: str) -> List[Dict[str, Any]]:
        """Get all direct children of a log entry."""
        return self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="tree_info.parent_id",
                        match=qmodels.MatchValue(value=log_id)
                    )
                ]
            )
        )[0]
    
    def get_logs_by_depth(self, depth: int, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all logs at a specific depth level, optionally filtered by session."""
        filter_conditions = [
            qmodels.FieldCondition(
                key="tree_info.depth",
                match=qmodels.MatchValue(value=depth)
            )
        ]
        
        if session_id:
            filter_conditions.append(
                qmodels.FieldCondition(
                    key="session_id",
                    match=qmodels.MatchValue(value=session_id)
                )
            )
        
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(must=filter_conditions)
        )[0]
        
        return [{**point.payload, '_qdrant_id': point.id} for point in results]

    def get_logs_by_step(self, step_id: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all logs with a specific step_id, optionally filtered by session."""
        filter_conditions = [
            qmodels.FieldCondition(
                key="tree_info.step_id",
                match=qmodels.MatchValue(value=step_id)
            )
        ]
        
        if session_id:
            filter_conditions.append(
                qmodels.FieldCondition(
                    key="session_id",
                    match=qmodels.MatchValue(value=session_id)
                )
            )
        
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(must=filter_conditions)
        )[0]
        
        return [{**point.payload, '_qdrant_id': point.id} for point in results]

    def get_logs_by_parent_step(self, parent_step_id: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all logs with a specific parent_step_id, optionally filtered by session."""
        filter_conditions = [
            qmodels.FieldCondition(
                key="tree_info.parent_step_id",
                match=qmodels.MatchValue(value=parent_step_id)
            )
        ]
        
        if session_id:
            filter_conditions.append(
                qmodels.FieldCondition(
                    key="session_id",
                    match=qmodels.MatchValue(value=session_id)
                )
            )
        
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(must=filter_conditions)
        )[0]
        
        return [{**point.payload, '_qdrant_id': point.id} for point in results]

    def delete_log(self, log_id: str):
        """Delete a log from Qdrant by its ID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.PointIdsList(points=[log_id])
        )

    def delete_all_logs(self):
        """Delete all logs from the Qdrant collection."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.Filter(must=[])
        )

    def exists(self, log_id: str) -> bool:
        res = self.client.retrieve(collection_name=self.collection_name, ids=[log_id])
        return bool(res)