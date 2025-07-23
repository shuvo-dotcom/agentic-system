import logging
from typing import List, Dict, Any, Optional
from .vector_store import QdrantVectorStore
from .llm_utils import summarize_logs, answer_log_query
from pymongo import MongoClient
import os
import json
import traceback
import uuid
import re
import ast

class LogHandler:
    def __init__(self, mongo_uri: str, mongo_db: str, mongo_collection: str, qdrant_url: str, openai_api_key: str, qdrant_collection: str = 'agent_logs'):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.mongo_collection = mongo_collection
        self.qdrant_url = qdrant_url
        self.openai_api_key = openai_api_key
        self.qdrant_collection = qdrant_collection
        self.vector_store = QdrantVectorStore(qdrant_url, openai_api_key, collection_name=qdrant_collection)
        self.mongo_client = MongoClient(mongo_uri)
        self.mongo_logs = self.mongo_client[mongo_db][mongo_collection]
        self.logger = logging.getLogger(__name__)
        self.depth_tracking = {}

    def ingest_logs(self, limit: Optional[int] = None):
        """Ingest logs from MongoDB, embed, and store in Qdrant."""
        cursor = self.mongo_logs.find().sort([('_id', -1)])
        if limit:
            cursor = cursor.limit(limit)
        count = 0
        for log in cursor:
            log_id = str(log.get('_id'))
            message = log.get('message', '')
            if not message:
                continue
            # Check if already in Qdrant
            if self.vector_store.exists(log_id):
                continue
            embedding = self.vector_store.embed_text(message)
            self.vector_store.upsert_log(log_id, message, embedding, metadata=log)
            count += 1
        self.logger.info(f"Ingested {count} logs into Qdrant.")
        return count

    def search_logs(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search logs using Qdrant and OpenAI embedding."""
        return self.vector_store.semantic_search(query, top_k=top_k)

    def keyword_search(self, keyword: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Keyword search using MongoDB."""
        cursor = self.mongo_logs.find({"message": {"$regex": keyword, "$options": "i"}}).limit(top_k)
        return list(cursor)

    def summarize(self, logs: Optional[List[Dict[str, Any]]] = None) -> str:
        """Summarize logs using LLM."""
        if logs is None:
            logs = list(self.mongo_logs.find().sort(["_id", -1]).limit(20))
        return summarize_logs([log.get('message', '') for log in logs], self.openai_api_key)

    def answer_query(self, question: str) -> str:
        """LLM-powered Q&A over logs."""
        logs = list(self.mongo_logs.find().sort(["_id", -1]).limit(100))
        return answer_log_query(question, [log.get('message', '') for log in logs], self.openai_api_key)

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Stub: Detect recurring errors/anomalies for self-healing."""
        # TODO: Implement anomaly detection logic
        return []

    def trigger_self_healing(self, anomaly: Dict[str, Any]):
        """Stub: Trigger automated remediation based on anomaly."""
        # TODO: Implement self-healing logic
        pass

    def self_heal_session(self, session_id: int, top_k: int = 100):
        """Scan logs for a session_id for errors, use LLM to suggest a fix, and log the self-healing action."""
        # 1. Fetch all logs for the session_id from Qdrant
        logs = self.vector_store.semantic_search(str(session_id), top_k=top_k)
        # 2. Filter for error logs
        error_logs = [log for log in logs if log.get('level', '').upper() == 'ERROR']
        if not error_logs:
            return {'status': 'no_errors', 'message': 'No errors found for this session.', 'error_logs': []}
        # 3. Prepare context for LLM
        log_texts = [log.get('message', '') for log in logs]
        context = '\n'.join(log_texts)
        prompt = f"Given the following agent logs for session_id {session_id}, identify the root cause of the error(s) and suggest a fix.\n\nLogs:\n{context}\n\nRoot Cause and Suggested Fix:"
        import openai
        openai.api_key = self.openai_api_key
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        suggestion = response.choices[0].message.content.strip()
        # 4. Log the self-healing action
        self.logger.info(f"Self-healing suggestion for session {session_id}: {suggestion}", extra={
            'session_id': session_id,
            'agent_name': 'LogHandler',
            'stage': 'self_heal',
            'step_id': 'self_heal',
            'parent_step_id': None
        })
        return {'status': 'error_found', 'suggestion': suggestion, 'error_logs': error_logs}

    def upload_logs_from_file(self, log_file_path: str, batch_size: int = 16):
        """Read logs from a local file, batch embed, and upload to Qdrant with structured metadata."""
        log_pattern = re.compile(
            r'(?P<asctime>.*?) - (?P<name>.*?) - (?P<level>.*?) - \[session_id:(?P<session_id>.*?)\] \[agent:(?P<agent_name>.*?)\]' \
            r'( \[depth:(?P<depth>\d*)\])?( \[parent:(?P<parent_id>[^\]]*)\])?( \[step_id:(?P<step_id>[^\]]*)\])?( \[parent_step_id:(?P<parent_step_id>[^\]]*)\])? - (?P<message>.*)'
        )
        print(f"[DEBUG] Qdrant upload: Reading from {log_file_path}")
        if not os.path.exists(log_file_path):
            self.logger.warning(f"Log file {log_file_path} does not exist.")
            print(f"[DEBUG] Qdrant upload: Log file {log_file_path} does not exist.")
            return 0
        with open(log_file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"[DEBUG] Qdrant upload: {len(lines)} lines to upload.")
        if not lines:
            self.logger.info(f"No logs to upload from {log_file_path}.")
            print(f"[DEBUG] Qdrant upload: No logs to upload from {log_file_path}.")
            return 0
        count = 0
        found_call_tree = False
        current_session_tree = {}

        def process_batch(batch, current_batch_size):
            nonlocal count
            try:
                import openai
                openai.api_key = self.openai_api_key
                print(f"[DEBUG] Qdrant upload: Embedding batch with size {current_batch_size}")
                response = openai.embeddings.create(
                    input=batch,
                    model="text-embedding-3-large"
                )
                embeddings = [item.embedding for item in response.data]
                return embeddings
            except openai.BadRequestError as e:
                if "maximum context length" in str(e):
                    if current_batch_size > 1:
                        # Split the batch and try again
                        mid = current_batch_size // 2
                        print(f"[DEBUG] Qdrant upload: Reducing batch size to {mid}")
                        first_half = process_batch(batch[:mid], mid)
                        if not first_half:
                            print(f"[DEBUG] Qdrant upload: First half failed, skipping remaining")
                            return None
                        second_half = process_batch(batch[mid:], current_batch_size - mid)
                        if not second_half:
                            print(f"[DEBUG] Qdrant upload: Second half failed, returning first half only")
                            return first_half
                        return first_half + second_half
                    else:
                        self.logger.error(f"Single log too long: {e}")
                        print(f"[DEBUG] Qdrant upload: Single log too long: {e}")
                        return None
                self.logger.error(f"Embedding batch failed: {e}")
                print(f"[DEBUG] Qdrant upload: Embedding batch failed: {e}\n{traceback.format_exc()}")
                return None
            except Exception as e:
                self.logger.error(f"Embedding batch failed: {e}")
                print(f"[DEBUG] Qdrant upload: Embedding batch failed: {e}\n{traceback.format_exc()}")
                return None

        i = 0
        while i < len(lines):
            batch = lines[i:i+batch_size]
            embeddings = process_batch(batch, len(batch))
            if not embeddings:
                # If batch processing failed, try processing each log individually
                print(f"[DEBUG] Qdrant upload: Batch failed, trying individual logs")
                for single_log in batch:
                    single_embedding = process_batch([single_log], 1)
                    if single_embedding:
                        log_line = single_log
                        embedding = single_embedding[0]
                        log_id = str(uuid.uuid4())
                        match = log_pattern.match(log_line)
                        if not match:
                            self.logger.warning(f"Log line did not match expected pattern: {log_line}")
                            print(f"[DEBUG] Qdrant upload: Log line did not match pattern: {log_line}")
                            continue
                        metadata = {k: (v.strip() if v is not None else '') for k, v in match.groupdict().items()}
                        metadata["source"] = "file_upload"

                        session_id = metadata.get('session_id', '')
                        depth_str = metadata.get('depth', '')
                        if depth_str is None:
                            depth = 0
                        elif isinstance(depth_str, str) and depth_str.strip().isdigit():
                            depth = int(depth_str.strip())
                        else:
                            depth = 0
                        parent_id = metadata.get('parent_id', '')
                        step_id = metadata.get('step_id', '')
                        parent_step_id = metadata.get('parent_step_id', '')

                        if session_id not in current_session_tree:
                            current_session_tree[session_id] = {}
                        
                        current_session_tree[session_id][log_id] = {
                            'depth': depth,
                            'parent_id': parent_id,
                            'step_id': step_id,
                            'parent_step_id': parent_step_id,
                            'children': []
                        }
                        
                        if parent_id and parent_id in current_session_tree[session_id]:
                            current_session_tree[session_id][parent_id]['children'].append(log_id)
                        
                        metadata['tree_info'] = {
                            'depth': depth,
                            'parent_id': parent_id,
                            'step_id': step_id,
                            'parent_step_id': parent_step_id,
                            'children': current_session_tree[session_id][log_id]['children']
                        }
                        
                        try:
                            self.vector_store.upsert_log(log_id, log_line, embedding, metadata=metadata)
                            print(f"[DEBUG] Qdrant upload: Upserted individual log_id {log_id}")
                            count += 1
                        except Exception as e:
                            self.logger.error(f"Qdrant upsert failed for individual log: {e}")
                            print(f"[DEBUG] Qdrant upload: Upsert failed for individual log_id {log_id}: {e}\n{traceback.format_exc()}")
                i += 1  # Move to next log after individual processing
                continue

            # Process the successful batch
            for log_line, embedding in zip(batch, embeddings):
                log_id = str(uuid.uuid4())
                match = log_pattern.match(log_line)
                if not match:
                    self.logger.warning(f"Log line did not match expected pattern: {log_line}")
                    print(f"[DEBUG] Qdrant upload: Log line did not match pattern: {log_line}")
                    continue
                metadata = {k: (v.strip() if v is not None else '') for k, v in match.groupdict().items()}
                metadata["source"] = "file_upload"
                # Track depth and relationships
                session_id = metadata.get('session_id')
                depth_str = metadata.get('depth', '')
                if depth_str is None or not depth_str.strip().isdigit():
                    depth = 0
                else:
                    depth = int(depth_str.strip())
                parent_id = metadata.get('parent_id', '')
                step_id = metadata.get('step_id', '')
                parent_step_id = metadata.get('parent_step_id', '')
                
                if session_id not in current_session_tree:
                    current_session_tree[session_id] = {}
                
                current_session_tree[session_id][log_id] = {
                    'depth': depth,
                    'parent_id': parent_id,
                    'step_id': step_id,
                    'parent_step_id': parent_step_id,
                    'children': []
                }
                
                if parent_id and parent_id in current_session_tree[session_id]:
                    current_session_tree[session_id][parent_id]['children'].append(log_id)
                
                metadata['tree_info'] = {
                    'depth': depth,
                    'parent_id': parent_id,
                    'step_id': step_id,
                    'parent_step_id': parent_step_id,
                    'children': current_session_tree[session_id][log_id]['children']
                }

                try:
                    self.vector_store.upsert_log(log_id, log_line, embedding, metadata=metadata)
                    print(f"[DEBUG] Qdrant upload: Upserted log_id {log_id}")
                    count += 1
                except Exception as e:
                    self.logger.error(f"Qdrant upsert failed: {e}")
                    print(f"[DEBUG] Qdrant upload: Upsert failed for log_id {log_id}: {e}\n{traceback.format_exc()}")
            i += batch_size  # Move to next batch

        self.logger.info(f"Uploaded {count} logs from {log_file_path} to Qdrant.")
        print(f"[DEBUG] Qdrant upload: Uploaded {count} logs from {log_file_path} to Qdrant.")
        return count

    def get_tree_structure(self, session_id: str) -> dict:
        """Retrieve and build the complete tree structure for a session."""
        logs = self.vector_store.semantic_search(str(session_id), top_k=1000)
        tree = {}
        root_nodes = []

        # First pass: Create nodes
        for log in logs:
            log_id = log.get('_qdrant_id')
            tree_info = log.get('tree_info', {})
            tree[log_id] = {
                'data': log,
                'depth': tree_info.get('depth', 0),
                'step_id': tree_info.get('step_id', ''),
                'parent_step_id': tree_info.get('parent_step_id', ''),
                'children': [],
                'parent_id': tree_info.get('parent_id', '')
            }
            if not tree_info.get('parent_id'):
                root_nodes.append(log_id)

        # Second pass: Build relationships
        for log_id, node in tree.items():
            parent_id = node['parent_id']
            if parent_id and parent_id in tree:
                tree[parent_id]['children'].append(log_id)

        return {'tree': tree, 'root_nodes': root_nodes}

    def analyze_logs_llm(self, question: str, session_id: int = None, top_k: int = 100):
        """LLM-powered analysis and suggestions over logs. If session_id is given, restrict to that session."""
        if session_id:
            logs = self.vector_store.semantic_search(str(session_id), top_k=top_k)
        else:
            logs = self.vector_store.semantic_search('log', top_k=top_k)
        log_texts = [log.get('message', '') for log in logs]
        context = '\n'.join(log_texts)
        prompt = f"Given the following agent logs, answer the user's question as precisely as possible.\n\nLogs:\n{context}\n\nQuestion: {question}\nAnswer:"
        import openai
        openai.api_key = self.openai_api_key
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        # Log the analysis action
        self.logger.info(f"LLM log analysis for question '{question}' (session_id={session_id}): {answer}", extra={
            'session_id': session_id,
            'agent_name': 'LogHandler',
            'stage': 'llm_log_analysis',
            'step_id': 'llm_log_analysis',
            'parent_step_id': None
        })
        return answer

    def drs_agent_decision_review(self, session_id: int, top_k: int = 100):
        """DRS Agent: Review all decision points in the logs for a session, critique them, and suggest alternatives."""
        logs = self.vector_store.semantic_search(str(session_id), top_k=top_k)
        log_texts = [log.get('message', '') for log in logs]
        context = '\n'.join(log_texts)
        prompt = (
            f"You are a Decision Review System (DRS) agent. "
            f"Given the following agent logs for session_id {session_id}, review all decision points (accept, retry, fallback, etc.). "
            f"For each, explain if the decision was optimal given the context and results, and suggest any better alternatives.\n\n"
            f"Logs:\n{context}\n\n"
            f"For each decision point, provide:\n- The decision taken\n- Was it optimal? Why or why not?\n- Alternative actions that could have been taken\n- Your overall suggestions for future improvement."
        )
        import openai
        openai.api_key = self.openai_api_key
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        drs_analysis = response.choices[0].message.content.strip()
        # Log the DRS analysis action
        self.logger.info(f"DRS agent decision review for session {session_id}: {drs_analysis}", extra={
            'session_id': session_id,
            'agent_name': 'DRSAgent',
            'stage': 'drs_decision_review',
            'step_id': 'drs_decision_review',
            'parent_step_id': None
        })
        return drs_analysis

    def get_subtree(self, root_id: str, max_depth: Optional[int] = None) -> dict:
        """Get the complete subtree starting from a given root node."""
        if max_depth is None:
            config = get_log_agent_config()
            max_depth = config['tree_settings'].get('max_depth', 10)
        
        root_log = self.vector_store.semantic_search(root_id, top_k=1)[0]
        current_depth = root_log.get('tree_info', {}).get('depth', 0)
        
        subtree = {
            'node': root_log,
            'children': []
        }
        
        if current_depth >= max_depth:
            return subtree
        
        children = self.vector_store.get_children(root_id)
        for child in children:
            child_subtree = self.get_subtree(child['_qdrant_id'], max_depth)
            subtree['children'].append(child_subtree)
        
        return subtree
    
    def get_logs_at_depth(self, depth: int, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all logs at a specific depth level."""
        return self.vector_store.get_logs_by_depth(depth, session_id)
    
    def get_logs_by_step(self, step_id: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all logs with a specific step_id."""
        return self.vector_store.get_logs_by_step(step_id, session_id)

    def get_logs_by_parent_step(self, parent_step_id: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all logs with a specific parent_step_id."""
        return self.vector_store.get_logs_by_parent_step(parent_step_id, session_id)

    def get_step_tree(self, step_id: str, session_id: Optional[str] = None) -> dict:
        """Get a tree structure based on step relationships."""
        # Get the root step
        root_logs = self.get_logs_by_step(step_id, session_id)
        if not root_logs:
            return {'tree': {}, 'root_nodes': []}

        tree = {}
        root_nodes = []

        # First pass: Create nodes for all related steps
        for log in root_logs:
            log_id = log.get('_qdrant_id')
            tree_info = log.get('tree_info', {})
            tree[log_id] = {
                'data': log,
                'depth': tree_info.get('depth', 0),
                'step_id': tree_info.get('step_id', ''),
                'parent_step_id': tree_info.get('parent_step_id', ''),
                'children': [],
                'parent_id': tree_info.get('parent_id', '')
            }
            if not tree_info.get('parent_step_id'):
                root_nodes.append(log_id)

            # Get child steps
            child_logs = self.get_logs_by_parent_step(tree_info.get('step_id', ''), session_id)
            for child_log in child_logs:
                child_id = child_log.get('_qdrant_id')
                child_tree_info = child_log.get('tree_info', {})
                tree[child_id] = {
                    'data': child_log,
                    'depth': child_tree_info.get('depth', 0),
                    'step_id': child_tree_info.get('step_id', ''),
                    'parent_step_id': child_tree_info.get('parent_step_id', ''),
                    'children': [],
                    'parent_id': child_tree_info.get('parent_id', '')
                }

        # Second pass: Build relationships based on step_id and parent_step_id
        for log_id, node in tree.items():
            parent_step_id = node['parent_step_id']
            for potential_parent_id, potential_parent in tree.items():
                if potential_parent['step_id'] == parent_step_id:
                    potential_parent['children'].append(log_id)
                    break

        return {'tree': tree, 'root_nodes': root_nodes}
    
    def get_path_to_root(self, log_id: str) -> List[Dict[str, Any]]:
        """Get the path from a log entry to its root node."""
        path = []
        current_log = self.vector_store.semantic_search(log_id, top_k=1)[0]
        
        while current_log:
            path.append(current_log)
            parent_id = current_log.get('tree_info', {}).get('parent_id')
            if not parent_id:
                break
            current_log = self.vector_store.semantic_search(parent_id, top_k=1)[0]
        
        return list(reversed(path))