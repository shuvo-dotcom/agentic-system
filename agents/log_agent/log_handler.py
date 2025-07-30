import logging
from typing import List, Dict, Any, Optional
from .vector_store import QdrantVectorStore
from .llm_utils import summarize_logs, answer_log_query
from utils.llm_provider import get_llm_response, create_embeddings, get_openai_client
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

    def upload_logs_from_file(self, log_file_path: str, batch_size: int = 16):
        """Read logs from a local file, batch embed, and upload to Qdrant with structured metadata."""
        # Updated regex pattern to handle multi-line messages and all possible fields
        log_pattern = re.compile(
            r'(?P<asctime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (?P<name>\S+) - (?P<levelname>\w+) - \[session_id:(?P<session_id>\w+)\] \[agent:(?P<agent_name>[^\]]+)\] \[step_id:(?P<step_id>[^\]]*)\] \[parent_id:(?P<parent_id>[^\]]*)\] - (?P<message>.*)',
            re.DOTALL  # Allow . to match newlines for multi-line messages
        )
        print(f"[DEBUG] Qdrant upload: Reading from {log_file_path}")
        if not os.path.exists(log_file_path):
            self.logger.warning(f"Log file {log_file_path} does not exist.")
            print(f"[DEBUG] Qdrant upload: Log file {log_file_path} does not exist.")
            return 0
        
        # Read the entire file content to handle multi-line logs properly
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split logs by timestamp pattern (more reliable than line-by-line)
        log_entries = []
        lines = content.split('\n')
        current_log = ""
        
        for line in lines:
            if line.strip():
                # Check if this line starts a new log entry (has timestamp pattern)
                if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', line):
                    if current_log.strip():
                        log_entries.append(current_log.strip())
                    current_log = line
                else:
                    # This is a continuation of the previous log
                    current_log += "\n" + line
        
        # Don't forget the last log
        if current_log.strip():
            log_entries.append(current_log.strip())
            
        print(f"[DEBUG] Qdrant upload: {len(log_entries)} log entries to upload.")
        if not log_entries:
            self.logger.info(f"No logs to upload from {log_file_path}.")
            print(f"[DEBUG] Qdrant upload: No logs to upload from {log_file_path}.")
            return 0
        
        count = 0
        current_session_tree = {}

        def process_batch(batch, current_batch_size):
            nonlocal count
            try:
                print(f"[DEBUG] Qdrant upload: Embedding batch with size {current_batch_size}")
                embeddings = create_embeddings(batch, model="text-embedding-3-large")
                return embeddings
            except Exception as e:
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

        i = 0
        while i < len(log_entries):
            batch = log_entries[i:i+batch_size]
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
                        
                        # Enhanced metadata extraction - parse the message for structured data
                        message = metadata.get('message', '')
                        
                        # Try to extract structured data from the message
                        try:
                            # Look for decision-related data in the message
                            if 'Decision:' in message:
                                parts = message.split(' | ')
                                for part in parts:
                                    if ':' in part:
                                        key, value = part.split(':', 1)
                                        key = key.strip().lower().replace(' ', '_')
                                        metadata[f'extracted_{key}'] = value.strip()
                        except Exception as e:
                            print(f"[DEBUG] Failed to parse structured message: {e}")
                        
                        # Preserve the full message without truncation
                        metadata['full_message'] = message
                        metadata['message_length'] = len(message)
                        
                        # Add timestamp parsing
                        try:
                            from datetime import datetime
                            timestamp = datetime.strptime(metadata.get('asctime', ''), '%Y-%m-%d %H:%M:%S,%f')
                            metadata['timestamp'] = timestamp.isoformat()
                        except:
                            metadata['timestamp'] = metadata.get('asctime', '')

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
                
                # Enhanced metadata extraction
                message = metadata.get('message', '')
                metadata['full_message'] = message
                metadata['message_length'] = len(message)
                
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
        
        # Verify storage and provide statistics
        try:
            storage_stats = self.vector_store.get_storage_stats()
            print(f"[DEBUG] Storage verification: {storage_stats}")
            
            # Sample verification of a few logs
            if count > 0:
                print(f"[DEBUG] Verifying log storage...")
                sample_size = min(3, count)
                print(f"[DEBUG] Checking {sample_size} sample logs for proper storage")
        except Exception as e:
            print(f"[DEBUG] Verification failed: {e}")
        
        return count
