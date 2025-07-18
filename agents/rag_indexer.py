"""
RAG Indexer Agent - Builds searchable vector and SQL indices for knowledge retrieval.
"""
import os
import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

import chromadb
from chromadb.config import Settings
import openai
from sqlalchemy import create_engine, text

from core.simple_base_agent import SimpleBaseAgent
from config.settings import (
    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, 
    DATABASE_URL, VECTOR_DB_PATH
)


class RAGIndexer(SimpleBaseAgent):
    """
    Agent responsible for building and maintaining searchable vector and SQL indices
    of all structured and unstructured data for retrieval-augmented generation.
    """
    
    def __init__(self):
        # Define tools for RAG operations
        super().__init__(
            name="RAGIndexer",
            description="Builds searchable vector and SQL indices for knowledge retrieval."
        )
        
        self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.sql_engine = create_engine(DATABASE_URL)
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")
        self.collections = {}
        
    def create_vector_index(self, documents: List[Dict], collection_name: str = "default") -> Dict[str, Any]:
        """
        Create or update vector index from documents.
        
        Args:
            documents: List of documents with 'text', 'metadata', and optional 'id'
            collection_name: Name of the collection to store vectors
        
        Returns:
            Index creation status and statistics
        """
        try:
            self.logger.info(f"Creating vector index for collection: {collection_name}")
            
            # Get or create collection
            try:
                collection = self.chroma_client.get_collection(collection_name)
            except:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"created_at": datetime.now().isoformat()}
                )
            
            # Prepare documents for indexing
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                if 'text' not in doc:
                    continue
                    
                texts.append(doc['text'])
                metadatas.append(doc.get('metadata', {}))
                ids.append(doc.get('id', f"doc_{i}_{datetime.now().timestamp()}"))
            
            if not texts:
                return {"error": "No valid documents with text content found"}
            
            # Generate embeddings using OpenAI
            embeddings = self._generate_embeddings(texts)
            
            # Add to collection
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            # Store collection reference
            self.collections[collection_name] = collection
            
            return {
                "success": True,
                "collection_name": collection_name,
                "documents_indexed": len(texts),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Vector index creation failed: {str(e)}"}
    
    def create_sql_index(self, data: List[Dict], table_name: str, schema: Dict = None) -> Dict[str, Any]:
        """
        Create or update SQL index from structured data.
        
        Args:
            data: List of dictionaries representing rows
            table_name: Name of the SQL table
            schema: Optional schema definition
        
        Returns:
            SQL index creation status
        """
        try:
            self.logger.info(f"Creating SQL index for table: {table_name}")
            
            if not data:
                return {"error": "No data provided for SQL indexing"}
            
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(data)
            
            # Create table with data
            # Using to_sql with method='multi' can be slow for large datasets
            # Instead, we'll insert in chunks
            chunk_size = 1000  # Adjust chunk size as needed
            for i, chunk in enumerate(np.array_split(df, len(df) // chunk_size + 1)):
                if not chunk.empty:
                    chunk.to_sql(
                        table_name, 
                        self.sql_engine, 
                        if_exists='replace' if i == 0 else 'append', 
                        index=False
                    )
            # Create indices for common query patterns
            with self.sql_engine.connect() as conn:
                # Create index on timestamp columns if they exist
                timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                for col in timestamp_cols:
                    try:
                        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{col} ON {table_name}({col})"))
                    except:
                        pass
                
                # Create index on common energy-related columns
                energy_cols = [col for col in df.columns if any(term in col.lower() for term in ['energy', 'power', 'generation', 'demand', 'price'])]
                for col in energy_cols[:3]:  # Limit to first 3 to avoid too many indices
                    try:
                        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{col} ON {table_name}({col})"))
                    except:
                        pass
                
                conn.commit()
            
            return {
                "success": True,
                "table_name": table_name,
                "rows_indexed": len(df),
                "columns": list(df.columns),
                "indices_created": len(timestamp_cols) + min(len(energy_cols), 3),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"SQL index creation failed: {str(e)}"}
    
    def search_vector_db(self, query: str, collection_name: str = "default", 
                        n_results: int = 5, filters: Dict = None) -> Dict[str, Any]:
        """
        Search vector database using semantic similarity.
        
        Args:
            query: Search query text
            collection_name: Collection to search in
            n_results: Number of results to return
            filters: Optional metadata filters
        
        Returns:
            Search results with similarity scores
        """
        try:
            self.logger.info(f"Searching vector DB: {query[:50]}...")
            
            # Get collection
            if collection_name not in self.collections:
                try:
                    collection = self.chroma_client.get_collection(collection_name)
                    self.collections[collection_name] = collection
                except:
                    return {"error": f"Collection {collection_name} not found"}
            else:
                collection = self.collections[collection_name]
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])[0]
            
            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "id": results['ids'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
            
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "collection": collection_name,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            return {"error": f"Vector search failed: {str(e)}"}
    
    def search_sql_db(self, query: str, table_name: str = None, limit: int = 100) -> Dict[str, Any]:
        """
        Search SQL database using structured queries.
        
        Args:
            query: SQL query or natural language query
            table_name: Specific table to search (optional)
            limit: Maximum number of results
        
        Returns:
            Query results
        """
        try:
            self.logger.info(f"Searching SQL DB: {query[:50]}...")
            
            # If it's a natural language query, convert to SQL
            if not query.strip().upper().startswith(('SELECT', 'WITH')):
                sql_query = self._natural_language_to_sql(query, table_name)
            else:
                sql_query = query
            
            # Add limit if not present
            if 'LIMIT' not in sql_query.upper():
                sql_query += f" LIMIT {limit}"
            
            # Execute query
            with self.sql_engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                columns = list(result.keys())
            
            # Convert to list of dictionaries
            data = [dict(zip(columns, row)) for row in rows]
            
            return {
                "success": True,
                "query": query,
                "sql_query": sql_query,
                "results": data,
                "columns": columns,
                "total_results": len(data)
            }
            
        except Exception as e:
            return {"error": f"SQL search failed: {str(e)}"}
    
    def hybrid_search(self, query: str, vector_weight: float = 0.7, 
                     sql_weight: float = 0.3, n_results: int = 10) -> Dict[str, Any]:
        """
        Perform hybrid search combining vector and SQL results.
        
        Args:
            query: Search query
            vector_weight: Weight for vector search results
            sql_weight: Weight for SQL search results
            n_results: Total number of results to return
        
        Returns:
            Combined search results
        """
        try:
            self.logger.info(f"Performing hybrid search: {query[:50]}...")
            
            # Perform vector search
            vector_results = self.search_vector_db(query, n_results=n_results)
            
            # Perform SQL search
            sql_results = self.search_sql_db(query, limit=n_results)
            
            # Combine and rank results
            combined_results = []
            
            # Add vector results with weighted scores
            if vector_results.get("success"):
                for result in vector_results["results"]:
                    score = (1 - result.get("distance", 1)) * vector_weight
                    combined_results.append({
                        "content": result["document"],
                        "metadata": result["metadata"],
                        "source": "vector",
                        "score": score,
                        "id": result["id"]
                    })
            
            # Add SQL results with weighted scores
            if sql_results.get("success"):
                for i, result in enumerate(sql_results["results"]):
                    # Simple scoring based on position (could be improved)
                    score = (1 - i / len(sql_results["results"])) * sql_weight
                    combined_results.append({
                        "content": str(result),
                        "metadata": {"table_data": result},
                        "source": "sql",
                        "score": score,
                        "id": f"sql_result_{i}"
                    })
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "success": True,
                "query": query,
                "results": combined_results[:n_results],
                "vector_results_count": len(vector_results.get("results", [])),
                "sql_results_count": len(sql_results.get("results", [])),
                "total_results": len(combined_results[:n_results])
            }
            
        except Exception as e:
            return {"error": f"Hybrid search failed: {str(e)}"}
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            # Return dummy embeddings as fallback
            return [[0.0] * 1536 for _ in texts]
    
    def _natural_language_to_sql(self, query: str, table_name: str = None) -> str:
        """
        Convert natural language query to SQL (simplified implementation).
        In production, this would use a more sophisticated NL2SQL model.
        """
        # Get available tables
        with self.sql_engine.connect() as conn:
            tables_result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            available_tables = [row[0] for row in tables_result.fetchall()]
        
        if not available_tables:
            raise ValueError("No tables available for querying")
        
        # Use the specified table or the first available one
        target_table = table_name if table_name in available_tables else available_tables[0]
        
        # Simple keyword-based SQL generation
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['generation', 'generate', 'power']):
            return f"SELECT * FROM {target_table} WHERE LOWER(CAST(column_name AS TEXT)) LIKE '%generation%' OR LOWER(CAST(column_name AS TEXT)) LIKE '%power%'"
        elif any(word in query_lower for word in ['demand', 'load', 'consumption']):
            return f"SELECT * FROM {target_table} WHERE LOWER(CAST(column_name AS TEXT)) LIKE '%demand%' OR LOWER(CAST(column_name AS TEXT)) LIKE '%load%'"
        elif any(word in query_lower for word in ['price', 'cost', 'tariff']):
            return f"SELECT * FROM {target_table} WHERE LOWER(CAST(column_name AS TEXT)) LIKE '%price%' OR LOWER(CAST(column_name AS TEXT)) LIKE '%cost%'"
        else:
            # Default: return all data from the table
            return f"SELECT * FROM {target_table}"
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process RAG indexing and search requests.
        
        Args:
            input_data: Dictionary containing operation type and parameters
            
        Returns:
            Dictionary with operation results
        """
        try:
            self.log_activity("Starting RAG indexing/search operation", input_data)
            
            operation = input_data.get("operation")
            
            if operation == "create_vector_index":
                return self.create_vector_index(input_data.get("documents"), input_data.get("collection_name", "default"))
            elif operation == "create_sql_index":
                return self.create_sql_index(input_data.get("data"), input_data.get("table_name"), input_data.get("schema"))
            elif operation == "search_vector_db":
                return self.search_vector_db(input_data.get("query"), input_data.get("collection_name", "default"), input_data.get("n_results", 5), input_data.get("filters"))
            elif operation == "search_sql_db":
                return self.search_sql_db(input_data.get("query"), input_data.get("table_name"), input_data.get("limit", 100))
            elif operation == "hybrid_search":
                return self.hybrid_search(input_data.get("query"), input_data.get("vector_weight", 0.7), input_data.get("sql_weight", 0.3), input_data.get("n_results", 10))
            else:
                return self.create_error_response(f"Unsupported RAG operation: {operation}")
            
        except Exception as e:
            self.logger.error(f"Error in RAG operation: {str(e)}")
            return self.create_error_response(f"RAG operation failed: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in RAG operation: {str(e)}")
            return self.create_error_response(f"RAG operation failed: {str(e)}")
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get statistics about current indices."""
        try:
            stats = {
                "vector_collections": {},
                "sql_tables": {},
                "total_documents": 0,
                "total_rows": 0
            }
            
            # Vector database stats
            for name, collection in self.collections.items():
                count = collection.count()
                stats["vector_collections"][name] = {
                    "document_count": count,
                    "metadata": collection.metadata
                }
                stats["total_documents"] += count
            
            # SQL database stats
            with self.sql_engine.connect() as conn:
                tables_result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [row[0] for row in tables_result.fetchall()]
                
                for table in tables:
                    try:
                        count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = count_result.fetchone()[0]
                        stats["sql_tables"][table] = {"row_count": count}
                        stats["total_rows"] += count
                    except:
                        stats["sql_tables"][table] = {"row_count": "unknown"}
            
            return stats
            
        except Exception as e:
            return {"error": f"Failed to get index statistics: {str(e)}"}

