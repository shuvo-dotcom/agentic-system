"""
Configuration settings for the Agentic System.
"""
import os

# OpenAI API Configuration
OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4o"
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./temp_data.db")  # Temporary SQLite database
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./temp_vector_db") # Temporary vector database

# Plexos Configuration
PLEXOS_API_URL = os.getenv("PLEXOS_API_URL", "")
PLEXOS_API_KEY = os.getenv("PLEXOS_API_KEY", "")
PLEXOS_DB_PATH = os.getenv("PLEXOS_DB_PATH", "/home/ubuntu/agentic_system/temp_plexos_data") # Temporary path for Plexos data

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Agent Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TIMEOUT = int(os.getenv("TIMEOUT", "30"))

DELETE_LOG_FILE_AFTER_UPLOAD = True

LANGCHAIN_API_KEY="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"

# Langfuse configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
