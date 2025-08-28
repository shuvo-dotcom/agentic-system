"""
Configuration settings for the Agentic System.
"""
import os

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")  # Get from environment or .env file
OPENAI_MODEL = "gpt-4-turbo"
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID", "YOUR_PROJECT_ID")
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
