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




