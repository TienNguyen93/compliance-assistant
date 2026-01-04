"""
Configuration settings for the compliance assistant
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Vector store
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
COLLECTION_NAME = "compliance_docs"

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters

# Retrieval settings
TOP_K_RESULTS = 5

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORSTORE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)