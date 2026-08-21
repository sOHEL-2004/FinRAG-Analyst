import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")  # Required for LlamaParse
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional if running locally

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# PDR (Parent Document Retrieval) Settings
CHILD_CHUNK_TOKENS = 500
PARENT_CHUNK_TOKENS = 2500
CHUNK_OVERLAP = 50  # 10% overlap for the child chunks

# Model Settings
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"