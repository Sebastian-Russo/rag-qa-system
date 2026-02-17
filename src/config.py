"""
Shared configuration for the RAG system
All settings and API config in one place
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval
DEFAULT_TOP_K = 10
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# Paths
EMBEDDINGS_PATH = "data/vectorstore/embeddings.npy"
CHUNKS_PATH = "data/vectorstore/chunks.pkl"
