"""Central configuration for the RAG chatbot.

Most values can be overridden via environment variables to simplify
switching between CPU/GPU hosts and different model sizes.
"""

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# URL where Ollama server is running.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Global GPU toggle used by FAISS/reranker defaults.
USE_GPU = _env_bool("USE_GPU", True)

# FAISS runtime options.
FAISS_USE_GPU = _env_bool("FAISS_USE_GPU", USE_GPU)
FAISS_GPU_DEVICE = _env_int("FAISS_GPU_DEVICE", 0)

# Optional Ollama runtime options. Keep None to use Ollama defaults.
OLLAMA_NUM_GPU = os.getenv("OLLAMA_NUM_GPU")
OLLAMA_NUM_THREAD = os.getenv("OLLAMA_NUM_THREAD")
OLLAMA_NUM_CTX = _env_int("OLLAMA_NUM_CTX", 4096)


def build_ollama_options() -> dict:
    """Build optional Ollama options payload from environment settings."""
    options = {"num_ctx": OLLAMA_NUM_CTX}
    if OLLAMA_NUM_GPU is not None:
        try:
            options["num_gpu"] = int(OLLAMA_NUM_GPU)
        except ValueError:
            pass
    if OLLAMA_NUM_THREAD is not None:
        try:
            options["num_thread"] = int(OLLAMA_NUM_THREAD)
        except ValueError:
            pass
    return options


# Model defaults (GPU-friendly, can be overridden via env vars).
CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral")
QUERY_DECOMPOSER_MODEL = os.getenv("QUERY_DECOMPOSER_MODEL", CHAT_MODEL)
SERPAPI_API_KEY = "4dd0a5c8d6d997fc4236408ee07714848a07f99eff65c4d9c61c9da0caf83b1f"
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")

# Cross-encoder reranker defaults.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda" if USE_GPU else "cpu")
RERANKER_BATCH_SIZE = _env_int("RERANKER_BATCH_SIZE", 16)
RERANKER_MAX_LENGTH = _env_int("RERANKER_MAX_LENGTH", 512)

# Chunking and retrieval configuration.
CHUNK_SIZE = _env_int("CHUNK_SIZE", 400)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 75)
INITIAL_RETRIEVAL = _env_int("INITIAL_RETRIEVAL", 20)
TOP_K_FINAL = _env_int("TOP_K_FINAL", 4)

# Must match the embedding model dimension (mxbai-embed-large -> 1024).
VECTOR_DIM = _env_int("VECTOR_DIM", 1024)

# Weight for semantic vs lexical signals in hybrid retrieval (0.0-1.0).
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))

# Backward compatibility alias used by tools/hybrid_demo.py.
TOP_K_RETRIEVAL = INITIAL_RETRIEVAL
