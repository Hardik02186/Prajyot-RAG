"""Lightweight reranker using a Cross-Encoder model.

This module exposes `rerank(query, documents)` which scores text
pairs `(query, document_text)` and returns the top `TOP_K_FINAL`
documents ordered by relevance.
"""

# Import necessary libraries
import sys  # System operations
import os  # File operations

# Ensure proper imports work when modules are executed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import cross-encoder for reranking
from sentence_transformers import CrossEncoder  # Cross-encoder model
from config import (
    TOP_K_FINAL,
    RERANKER_MODEL,
    RERANKER_DEVICE,
    RERANKER_BATCH_SIZE,
    RERANKER_MAX_LENGTH,
)  # Reranker configuration


def _load_reranker():
    """Load cross-encoder on the preferred device with CPU fallback."""
    preferred_device = RERANKER_DEVICE
    try:
        return CrossEncoder(
            RERANKER_MODEL,
            device=preferred_device,
            max_length=RERANKER_MAX_LENGTH,
        )
    except Exception:
        if preferred_device != "cpu":
            try:
                return CrossEncoder(
                    RERANKER_MODEL,
                    device="cpu",
                    max_length=RERANKER_MAX_LENGTH,
                )
            except Exception:
                return None
        return None


# Instantiate the cross-encoder used for fine-grained reranking
reranker_model = _load_reranker()


def rerank(query, documents):
    """Return the top `TOP_K_FINAL` documents after cross-encoder scoring."""
    if not documents:
        return []
    if reranker_model is None:
        return documents[:TOP_K_FINAL]

    pairs = [(query, doc["text"]) for doc in documents]  # Create query-document pairs
    scores = reranker_model.predict(
        pairs,
        batch_size=RERANKER_BATCH_SIZE,
        show_progress_bar=False,
    )  # Score each pair with cross-encoder
    scored_docs = list(zip(documents, scores))  # Pair documents with scores
    scored_docs.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
    return [doc for doc, _ in scored_docs[:TOP_K_FINAL]]  # Return top documents
