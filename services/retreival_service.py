# Import necessary libraries
import logging
import sys  # System operations
import os  # File operations

# Ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Retrieval helpers: batch embedding and hybrid retrieval pipeline.

This module wraps the `VectorStore` and provides two functions used by the
application:

- `add_documents(chunks, source_file)`: embed a list of chunks and add them
    to the FAISS index and inverted index. Marks the source file as processed.
- `retrieve(query)`: run hybrid search (semantic + lexical) and optionally
    rerank the top candidates.
"""

# Import required services and config
from services.embedding_services import embed_text  # Function to create embeddings (used when adding documents)
from services.reranker_service import rerank  # Function to rerank results
from core.vector_store import VectorStore  # Vector storage class
from config import INITIAL_RETRIEVAL  # Retrieval configuration

# langchain components used for hybrid retrieval
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# Create global vector store instance
vector_store = VectorStore()
logger = logging.getLogger(__name__)


def _to_chunk_dict(doc: Document):
    """Normalize LangChain Document into the app's expected chunk schema."""
    md = doc.metadata or {}
    return {
        "text": doc.page_content,
        "source": md.get("source"),
        "page": md.get("page"),
        "document_id": md.get("document_id"),
        "chunk_id": md.get("chunk_id"),
        "metadata": md,
    }


def reciprocal_rank_fusion(results_list, k=60):
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF assigns each document a score based on its rank in each list:
    score = sum(1 / (k + rank)) across all retrievers

    Args:
        results_list: List of lists of Document objects from different retrievers
        k: Constant (typically 60) to avoid division by very small ranks

    Returns:
        List of unique documents sorted by RRF score (highest first)
    """
    # Dictionary to accumulate scores: doc_id -> (document, score)
    doc_scores = {}
    
    # For each retriever's results
    for results in results_list:
        # Assign RRF score based on rank in this retriever
        for rank, doc in enumerate(results):
            md = doc.metadata or {}
            doc_id = (
                md.get("document_id"),
                md.get("chunk_id"),
                md.get("source"),
                md.get("page"),
            )
            # RRF formula: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank + 1)  # rank + 1 because enumerate starts at 0
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = (doc, 0.0)  # (document, accumulated_score)
            
            # Accumulate score
            old_doc, old_score = doc_scores[doc_id]
            doc_scores[doc_id] = (old_doc, old_score + rrf_score)
    
    # Sort by score descending
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
    
    # Return just the documents (without scores)
    return [doc for doc, _ in sorted_docs]


def add_documents(chunks, source_file=None):
    """Embed and add a list of chunk dicts to the vector store.

    `chunks` must be a list of dicts containing a `text` key. The function
    batches embedding requests to avoid large single calls.
    """
    if not chunks:  # If no chunks to add
        return  # Do nothing
    
    try:
        embeddings = []  # List to store embeddings
        
        # Process in batches for better performance
        batch_size = 10  # Process 10 chunks at a time
        for i in range(0, len(chunks), batch_size):  # For each batch
            batch = chunks[i:i+batch_size]  # Get batch of chunks
            batch_embeddings = [embed_text(chunk["text"]) for chunk in batch]  # Embed each chunk
            embeddings.extend(batch_embeddings)  # Add to embeddings list
        vector_store.add(embeddings, chunks)  # Add to vector store
        if source_file:  # If source file provided
            vector_store.mark_file_processed(source_file, len(chunks))  # Mark as processed
    except Exception as e:  # If error
        raise  # Re-raise exception


def retrieve_no_rerank(query):
    """Hybrid retrieval (BM25 + semantic + RRF) without reranking."""
    if vector_store.index.ntotal == 0:
        return []

    try:
        query_embedding = embed_text(query)
        semantic_chunks = vector_store.search(query_embedding, INITIAL_RETRIEVAL)
        semantic_docs = [
            Document(page_content=chunk["text"], metadata=chunk) for chunk in semantic_chunks
        ]

        docs = [
            Document(page_content=md["text"], metadata=md)
            for md in vector_store.metadata
        ]

        bm25 = BM25Retriever.from_documents(docs, k=INITIAL_RETRIEVAL)
        bm25_results = bm25.invoke(query)
        fused_results = reciprocal_rank_fusion([semantic_docs, bm25_results], k=60)
        return [_to_chunk_dict(d) for d in fused_results[:INITIAL_RETRIEVAL]]

    except Exception:
        logger.exception("Retrieval failed for query: %s", query)
        return []


def retrieve(query):
    """Full pipeline: hybrid retrieval + reranking."""
    initial_results = retrieve_no_rerank(query)
    if len(initial_results) > 0:
        return rerank(query, initial_results)
    return initial_results
