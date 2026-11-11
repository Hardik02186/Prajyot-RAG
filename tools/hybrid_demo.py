"""Small CLI demo to exercise the hybrid retrieval pipeline.

Usage:
  python tools/hybrid_demo.py --query "your question"

The script embeds the query, runs `vector_store.hybrid_search` and prints
the top candidates along with semantic and lexical component scores.
"""

# Import necessary libraries
import sys  # System operations
import os  # File operations
import argparse  # For command line arguments

# Ensure project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required services
from services.embedding_services import embed_text  # Function to embed text
from services.retreival_service import vector_store  # Global vector store instance
from config import TOP_K_RETRIEVAL, HYBRID_ALPHA  # Default configuration values


def run_demo(query: str, k: int = TOP_K_RETRIEVAL, alpha: float = HYBRID_ALPHA):
    """Run a single query and pretty-print hybrid scores and metadata."""
    if vector_store.index.ntotal == 0:  # Check if any documents are indexed
        print("Vector store is empty. Index documents first.")  # Error message
        return  # Exit function

    emb = embed_text(query)  # Create embedding for the query
    results = vector_store.hybrid_search(emb, query, k=k, alpha=alpha)  # Perform hybrid search

    if not results:  # If no results found
        print("No results returned.")  # Message
        return  # Exit

    print(f"Top {len(results)} results (alpha={alpha}):\n")  # Header with parameters
    for rank, r in enumerate(results, start=1):  # For each result
        score = r.get("_score")  # Combined score
        v = r.get("_vector_score")  # Semantic score
        l = r.get("_lexical_score")  # Lexical score
        source = r.get("source")  # Source document
        page = r.get("page")  # Page number
        cid = r.get("chunk_id", "n/a")  # Chunk ID
        text_snip = (r.get("text") or "")[:300].replace("\n", " ")  # First 300 chars of text
        print(f"{rank}. score={score:.4f} (vec={v:.4f} lex={l:.4f}) source={source} page={page} chunk={cid}")  # Print result info
        print(f"   {text_snip}...\n")  # Print text snippet


def main():
    """Main function to parse arguments and run demo."""
    p = argparse.ArgumentParser(description="Hybrid retrieval demo (semantic + lexical)")  # Create argument parser
    p.add_argument("--query", "-q", required=True, help="Query text")  # Required query argument
    p.add_argument("--k", type=int, default=TOP_K_RETRIEVAL, help="Number of candidates")  # Optional k parameter
    p.add_argument("--alpha", type=float, default=HYBRID_ALPHA, help="Semantic weight (0..1)")  # Optional alpha parameter
    args = p.parse_args()  # Parse command line arguments

    run_demo(args.query, k=args.k, alpha=args.alpha)  # Run the demo with arguments


if __name__ == "__main__":
    main()  # Run main function when script is executed directly
