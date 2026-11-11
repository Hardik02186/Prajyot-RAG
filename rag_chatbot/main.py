"""
Main FastAPI application for the RAG chatbot.

This module wires together the document ingestion, retrieval and chat
endpoints. On startup it incrementally indexes PDF files from
`data/documents/` (if Ollama is available). It exposes endpoints for
chatting, fast chat, stats and a simple debug timing endpoint.

Endpoints:
- POST /chat       -> full retrieval + generation
- POST /chat/fast  -> fast (no rerank) retrieval + generation
- GET  /stats      -> vector store statistics
- POST /reindex    -> clear cache to force reindex on restart
"""

# Import necessary libraries
from fastapi import FastAPI, HTTPException  # FastAPI for building the web API
from pydantic import BaseModel  # For defining data models
import os  # For file system operations
import re  # Regular expressions for answer inspection
import sys  # For system path manipulation
import time  # For timing operations
import threading  # For running background tasks
import logging
import ollama
# Add the current directory to Python path so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from our services
from services.document_service import process_document  # To extract text from PDFs
from services.chunk_service import chunk_pages  # To split text into smaller pieces
from services.retreival_service import add_documents, vector_store, retrieve, retrieve_no_rerank  # For storing and searching documents
from services.chat_service import generate_answer  # To generate answers using AI
from services.query_decomposer import get_subqueries  # For optional query decomposition
from services.web_search_service import search_web  # Fallback web search for missing data
from services.embedding_services import is_ollama_available  # To check if AI service is running
from services.reranker_service import rerank  # For fine-grained reranking of results

# Create the FastAPI application instance
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the folder where PDF documents are stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this file
PRIMARY_DATA_FOLDER = os.path.join(BASE_DIR, "data", "documents")
LEGACY_DATA_FOLDER = os.path.join(os.path.dirname(BASE_DIR), "data", "documents")
DATA_FOLDER = PRIMARY_DATA_FOLDER if os.path.exists(PRIMARY_DATA_FOLDER) else LEGACY_DATA_FOLDER

# Define the structure of chat requests
class ChatRequest(BaseModel):
    question: str  # The user's question


def index_documents():
    """Index new or modified PDF documents from the data folder."""
    # Check if the AI service (Ollama) is available
    if not is_ollama_available():
        print("⚠️ Ollama is not running at http://localhost:11434")
        return

    # Check if the documents folder exists
    if not os.path.exists(DATA_FOLDER):
        logger.warning("Documents folder not found: %s", DATA_FOLDER)
        return

    try:
        # Get list of all PDF files in the folder
        pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logger.info("No PDF files found in documents folder: %s", DATA_FOLDER)
        tracked_files = set(vector_store.file_tracker.keys())
        current_files = set(pdf_files)
        deleted_files = tracked_files - current_files
        reindex_all = False

        # If indexed files were deleted from disk, rebuild so stale embeddings are removed.
        if deleted_files:
            print(f"Deleted files detected. Rebuilding index to remove stale chunks: {sorted(deleted_files)}")
            vector_store.reset_store()
            reindex_all = True

        new_files = 0  # Counter for newly indexed files
        for filename in pdf_files:
            path = os.path.join(DATA_FOLDER, filename)  # Full path to the PDF
            if not reindex_all and vector_store.is_file_processed(path):  # Skip if already processed
                continue
            try:
                pages = process_document(path)  # Extract text from PDF
                if pages:  # If extraction succeeded
                    chunks = chunk_pages(pages, filename)  # Split into chunks
                    if chunks:  # If chunking succeeded
                        add_documents(chunks, source_file=path)  # Add to vector store
                        new_files += 1
            except Exception as e:  # Skip files that fail to process
                logger.exception("Failed to index file '%s': %s", filename, e)
                continue
        if new_files > 0:
            print(f"Indexed {new_files} new document(s)")
    except Exception:
        pass  # Ignore any errors in the indexing process


def background_indexer():
    """Background thread to periodically check for new documents."""
    while True:  # Run forever
        time.sleep(60)  # Wait 60 seconds
        index_documents()  # Check and index new documents


@app.on_event("startup")
def startup_event():
    """Function that runs when the application starts."""
    index_documents()  # Index documents on startup
    # Start a background thread for periodic indexing
    threading.Thread(target=background_indexer, daemon=True).start()


# ── Set this to whatever model you use in chat_service.py ──────────────────
OLLAMA_MODEL = "llama3.1"  # e.g. "mistral", "llama3.1:8b", "gemma2" etc.


def extract_final_answer(verbose_answer: str, question: str) -> str:
    """Post-process a verbose answer into the shortest correct phrase."""
    # Don't run extraction on controlled fallback strings or IDKs
    if "I don't know" in verbose_answer or "could not find" in verbose_answer:
        return verbose_answer

    extraction_prompt = (
        f'Question: "{question}"\n'
        f'Verbose answer: "{verbose_answer}"\n\n'
        "Determine what TYPE of thing the question asks for, then extract only that:\n"
        "- 'which magazine/film/team' -> the magazine/film/team NAME (not a date or person)\n"
        "- 'what character did X play' -> the CHARACTER name (not the actor)\n"
        "- 'what party/profession/industry' -> the party/profession/industry (not a person)\n"
        "- 'where/which county/which country' -> place at EXACT level asked (county not state)\n"
        "- 'yes/no question' -> only 'yes' or 'no'\n"
        "- 'who' -> full name as given\n"
        "- 'when/what year' -> only the year or date\n"
        "Output ONLY the minimal answer. No explanation. No punctuation outside the answer.\n"
        "Minimal answer:"
    )
    try:
        response = ollama.generate(model=OLLAMA_MODEL, prompt=extraction_prompt)
        return response["response"].strip()
    except Exception as exc:
        logger.warning("extract_final_answer failed, returning original: %s", exc)
        return verbose_answer


def _should_use_web_search(answer: str) -> bool:
    patterns = [
        r"information is unavailable",
        r"not enough information",
        r"missing details",
        r"cannot answer",
        r"do not have enough",
        r"insufficient information",
        r"does not contain",
        r"did not contain",
        r"no relevant information",
        r"no information .* contain",
        r"information .* not .* available",
    ]
    lower = answer.lower()
    return any(re.search(pattern, lower) for pattern in patterns)
    return result


@app.post("/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint that retrieves documents and generates answers."""
    start = time.time()  # Record start time for performance measurement
    try:
        subqueries = get_subqueries(request.question)
        was_decomposed = len(subqueries) > 1

        chunks = []
        seen_keys = set()
        # Per-subquery retrieval results, kept separate so the generator can see
        # which evidence corresponds to which sub-question (only meaningful when
        # the query was actually decomposed into >1 subquery).
        subquery_results = []

        # 1. Retrieve without reranking for each subquery
        for subquery in subqueries:
            sub_chunks = retrieve_no_rerank(subquery)
            subquery_results.append({"subquery": subquery, "chunks": sub_chunks})
            for chunk in sub_chunks:
                key = (chunk.get("source"), 
                       chunk.get("page"), 
                       chunk.get("chunk_id"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    chunks.append(chunk)

        # 2. Rerank once across all pooled n*20 chunks
        if len(chunks) > 1:
            chunks = rerank(request.question, chunks)  # use original question, not subqueries

        # Restrict each subquery's chunk list to the chunks that survived reranking,
        # so the generator's grouped context matches the same (reranked) evidence
        # set used for the flat fallback / sources list.
        if was_decomposed and chunks:
            kept_keys = {
                (c.get("source"), c.get("page"), c.get("chunk_id")) for c in chunks
            }
            for item in subquery_results:
                item["chunks"] = [
                    c for c in item["chunks"]
                    if (c.get("source"), c.get("page"), c.get("chunk_id")) in kept_keys
                ]

        decomposition_context = subquery_results if was_decomposed else None

        web_results = None
        web_search_used = False
        answer = None

        if chunks:
            answer = generate_answer(request.question, chunks, subquery_results=decomposition_context)
            if _should_use_web_search(answer):  # check BEFORE extraction (verbose form is more detectable)
                try:
                    web_results = search_web(request.question)
                    if web_results:
                        answer = generate_answer(
                            request.question, chunks, web_results=web_results, subquery_results=decomposition_context
                        )
                        answer = extract_final_answer(answer, request.question)
                        web_search_used = True
                    else:
                        answer = extract_final_answer(answer, request.question)
                except Exception as exc:
                    logger.warning("Web search fallback failed: %s", exc)
                    answer = extract_final_answer(answer, request.question)
            else:
                answer = extract_final_answer(answer, request.question)
        else:
            try:
                web_results = search_web(request.question)
                if web_results:
                    answer = generate_answer(request.question, [], web_results=web_results)
                    answer = extract_final_answer(answer, request.question)
                    web_search_used = True
                else:
                    answer = "I could not find any relevant information in the documents to answer your question."
            except Exception as exc:
                logger.warning("Web search fallback failed: %s", exc)
                answer = "I could not find any relevant information in the documents to answer your question."

        total_time = time.time() - start  # Calculate total time taken
        response = {
            "answer": answer,
            "sources": [{"source": c["source"], "page": c["page"]} for c in chunks],
            "response_time_seconds": round(total_time, 2),
        }
        if was_decomposed:
            response["subqueries"] = subqueries
            response["subquery_retrieval"] = [
                {"subquery": item["subquery"], "chunk_count": len(item["chunks"])}
                for item in subquery_results
            ]
        if web_search_used:
            response["web_search_used"] = True
            response["web_results"] = web_results

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Return error if something goes wrong


@app.get("/health")
def health_check():
    """Simple endpoint to check if the service is running."""
    return {"status": "ok"}  # Always returns OK if the server is up


@app.get("/stats")
def get_stats():
    """Get statistics about the vector store and processed documents."""
    stats = vector_store.get_stats()  # Get stats from the vector store
    return {
        "total_vectors": stats['total_vectors'],  # Number of vectors stored
        "total_chunks": stats['total_chunks'],  # Number of text chunks
        "processed_files": stats['processed_files'],  # Number of files processed
        "file_tracker": vector_store.file_tracker  # Details of processed files
    }


@app.post("/reindex")
def reindex_documents(force: bool = False):
    """Manually trigger reindexing of documents."""
    if force:  # If force is true, clear all cached data
        vector_store.reset_store()  # Clear index, metadata and tracker
        index_documents()  # Reindex all documents
        return {"message": "Cache cleared and all documents reindexed.", "force": True}
    else:  # Otherwise, just index new documents
        index_documents()  # Index any new documents
        return {"message": "Incremental indexing completed. New documents have been indexed.", "force": False}


@app.post("/chat/fast")
async def chat_fast(request: ChatRequest):
    """Fast chat endpoint that skips reranking for quicker responses."""
    start = time.time()  # Start timing
    try:
        if vector_store.index.ntotal == 0:  # Check if any documents are indexed
            return {"answer": "Vector store is empty. No documents indexed.", "sources": [], "response_time_seconds": 0}

        from services.embedding_services import embed_text  # Import embedding function
        from config import TOP_K_FINAL  # Import config value

        query_embedding = embed_text(request.question)  # Convert question to vector
        chunks = vector_store.search(query_embedding, TOP_K_FINAL)  # Search for similar chunks

        if not chunks:  # If no chunks found
            return {"answer": "No relevant documents found.", "sources": [], "response_time_seconds": round(time.time() - start, 2)}

        answer = generate_answer(request.question, chunks)  # Generate answer
        total_time = time.time() - start  # Calculate time
        return {"answer": answer, "sources": [{"source": c["source"], "page": c["page"]} for c in chunks], "response_time_seconds": round(total_time, 2), "mode": "fast"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Handle errors


@app.post("/chat/diagnose")
async def chat_diagnose(request: ChatRequest):
    """Compare normal and fast retrieval to debug recall issues quickly."""
    try:
        from services.embedding_services import embed_text
        from config import TOP_K_FINAL

        normal_chunks = retrieve(request.question)
        query_embedding = embed_text(request.question)
        fast_chunks = vector_store.search(query_embedding, TOP_K_FINAL)

        return {
            "question": request.question,
            "normal_retrieval_count": len(normal_chunks),
            "normal_retrieval_preview": [
                {"source": c.get("source"), "page": c.get("page"), "text_preview": c.get("text", "")[:200]}
                for c in normal_chunks
            ],
            "fast_retrieval_count": len(fast_chunks),
            "fast_retrieval_preview": [
                {"source": c.get("source"), "page": c.get("page"), "text_preview": c.get("text", "")[:200]}
                for c in fast_chunks
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/timing")
def debug_timing():
    """Debug endpoint to measure performance of different operations."""
    import time
    from services.embedding_services import embed_text

    results = {}  # Dictionary to store timing results
    try:
        start = time.time()  # Time embedding creation
        test_embedding = embed_text("Test query for timing")
        results["embedding_time_seconds"] = round(time.time() - start, 2)

        if vector_store.index.ntotal > 0:  # If documents are indexed
            start = time.time()  # Time vector search
            search_results = vector_store.search(test_embedding, 5)
            results["search_time_seconds"] = round(time.time() - start, 2)

            if len(search_results) > 1:  # If multiple results
                start = time.time()  # Time reranking
                _ = rerank("Test query", search_results)
                results["reranking_time_seconds"] = round(time.time() - start, 2)

        results["status"] = "ok"  # Indicate success
    except Exception as e:
        results["error"] = str(e)  # Record any errors
    return results