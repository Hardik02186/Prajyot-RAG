# RAG Chatbot (rag_chatbot)

This repository implements a small Retrieval-Augmented Generation (RAG)
pipeline that indexes PDF documents, runs hybrid retrieval (semantic +
lexical) and answers questions with an LLM via Ollama.

Overview (step-by-step)
1. Startup (run `main.py` via `uvicorn main:app --reload`)
   - The FastAPI app checks Ollama availability.
   - If Ollama is reachable, the app scans `data/documents/` for PDFs.
   - New or modified PDFs are processed: text extraction -> chunking -> embeddings -> stored.

2. Document processing
   - `services/document_service.py` extracts text pages from PDFs. If a
     PDF looks scanned, it falls back to OCR using `services/ocr_service.py`.
   - `services/chunk_service.py` splits pages into overlapping chunks and
     attaches metadata (`source`, `page`, `document_id`, `chunk_id`).

3. Indexing & storage
   - `core/vector_store.py` maintains a FAISS index for vectors and a
     compact inverted index (token -> set(chunk_indices)) to support
     hybrid retrieval. Metadata for each chunk is persisted alongside the index.

4. Retrieval
   - `services/retreival_service.py` embeds queries, performs semantic
     retrieval from the persistent FAISS index, runs BM25 lexical retrieval,
     fuses both rankings with reciprocal-rank fusion (RRF), then reranks
     with `services/reranker_service.py` (cross-encoder).

5. Response generation
   - `services/chat_service.py` composes a context from top chunks and
     calls the LLM via `core/ollama_client.py` to produce a final answer.

Useful scripts
- `tools/hybrid_demo.py` — run a sample hybrid retrieval query from CLI.

How to run locally (Windows PowerShell)
```powershell
cd rag_chatbot
& C:/Users/hardi/Desktop/rag-project/venv/Scripts/Activate.ps1
uvicorn main:app --reload
```

API endpoints
- POST `/chat` with JSON `{ "question": "..." }` — full retrieval + generation
- POST `/chat/fast` — faster flow using vector search only
- GET  `/stats` — vector store stats and file tracker
- POST `/reindex?force=true` — clear cache so on restart all PDFs are reindexed

GPU configuration
- The app now supports GPU-first defaults for FAISS and reranking with CPU fallback.
- Recommended environment variables:
  - `USE_GPU=true`
  - `FAISS_USE_GPU=true`
  - `RERANKER_DEVICE=cuda`
  - `CHAT_MODEL=qwen2.5:14b-instruct`
  - `EMBED_MODEL=nomic-embed-text`
- Optional Ollama runtime tuning:
  - `OLLAMA_NUM_GPU` (number of layers to offload)
  - `OLLAMA_NUM_THREAD`
  - `OLLAMA_NUM_CTX` (default: 4096)

Install notes
- `requirements.txt` uses platform markers:
  - Windows: `faiss-cpu`
  - non-Windows: `faiss-gpu-cu12`
- If you run on Linux with CUDA, install with:
  - `pip install -r requirements.txt`

Model pull examples (Ollama)
- `ollama pull nomic-embed-text`
- `ollama pull qwen2.5:14b-instruct`

Important reindex note
- If you change `EMBED_MODEL` or `VECTOR_DIM`, clear/rebuild embeddings:
  - `POST /reindex?force=true`

Notes & next improvements
- Consider replacing prints with Python `logging` to control verbosity
- For large-scale document collections, persist the inverted index in a
  lightweight DB (SQLite) or use a dedicated retrieval engine (BM25)
  for better scaling.
