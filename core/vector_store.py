# Import necessary libraries
import faiss  # FAISS library for efficient vector similarity search
import numpy as np  # NumPy for numerical operations
import sys  # System operations
import os  # File system operations
import json  # JSON handling for file tracker
import hashlib  # For computing file hashes
import pickle  # For serializing Python objects
import re  # Regular expressions for text processing
import math  # Mathematical functions

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import VECTOR_DIM, FAISS_USE_GPU, FAISS_GPU_DEVICE  # FAISS/vector config


class VectorStore:
    """FAISS-backed vector store with a simple inverted index.

    Each chunk has metadata stored in `self.metadata` (a list aligned with
    FAISS vector indices). We also maintain a small in-memory inverted index
    (`token -> set(indices)`) plus per-token document frequencies to enable
    hybrid (semantic + lexical) retrieval without an external DB.
    """
    def __init__(self, persist_dir="rag_chatbot/data/vector_store"):
        # Directory where vector store data is saved
        self.persist_dir = persist_dir
        # Paths to different data files
        self.index_path = os.path.join(persist_dir, "index.faiss")  # FAISS index file
        self.metadata_path = os.path.join(persist_dir, "metadata.pkl")  # Metadata pickle file
        self.inverted_path = os.path.join(persist_dir, "inverted.pkl")  # Inverted index file
        self.file_tracker_path = os.path.join(persist_dir, "file_tracker.json")  # File tracking
        
        # Create the directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)

        self.gpu_resources = None
        self.use_gpu = False
        reset_tracker = False
        
        # Load existing data or create new
        if os.path.exists(self.index_path):  # If index exists, load it
            cpu_index = faiss.read_index(self.index_path)  # Load FAISS index
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)  # Load metadata
            # Load inverted index if it exists
            if os.path.exists(self.inverted_path):
                with open(self.inverted_path, 'rb') as f:
                    data = pickle.load(f)
                    self.inverted_index = data.get('inverted_index', {})  # Token to indices mapping
                    self.doc_freq = data.get('doc_freq', {})  # Document frequency per token
            else:
                self.inverted_index = {}  # Empty inverted index
                self.doc_freq = {}  # Empty document frequencies

            # If embedding dimension changed, force a clean index to prevent runtime errors.
            if cpu_index.d != VECTOR_DIM:
                cpu_index = self._build_cpu_index()
                self.metadata = []
                self.inverted_index = {}
                self.doc_freq = {}
                reset_tracker = True
        else:  # Create new empty index
            cpu_index = self._build_cpu_index()  # Create new FAISS index
            self.metadata = []  # Empty metadata list
            self.inverted_index = {}
            self.doc_freq = {}

        self.index = self._attach_runtime_index(cpu_index)
        
        # Load file tracker
        self.file_tracker = self._load_file_tracker()
        if reset_tracker:
            self.file_tracker = {}
            self._save_file_tracker()

    def _build_cpu_index(self):
        """Build the base CPU FAISS index used for persistence."""
        return faiss.IndexFlatL2(VECTOR_DIM)

    def _attach_runtime_index(self, cpu_index):
        """Attach index to GPU when available; otherwise use CPU index."""
        self.use_gpu = False
        if not FAISS_USE_GPU:
            return cpu_index

        try:
            if not hasattr(faiss, "get_num_gpus"):
                return cpu_index
            gpu_count = faiss.get_num_gpus()
            if gpu_count <= FAISS_GPU_DEVICE:
                return cpu_index
            if not hasattr(faiss, "StandardGpuResources"):
                return cpu_index

            self.gpu_resources = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, FAISS_GPU_DEVICE, cpu_index)
            self.use_gpu = True
            return gpu_index
        except Exception:
            return cpu_index

    def _runtime_to_cpu_index(self):
        """Return a CPU FAISS index snapshot for persistence."""
        if self.use_gpu and hasattr(faiss, "index_gpu_to_cpu"):
            return faiss.index_gpu_to_cpu(self.index)
        return self.index

    def _load_file_tracker(self):
        """Load tracking info about processed files"""
        if os.path.exists(self.file_tracker_path):
            with open(self.file_tracker_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_file_tracker(self):
        """Save tracking info about processed files"""
        with open(self.file_tracker_path, 'w') as f:
            json.dump(self.file_tracker, f, indent=2)

    def _tokenize(self, text: str):
        """Basic tokenizer: lowercase, keep alphanumerics, split on non-word chars"""
        if not text:
            return []
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9]{2,}\b", text)
        return tokens

    def _compute_file_hash(self, filepath):
        """Compute SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def is_file_processed(self, filepath):
        """Check if a file has already been processed"""
        if not os.path.exists(filepath):  # If file doesn't exist
            return False  # Can't be processed
        
        filename = os.path.basename(filepath)  # Get just the filename
        try:
            file_hash = self._compute_file_hash(filepath)  # Compute current hash
            
            if filename in self.file_tracker:  # If we've seen this file before
                stored_hash = self.file_tracker[filename].get('hash')  # Get stored hash
                if stored_hash == file_hash:  # If hashes match
                    return True  # File hasn't changed
        except Exception:  # If error computing hash
            pass  # Assume not processed
        
        return False  # File not processed or changed

    def mark_file_processed(self, filepath, chunk_count):
        """Mark a file as processed with its hash"""
        filename = os.path.basename(filepath)  # Get filename
        try:
            file_hash = self._compute_file_hash(filepath)  # Compute hash
            self.file_tracker[filename] = {  # Store file info
                'hash': file_hash,  # File hash
                'chunks': chunk_count,  # Number of chunks created
                'timestamp': str(__import__('datetime').datetime.now())  # When processed
            }
            self._save_file_tracker()  # Save to disk
        except Exception:  # If error
            pass  # Skip marking

    def add(self, embeddings, metadata_list):
        """Add embeddings and metadata to the index"""
        if not embeddings:  # If no embeddings to add
            return  # Do nothing
        
        embeddings_array = np.array(embeddings).astype("float32")  # Convert to numpy array
        # Keep track of where new metadata indices will start
        start_idx = int(self.index.ntotal)  # Current number of vectors
        self.index.add(embeddings_array)  # Add to FAISS index
        self.metadata.extend(metadata_list)  # Add metadata

        # Update inverted index and document-frequency counts
        for i, md in enumerate(metadata_list):  # For each metadata item
            abs_idx = start_idx + i  # Absolute index in the index
            text = md.get('text', '')  # Get the text
            tokens = set(self._tokenize(text))  # Tokenize the text
            for t in tokens:  # For each unique token
                # inverted_index maps token -> set of indices
                postings = self.inverted_index.get(t)  # Get existing postings
                if postings is None:  # If no postings yet
                    self.inverted_index[t] = {abs_idx}  # Create new set
                else:  # Add to existing set
                    postings.add(abs_idx)
                # doc_freq counts chunks containing token
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1  # Increment count
        
        # Save to disk
        self._save()  # Persist changes

    def _save(self):
        """Persist index and metadata to disk"""
        try:
            cpu_index = self._runtime_to_cpu_index()
            faiss.write_index(cpu_index, self.index_path)  # Save FAISS index
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)  # Save metadata
            # Persist inverted index and document frequencies
            try:
                with open(self.inverted_path, 'wb') as f:
                    pickle.dump({'inverted_index': self.inverted_index, 'doc_freq': self.doc_freq}, f)  # Save inverted index
            except Exception:  # If error saving inverted index
                pass  # Skip it
        except Exception as e:  # If error saving
            pass  # Skip saving

    def search(self, query_embedding, k):
        """Search for k nearest neighbors of the query embedding"""
        D, I = self.index.search(  # Perform FAISS search
            np.array([query_embedding]).astype("float32"), k  # Query vector, k results
        )
        # Return metadata list like before (compatibility)
        results = []  # List to store results
        for idx, dist in zip(I[0], D[0]):  # For each result
            if idx < 0 or idx >= len(self.metadata):  # Invalid index
                continue  # Skip
            md = dict(self.metadata[idx])  # Get metadata copy
            # Convert L2 distance to a similarity-like score
            md['_vector_score'] = 1.0 / (1.0 + float(dist))  # Higher score for closer vectors
            results.append(md)  # Add to results
        return results  # Return list of metadata with scores

    def hybrid_search(self, query_embedding, query_text, k=10, alpha=0.6):
        """(Deprecated) Manual hybrid search using the built‑in inverted index.

        The application now uses LangChain's BM25+cosine+RRF retriever; this
        helper is retained for backwards compatibility and testing but is no
        longer invoked by the service layer.

        - `alpha` controls weight for semantic score (0..1). The lexical score gets (1-alpha).
        """
        # Semantic candidates - vector similarity
        D, I = self.index.search(np.array([query_embedding]).astype("float32"), k)  # Search vectors
        sem_scores = {}  # Semantic scores dictionary
        sem_candidates = set()  # Set of candidate indices
        for idx, dist in zip(I[0], D[0]):  # For each semantic result
            if idx < 0 or idx >= len(self.metadata):  # Invalid index
                continue  # Skip
            score = 1.0 / (1.0 + float(dist))  # Convert distance to score
            sem_scores[idx] = score  # Store score
            sem_candidates.add(idx)  # Add to candidates

        # Lexical candidates via inverted index
        tokens = set(self._tokenize(query_text))  # Tokenize query
        lex_scores = {}  # Lexical scores dictionary
        for t in tokens:  # For each token in query
            postings = self.inverted_index.get(t, set())  # Get documents containing token
            idf = math.log(1 + (max(1, len(self.metadata)) / (1 + self.doc_freq.get(t, 0))))  # IDF calculation
            for idx in postings:  # For each document with token
                lex_scores[idx] = lex_scores.get(idx, 0.0) + idf  # Add IDF score

        if lex_scores:  # If there are lexical scores
            max_lex = max(lex_scores.values())  # Get max lexical score
        else:
            max_lex = 1.0  # Default

        # Combine candidate pools
        all_candidates = set(list(sem_candidates) + list(lex_scores.keys()))  # Union of candidates
        scored = []  # List of (score, metadata) tuples
        for idx in all_candidates:  # For each candidate
            s_sem = sem_scores.get(idx, 0.0)  # Get semantic score
            s_lex = lex_scores.get(idx, 0.0) / max_lex if max_lex > 0 else 0.0  # Normalize lexical score
            combined = alpha * s_sem + (1.0 - alpha) * s_lex  # Weighted combination
            md = dict(self.metadata[idx])  # Get metadata
            md['_vector_score'] = s_sem  # Add semantic score
            md['_lexical_score'] = s_lex  # Add lexical score
            md['_score'] = combined  # Add combined score
            scored.append((combined, md))  # Add to scored list

        # Sort and return top-k
        scored.sort(key=lambda x: x[0], reverse=True)  # Sort by combined score descending
        return [md for _, md in scored[:k]]  # Return top k metadata

    def get_stats(self):
        """Get statistics about the vector store"""
        return {
            'total_vectors': self.index.ntotal,  # Number of vectors in index
            'total_chunks': len(self.metadata),  # Number of text chunks
            'processed_files': len(self.file_tracker)  # Number of processed files
        }

    def reset_store(self):
        """Clear vectors, metadata and tracking data, then persist empty state."""
        self.index = self._attach_runtime_index(self._build_cpu_index())

        self.metadata = []
        self.inverted_index = {}
        self.doc_freq = {}
        self.file_tracker = {}
        self._save()
        self._save_file_tracker()

