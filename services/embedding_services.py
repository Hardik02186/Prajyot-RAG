# Import necessary libraries
import requests  # For HTTP requests to Ollama
import sys  # System operations
import os  # File operations
import time  # Timing operations

# Ensure proper imports by adding parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration values
from config import OLLAMA_URL, EMBED_MODEL, build_ollama_options  # Ollama config

# Timeout for embedding requests
EMBEDDING_TIMEOUT = 30  # seconds - how long to wait for embedding response


def is_ollama_available():
    """Return True if the Ollama server is reachable.

    Used at startup to decide whether to attempt indexing and embeddings.
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)  # Check server status
        return response.status_code == 200  # Return True if server responds OK
    except Exception:  # If any error (connection, timeout, etc.)
        return False  # Server not available


def embed_text(text):
    """Request an embedding vector for `text` from Ollama.

    Raises an Exception on connectivity or timeout issues so callers may
    surface an error to the user or retry.
    """
    try:
        start_time = time.time()  # Record start time (though not used)
        payload = {"model": EMBED_MODEL, "prompt": text}
        options = build_ollama_options()
        if options:
            payload["options"] = options
        response = requests.post(  # Send POST request to Ollama
            f"{OLLAMA_URL}/api/embeddings",  # Embedding API endpoint
            json=payload,  # Request data
            timeout=EMBEDDING_TIMEOUT,  # Timeout for request
        )
        response.raise_for_status()  # Raise exception for bad HTTP status
        return response.json()["embedding"]  # Return the embedding vector
    except requests.exceptions.ConnectionError:  # If can't connect
        raise Exception(f"Cannot connect to Ollama at {OLLAMA_URL}.")
    except requests.exceptions.Timeout:  # If request times out
        raise Exception(f"Embedding request timed out after {EMBEDDING_TIMEOUT}s.")
    except Exception as e:  # Other errors
        raise Exception(f"Embedding service error: {str(e)}")
