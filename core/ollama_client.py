"""
Light wrapper around Ollama HTTP API used for chat and image generation.

This module centralizes calls to the Ollama server (`OLLAMA_URL`) and
provides helper functions used across the project:
- `chat(model, messages)`: send chat-style messages and return text
- `generate_with_image(...)`: send an image + prompt and return response

Both functions raise Exceptions on HTTP or timeout errors so callers can
handle retries or fallback behavior.
"""

# Import necessary libraries
import requests  # For making HTTP requests to Ollama
import sys  # For system operations
import os  # For path operations
import time  # For timing (though not used here)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import OLLAMA_URL, build_ollama_options  # URL and runtime options

# Timeout values for different operations
EMBEDDING_TIMEOUT = 30  # Timeout for embedding requests (not used in this file)
CHAT_TIMEOUT = 120  # Timeout for chat requests


def chat(model: str, messages: list, timeout: int = None, options: dict = None):
    """Call Ollama chat endpoint and return the assistant text.

    - `model`: model name to query
    - `messages`: list-of-dicts in role/content format
    - returns: assistant message content (string)
    """
    if timeout is None:  # Use default timeout if not specified
        timeout = CHAT_TIMEOUT

    payload = {"model": model, "messages": messages, "stream": False}
    merged_options = build_ollama_options()
    if options:
        merged_options.update(options)
    if merged_options:
        payload["options"] = merged_options

    try:
        # Send POST request to Ollama's chat API
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",  # API endpoint URL
            json=payload,  # Request data
            timeout=timeout,  # Timeout for the request
        )
        response.raise_for_status()  # Raise exception for bad HTTP status
        return response.json()["message"]["content"]  # Extract and return the response text
    except requests.exceptions.Timeout:  # Handle timeout errors
        raise Exception(f"Ollama chat request timed out after {timeout}s")
    except requests.exceptions.ConnectionError:  # Handle connection errors
        raise Exception(f"Cannot reach Ollama at {OLLAMA_URL}")
    except Exception as e:  # Handle other errors
        raise Exception(f"Chat error: {str(e)}")


def generate_with_image(model: str, prompt: str, image_base64: str, timeout: int = None):
    """Send an image and prompt to the Ollama generate API and return text."""
    if timeout is None:  # Use default timeout if not specified
        timeout = CHAT_TIMEOUT

    payload = {"model": model, "prompt": prompt, "images": [image_base64], "stream": False}
    options = build_ollama_options()
    if options:
        payload["options"] = options

    try:
        # Send POST request to Ollama's generate API with image
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",  # API endpoint for generation
            json=payload,  # Include image data
            timeout=timeout,  # Timeout for the request
        )
        response.raise_for_status()  # Check for HTTP errors
        return response.json()["response"]  # Return the generated text
    except requests.exceptions.Timeout:  # Handle timeout
        raise Exception(f"Image generation timed out after {timeout}s")
    except Exception as e:  # Handle other errors
        raise Exception(f"Image generation error: {str(e)}")
