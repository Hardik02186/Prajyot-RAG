"""Web search fallback support for the RAG chatbot.

This module provides SerpAPI integration for external web searches. It is
used when the local document corpus does not contain enough information to
answer the user's question.
"""

import os
import sys
import requests

# Make sure the project root is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SERPAPI_API_KEY

SERPAPI_URL = "https://serpapi.com/search.json"


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Run a web search query using SerpAPI and return a list of result snippets."""
    if not SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY is required for web search. Set it via environment variable.")

    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "num": max_results,
    }
    response = requests.get(SERPAPI_URL, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    results = []
    for item in payload.get("organic_results", [])[:max_results]:
        results.append(
            {
                "title": item.get("title"),
                "snippet": item.get("snippet") or item.get("description"),
                "url": item.get("link"),
            }
        )
    return results
