"""Ollama-powered query decomposition for complex RAG queries.

This module detects when a user question is complex enough to benefit from
multiple focused retrieval queries. It avoids unnecessary decomposition for
simple questions and only calls Ollama twice when decomposition is needed:
1) complexity assessment
2) subquery generation.
"""

import json
import os
import re
import sys
from datetime import datetime

# Ensure the application root is on the import path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ollama_client import chat
from config import QUERY_DECOMPOSER_MODEL

# Maximum number of subqueries to generate for a complex query
MAX_SUBQUERIES = 5
# Complexity values greater than 1 trigger decomposition.
COMPLEXITY_THRESHOLD = 1
# Debug file path for query decomposition details. Write to the parent application package
# so the file is easier to locate during development.
DEBUG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "query_decomposer_debug.json")

_COMPLEXITY_PROMPT = (
    "You are a retrieval planner. Decide whether the user question needs to be "
    "split into multiple focused retrieval queries. If the question can be handled "
    "by a single search query, return exactly:\n"
    "COMPLEXITY: 1\nDECOMPOSE: NO\n"
    "If it requires a multi-step retrieval plan, return exactly:\n"
    "COMPLEXITY: N\nDECOMPOSE: YES\n"
    "where N is an integer between 2 and 5 describing the number of subqueries. "
    "Do not add any additional explanation or commentary."
)

_DECOMPOSITION_PROMPT = (
    "You are a query decomposer. Given a user question and a target number of "
    "subqueries, produce that many focused retrieval queries. Return only the "
    "final subqueries as separate lines or numbered items, with no extra text."
)


def _parse_complexity_response(response: str) -> int:
    """Parse the model response and return a complexity level from 1 to 5."""
    if not response or not isinstance(response, str):
        return 1

    # Prefer explicitly labeled complexity values.
    match = re.search(r"COMPLEXITY\s*[:\-]?\s*(\d)", response, re.IGNORECASE)
    if match:
        level = int(match.group(1))
        return max(1, min(5, level))

    # Fallback to keyword heuristics.
    response_lower = response.lower()
    if "decompose: yes" in response_lower or "yes" in response_lower:
        return 3
    return 1


def _parse_subqueries(response: str, expected_count: int) -> list[str]:
    """Extract subqueries from Ollama response text."""
    if not response or not isinstance(response, str):
        return []

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    parsed = []
    for line in lines:
        # Drop numeric bullets and common list markers.
        cleaned = re.sub(r"^\s*[\-\*\d\.\)]+\s*", "", line)
        if cleaned:
            parsed.append(cleaned)

    if len(parsed) >= expected_count:
        return parsed[:expected_count]
    if len(parsed) > 1:
        return parsed

    # If no bullets were provided, split on sentence boundaries.
    sentences = re.split(r"(?<=[.?!])\s+", response.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()][:expected_count]


def assess_query_complexity(question: str) -> int:
    """Ask Ollama whether the query is complex enough to decompose."""
    messages = [
        {"role": "system", "content": _COMPLEXITY_PROMPT},
        {"role": "user", "content": f"User query:\n{question}"},
    ]
    response = chat(QUERY_DECOMPOSER_MODEL, messages)
    return _parse_complexity_response(response)


def decompose_query(question: str, complexity_level: int) -> list[str]:
    """Generate a set of focused subqueries for a complex question."""
    query_count = max(2, min(MAX_SUBQUERIES, complexity_level))
    messages = [
        {"role": "system", "content": _DECOMPOSITION_PROMPT},
        {
            "role": "user",
            "content": (
                f"User question:\n{question}\n\n"
                f"Number of subqueries: {query_count}\n"
                "Produce each subquery as a clear, focused retrieval question."
            ),
        },
    ]
    response = chat(QUERY_DECOMPOSER_MODEL, messages)
    return _parse_subqueries(response, query_count)


def _write_debug_file(question: str, decomposed: bool, subqueries: list[str], complexity: int) -> None:
    """Write query decomposition debug state to a JSON file."""
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "original_query": question,
        "decomposed": decomposed,
        "complexity_level": complexity,
        "subqueries": subqueries,
    }
    try:
        with open(DEBUG_FILE_PATH, "w", encoding="utf-8") as debug_file:
            json.dump(payload, debug_file, indent=2, ensure_ascii=False)
    except Exception:
        # Debug logging should not break the main query flow.
        pass


def get_subqueries(question: str) -> list[str]:
    """Return either the original question or a list of decomposed subqueries."""
    if not question or not question.strip():
        _write_debug_file(question, False, [question], 0)
        return [question]

    complexity = assess_query_complexity(question)
    decomposed = complexity > COMPLEXITY_THRESHOLD
    if not decomposed:
        _write_debug_file(question, False, [question], complexity)
        return [question]

    subqueries = decompose_query(question, complexity)
    if not subqueries:
        _write_debug_file(question, False, [question], complexity)
        return [question]

    subqueries = [subquery for subquery in subqueries if subquery and subquery.strip()] or [question]
    _write_debug_file(question, decomposed, subqueries, complexity)
    return subqueries
