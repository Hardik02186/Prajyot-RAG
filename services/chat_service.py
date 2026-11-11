# Import necessary libraries
import sys  # System operations
import os  # File operations
import time  # Timing (not used here)
import re  # For parsing the FINAL ANSWER marker out of model responses

# Ensure proper imports by adding parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import chat function and model configuration
from core.ollama_client import chat  # Function to chat with AI model
from config import CHAT_MODEL  # Model configuration


def _extract_final_answer(raw_response):
    """Strip any comparison-reasoning scratchpad and return only the content
    after the last 'FINAL ANSWER:' marker (case-insensitive).

    The system prompt asks the model to show VALUE(...) extraction lines for
    comparison questions before committing to an answer, terminated by a
    'FINAL ANSWER:' line. This keeps that reasoning private to the model and
    returns only the clean final answer to callers, matching the original
    "shortest possible phrase" contract. If the marker is absent (e.g. the
    model didn't follow the format), the raw response is returned unchanged
    so behavior degrades gracefully rather than returning an empty string.
    """
    if not raw_response:
        return raw_response

    matches = list(re.finditer(r"final answer\s*:\s*", raw_response, re.IGNORECASE))
    if not matches:
        return raw_response.strip()

    return raw_response[matches[-1].end():].strip()


def _format_web_results(web_results):
    if not web_results:
        return ""

    lines = ["Web search results:"]
    for idx, result in enumerate(web_results, start=1):
        title = result.get("title") or "(no title)"
        snippet = result.get("snippet") or "(no snippet)"
        url = result.get("url") or ""
        lines.append(f"{idx}. {title}\n{snippet}\n{url}")
    return "\n\n".join(lines)


def _format_chunks(chunks):
    """Render a list of chunk dicts into a 'Source | Page' context block."""
    parts = []
    for c in chunks:
        source = c.get("source")
        page = c.get("page")
        if (source is None or page is None) and isinstance(c.get("metadata"), dict):
            source = source if source is not None else c["metadata"].get("source")
            page = page if page is not None else c["metadata"].get("page")
        parts.append(f"Source: {source} | Page: {page}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _build_context(chunks, subquery_results=None):
    """Build the context string, grouped by sub-query when decomposition happened.

    subquery_results: optional list of {"subquery": str, "chunks": [...]}
    representing the retrieval performed for each decomposed sub-question.
    If provided, this takes precedence over the flat `chunks` list and lets
    the model see which evidence answers which part of the original question.
    """
    if subquery_results:
        blocks = []
        for i, item in enumerate(subquery_results, start=1):
            subq = item.get("subquery", "")
            sub_chunks = item.get("chunks", [])
            chunk_text = _format_chunks(sub_chunks) if sub_chunks else "(no relevant context found)"
            blocks.append(f"[Sub-question {i}: {subq}]\n{chunk_text}")
        return "\n\n===\n\n".join(blocks)

    return _format_chunks(chunks)


def _build_prompt(question, context, web_results=None, was_decomposed=False):
    web_context = _format_web_results(web_results)
    system_content = """You are a precise question-answering assistant.

CRITICAL RULES:
1. Answer using ONLY the provided context. NEVER use your training knowledge to fill gaps.
   If the context does not contain the answer, say only: "I don't know"
2. Answer with the shortest possible phrase or word that directly answers the question.
   Do NOT repeat the question. Do NOT say "based on the context".
3. Match the TYPE of answer the question asks for:
   - "which magazine / which team / which film" → give the magazine/team/film NAME, not a date or person
   - "what character did X play" → give the CHARACTER name, not the actor
   - "what party / what profession / what industry" → give the party/profession/industry, not a person's name
   - "what year / when" → give only the year or date
   - "where / which country / which county" → give the place at the EXACT level asked (county not state, city not country)
   - "yes/no question" → answer only "yes" or "no"
   - "who" → give the person's full name as it appears in the context
4. Before saying "I don't know", carefully re-read ALL context paragraphs.
   The answer may use an alias, different phrasing, or be implied rather than stated directly.
   Only say "I don't know" if after careful re-reading the answer is truly absent.
5. Numbers: write them as digits (e.g. "17%", "6", "1844"), not as words.
6. COMPARISON / SUPERLATIVE questions (which is older/first/more/longer/etc. between two or
   more named things) are the most common source of mistakes. For these questions ONLY, before
   answering you MUST extract the specific value being compared for EACH entity on its own line,
   like:
     VALUE(EntityA): ...
     VALUE(EntityB): ...
   Then reason in one short line about which value satisfies the comparison asked
   (earlier date = "started first", higher count = "more", etc.). Only after that,
   on a new final line, output:
     FINAL ANSWER: <the minimal answer per rule 2 and rule 3>
   For ALL OTHER question types (non-comparison), skip the extraction step entirely and just output:
     FINAL ANSWER: <the minimal answer>
   The "FINAL ANSWER:" line is the only part that gets shown to the user, so the extraction
   step above it is your private workspace, not your final response, and it does not need to obey
   rule 2's brevity constraint.
7. EXACT REFERENT, not an adjacent one: context chunks often contain several similar-looking
   names, terms, or values in the same sentence or paragraph (other chemicals, other taxonomic
   ranks, other dates, other people). Before answering, re-read the question and identify the
   EXACT noun phrase it is asking about (e.g. "soluble in THIS chemical" refers to one specific
   substance named earlier in the same sentence, not a different substance mentioned nearby;
   "this genus" means the taxonomic genus, not a different rank like family or order). Do not
   substitute a nearby term just because it appears close to the relevant sentence — verify the
   value you extract is grammatically and logically the one the question points to, not merely
   the most prominent or first-mentioned term in the chunk.
8. ALIASES: if the context gives a person's stage name / nickname / shortened form AND their
   full legal or given name, prefer whichever form the QUESTION itself implies, and check whether
   the context explicitly equates the two ("X, real name Y" / "X (born Y)" / "also known as Y").
   If the question asks for an executive's, author's, or credited person's identity and the context
   distinguishes a real name from a stage name, default to the real/full name unless the question
   specifically asks for the stage name.

FEW-SHOT EXAMPLES (these are illustrative templates only — none of the entities,
facts, or answers below relate to your actual document collection; do not assume
similarity to any real question implies a matching answer):

Q: Which novel was published first, "Pride and Prejudice" or "Frankenstein"?
Context: "Pride and Prejudice" was published in 1813. "Frankenstein" was published in 1818.
VALUE(Pride and Prejudice): published 1813
VALUE(Frankenstein): published 1818
1813 is earlier than 1818, so Pride and Prejudice was published first.
FINAL ANSWER: Pride and Prejudice

Q: Which lake has the larger surface area, Lake Superior or Lake Victoria?
Context: Lake Superior has a surface area of about 82,100 square kilometers. Lake Victoria has a surface area of about 68,800 square kilometers.
VALUE(Lake Superior): ~82,100 km²
VALUE(Lake Victoria): ~68,800 km²
82,100 is greater than 68,800, so Lake Superior is larger.
FINAL ANSWER: Lake Superior

Q: This compound is slightly soluble in a particular solvent — what is that solvent called?
Context: Compound X is slightly soluble in glycerol. It reacts vigorously with vinegar to produce carbon dioxide.
The question asks for the solvent the compound is soluble IN, not a separate substance it reacts with
elsewhere in the same passage. The relevant sentence names that solvent as "glycerol".
FINAL ANSWER: glycerol

Q: A 2020 documentary included narration from a musician known publicly by a one-word stage name. What is that person's birth name?
Context: The documentary featured narration from Sting (born Gordon Sumner), an English musician.
The context equates the stage name "Sting" with the birth name "Gordon Sumner" via "(born Gordon Sumner)".
The question asks for the birth name specifically, not the stage name.
FINAL ANSWER: Gordon Sumner

Q: What character did the lead actor play in the film?
Context: The lead actor, playing a reclusive lighthouse keeper named Soren Voss, delivered a quietly intense performance.
FINAL ANSWER: Soren Voss

Q: What political party did the senator who signed the bill belong to?
Context: The bill was signed into law by Senator Mark Reilly, a Republican from Ohio.
FINAL ANSWER: Republican

Q: Which city is the capital of this fictional country in the example?
Context: Veridia's capital and largest city is Thornhaven.
FINAL ANSWER: Thornhaven

Q: In what year did construction begin on this bridge?
Context: Construction of the Ardmoor Bridge began in 1932 and concluded four years later.
FINAL ANSWER: 1932
"""
    if was_decomposed:
        system_content += (
            "\nThe context below is organized by sub-question, since the original "
            "question was broken into smaller retrieval queries. Synthesize across "
            "ALL sub-question blocks to answer the original question — the answer "
            "may require combining facts from more than one block."
        )

    if web_results:
        system_content += (
            "\nWhen web search results are included, answer using them carefully, cite the web sources, "
            "and explicitly state that this additional information comes from external web search results, not from the local document database."
        )

    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": f"{web_context}\n\nContext:\n{context}\n\nQuestion:\n{question}",
        },
    ]


def generate_answer(question, chunks=None, web_results=None, subquery_results=None):
    """Compose a prompt from retrieved chunks and optional decomposition/web results.

    Args:
        question: original user question.
        chunks: flat list of chunk dicts (used when no decomposition occurred,
            or as a fallback if subquery_results is not provided).
        web_results: optional web search results.
        subquery_results: optional list of {"subquery": str, "chunks": [...]}
            produced by running retrieval on each decomposed sub-query. When
            given, context is grouped per sub-question instead of flattened,
            giving the model explicit signal about which evidence supports
            which part of the original question.
    """
    was_decomposed = bool(subquery_results) and len(subquery_results) > 1
    context = _build_context(chunks or [], subquery_results)

    messages = _build_prompt(question, context, web_results, was_decomposed=was_decomposed)

    # Send to AI and return response
    raw_response = chat(CHAT_MODEL, messages)  # Call chat function with chat model
    return _extract_final_answer(raw_response)