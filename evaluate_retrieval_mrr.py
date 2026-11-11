#!/usr/bin/env python3
"""Evaluate RAG retrieval performance using MRR and retrieval accuracy metrics.

This script reads QA pairs from a PDF (HOTPOTQA format), retrieves documents
for each question using the full RAG pipeline (query decomposition + hybrid
retrieval + reranking + sub-query-grouped generation), and evaluates:
- MRR (Mean Reciprocal Rank): averaged across ALL golden supporting documents
  per question, not just the first one found. HotpotQA questions are
  multi-hop and typically cite 2+ distinct supporting articles; collapsing
  to a single rank per question hides cases where only one of the two
  required hops was actually retrieved.
- Retrieval Accuracy (any-hit): % of questions where >=1 golden doc was retrieved
- Retrieval Accuracy (full-coverage): % of questions where ALL golden docs were retrieved

Usage examples:
    python evaluate_retrieval_mrr.py output/qa/qa_pairs.pdf
    python evaluate_retrieval_mrr.py output/qa/qa_pairs.pdf --top-k 5
    python evaluate_retrieval_mrr.py output/qa/qa_pairs.pdf --max-examples 50
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# Ensure proper imports
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import fitz
except ImportError as exc:
    raise ImportError(
        "PyMuPDF is required. Install it with `pip install pymupdf`."
    ) from exc

from rag_chatbot.services.retreival_service import retrieve_no_rerank, rerank, vector_store
from rag_chatbot.services.chat_service import generate_answer
from rag_chatbot.services.query_decomposer import get_subqueries
from rag_chatbot.main import index_documents


# Regex patterns to extract QA items from PDF (HotpotQA format)
QUESTION_PATTERN = re.compile(r"Question:\s*(.+?)(?=Answer:)", re.DOTALL)
ANSWER_PATTERN = re.compile(r"Answer:\s*(.+?)(?=Supporting Facts:)", re.DOTALL)
SUPPORTING_FACTS_PATTERN = re.compile(r"Supporting Facts:\s*(.+?)(?=Question:|\Z)", re.DOTALL)


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace while preserving content."""
    return re.sub(r"\s+", " ", text.strip())


def parse_supporting_facts(facts_text: str) -> List[str]:
    """Parse supporting facts to extract unique golden document names.

    Format: "DocumentName: sentence_id; DocumentName: sentence_id; ..."
    The same document name commonly repeats for several supporting
    sentences within that article (e.g. "Allie Goertz: 0; Allie Goertz: 1;
    Allie Goertz: 2") -- that's ONE golden document, not several. We
    deduplicate while preserving first-seen order so each entry in the
    returned list represents a distinct golden document that retrieval
    needs to surface, which is what HotpotQA's multi-hop supporting-doc
    set actually means.
    """
    if not facts_text:
        return []

    docs = []
    seen = set()
    for fact in facts_text.split(";"):
        fact = fact.strip()
        if fact and ":" in fact:
            doc_name = fact.split(":")[0].strip()
            key = doc_name.lower()
            if doc_name and key not in seen:
                seen.add(key)
                docs.append(doc_name)

    return docs


def load_qa_from_pdf(path: str) -> List[Dict[str, str]]:
    """Load QA pairs from a PDF file in HotpotQA format."""
    print(f"Loading QA pairs from {path}...")
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)

    entries = []
    qa_idx = 0

    questions = list(QUESTION_PATTERN.finditer(text))

    for idx, q_match in enumerate(questions):
        q_text = q_match.group(1).strip()

        search_start = q_match.end()

        answer_match = ANSWER_PATTERN.search(text, search_start)
        if not answer_match:
            continue

        a_text = answer_match.group(1).strip()

        facts_match = SUPPORTING_FACTS_PATTERN.search(text, answer_match.end())
        if not facts_match:
            facts = []
        else:
            facts = parse_supporting_facts(facts_match.group(1))

        entries.append({
            "id": str(qa_idx),
            "question": normalize_whitespace(q_text),
            "answer": normalize_whitespace(a_text),
            "supporting_docs": facts,  # List of distinct golden document names
        })
        qa_idx += 1

    print(f"Loaded {len(entries)} QA pairs")
    return entries


def ensure_indexed() -> None:
    """Index documents if the vector store is empty."""
    if vector_store.index.ntotal > 0:
        print(f"Vector store already indexed with {vector_store.index.ntotal} chunks")
        return

    print("Indexing documents from rag_chatbot/data/documents...")
    index_documents()

    if vector_store.index.ntotal == 0:
        raise RuntimeError(
            "No documents indexed. Ensure documents are in rag_chatbot/data/documents/"
        )
    print(f"Indexed {vector_store.index.ntotal} chunks")


def retrieve_with_decomposition(
    question: str, top_k: int, disable_decomposition: bool = False
) -> Tuple[List[Dict], Optional[List[Dict]], bool]:
    """Run the full retrieval pipeline: query decomposition + hybrid retrieval
    + dedup pooling + single rerank pass, mirroring the production /chat
    endpoint in main.py.

    Args:
        question: the question to retrieve for.
        top_k: number of chunks to keep after reranking.
        disable_decomposition: if True, skip get_subqueries entirely and treat
            the question as a single subquery. Used to run a controlled
            ablation -- same QA set, same retrieval/rerank/generation code,
            with only decomposition toggled off -- to isolate whether
            decomposition itself helps or hurts, independent of decomposition
            correlating with which questions happen to be harder.

    Returns:
        (chunks, subquery_results, was_decomposed)
        chunks: top_k reranked chunks (flat list, used for MRR/accuracy scoring
            and as the fallback context for generation).
        subquery_results: list of {"subquery": str, "chunks": [...]} restricted
            to the same top_k chunks, or None if the question wasn't decomposed.
            Passed to generate_answer so it can group context by sub-question.
        was_decomposed: True if more than one subquery was produced.
    """
    if disable_decomposition:
        subqueries = [question]
    else:
        subqueries = get_subqueries(question)
    was_decomposed = len(subqueries) > 1

    chunks: List[Dict] = []
    seen_keys = set()
    subquery_results = []

    for subquery in subqueries:
        sub_chunks = retrieve_no_rerank(subquery)
        subquery_results.append({"subquery": subquery, "chunks": sub_chunks})
        for chunk in sub_chunks:
            key = (chunk.get("source"), chunk.get("page"), chunk.get("chunk_id"))
            if key not in seen_keys:
                seen_keys.add(key)
                chunks.append(chunk)

    if len(chunks) > 1:
        chunks = rerank(question, chunks)

    # Slice to top_k for ranking metrics and for what generation actually sees.
    chunks = chunks[:top_k]

    decomposition_context = None
    if was_decomposed:
        kept_keys = {(c.get("source"), c.get("page"), c.get("chunk_id")) for c in chunks}
        for item in subquery_results:
            item["chunks"] = [
                c for c in item["chunks"]
                if (c.get("source"), c.get("page"), c.get("chunk_id")) in kept_keys
            ]
        decomposition_context = subquery_results

    return chunks, decomposition_context, was_decomposed


def _doc_appears_in_chunk(doc_name: str, chunk_text: str) -> bool:
    """Check whether a golden document name appears to be the source of a chunk."""
    doc_lower = doc_name.lower().strip()
    return (
        f"title: {doc_lower}" in chunk_text
        or f"{doc_lower}:" in chunk_text
        or doc_lower in chunk_text
    )


def check_context_match(retrieved_chunks: List[Dict], supporting_docs: List[str]) -> Dict[str, int]:
    """For EACH distinct golden document, find the earliest (best) rank at
    which it appears among the retrieved chunks.

    Unlike the previous implementation, this does not stop at the first
    golden doc found -- it tracks every golden doc independently so a
    question's score reflects whether retrieval surfaced ALL the documents
    a multi-hop question actually needs, not just one of them.

    Returns:
        Dict mapping golden doc name -> 1-based rank of first chunk that
        appears to come from it, or 0 if that doc was never found in the
        retrieved set.
    """
    doc_ranks: Dict[str, int] = {doc: 0 for doc in supporting_docs}
    if not supporting_docs:
        return doc_ranks

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "").lower()
        for doc in supporting_docs:
            if doc_ranks[doc] == 0 and _doc_appears_in_chunk(doc, text):
                doc_ranks[doc] = rank

    return doc_ranks


def calculate_question_mrr(doc_ranks: Dict[str, int]) -> float:
    """Average reciprocal rank across all golden docs for one question.

    A golden doc that was never retrieved contributes 0 to the average
    (not skipped), so questions where only one of two required hops was
    found are correctly penalized relative to questions where both were.
    """
    if not doc_ranks:
        return 0.0
    reciprocal_ranks = [(1.0 / rank) if rank > 0 else 0.0 for rank in doc_ranks.values()]
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def normalize_text(text: str) -> str:
    """Normalize text for exact-match and F1 comparisons."""
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[\s\n]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Split normalized text into tokens."""
    if not text:
        return []
    return [t for t in text.split() if t]


def compute_f1(prediction: str, reference: str) -> float:
    """Compute the token-level F1 score between prediction and reference."""
    pred_tokens = tokenize(normalize_text(prediction))
    ref_tokens = tokenize(normalize_text(reference))

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = __import__('collections').Counter(pred_tokens) & __import__('collections').Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_retrieval(
    qa_pairs: List[Dict[str, str]],
    top_k: int = 10,
    max_examples: int = None,
    disable_decomposition: bool = False,
) -> Dict:
    """Evaluate retrieval performance for all QA pairs."""

    if max_examples:
        qa_pairs = qa_pairs[:max_examples]

    results = {
        "total_questions": len(qa_pairs),
        "top_k": top_k,
        "decomposition_disabled": disable_decomposition,
        "questions": [],
        "mrr_scores": [],
        "any_hit_accuracies": [],       # >=1 golden doc retrieved
        "full_coverage_accuracies": [],  # ALL golden docs retrieved
        "f1_scores": [],
        "exact_match_scores": [],
    }

    print(f"\nEvaluating retrieval for {len(qa_pairs)} questions (top-k={top_k})...\n")

    for idx, qa in enumerate(qa_pairs, start=1):
        qa_id = qa["id"]
        supporting_docs = qa["supporting_docs"]
        question = qa["question"]

        if not supporting_docs:
            print(f"[{idx}/{len(qa_pairs)}] QA {qa_id}: SKIPPED (no supporting facts)")
            results["total_questions"] -= 1
            continue

        # Retrieve documents for this question using the full decomposition pipeline
        try:
            retrieved_chunks, subquery_results, was_decomposed = retrieve_with_decomposition(
                question, top_k, disable_decomposition=disable_decomposition
            )
        except Exception as e:
            print(f"[{idx}/{len(qa_pairs)}] QA {qa_id}: ERROR - {e}")
            results["questions"].append({
                "qa_id": qa_id,
                "question": question,
                "supporting_docs": supporting_docs,
                "doc_ranks": {},
                "any_found": False,
                "all_found": False,
                "mrr": 0.0,
                "error": str(e)
            })
            results["mrr_scores"].append(0.0)
            results["any_hit_accuracies"].append(0)
            results["full_coverage_accuracies"].append(0)
            continue

        # Check, per golden doc, whether and where it was retrieved
        doc_ranks = check_context_match(retrieved_chunks, supporting_docs)
        mrr = calculate_question_mrr(doc_ranks)
        any_found = any(rank > 0 for rank in doc_ranks.values())
        all_found = all(rank > 0 for rank in doc_ranks.values())

        # Generate an answer using the full RAG pipeline, including sub-query
        # grouped context when the question was decomposed.
        try:
            predicted = generate_answer(question, retrieved_chunks, subquery_results=subquery_results)
        except Exception:
            predicted = ""

        reference = qa.get("answer", "")
        f1 = compute_f1(predicted, reference)
        exact_match = 1 if normalize_text(predicted) == normalize_text(reference) else 0

        results["questions"].append({
            "qa_id": qa_id,
            "question": question[:60] + "..." if len(question) > 60 else question,
            "supporting_docs": supporting_docs,
            "doc_ranks": doc_ranks,
            "any_found": any_found,
            "all_found": all_found,
            "mrr": mrr,
            "was_decomposed": was_decomposed,
            "retrieved_sources": [c.get("source", "unknown") for c in retrieved_chunks[:3]],
            "predicted_answer": predicted,
            "reference_answer": reference,
            "f1": f1,
            "exact_match": exact_match,
        })
        results["mrr_scores"].append(mrr)
        results["any_hit_accuracies"].append(1 if any_found else 0)
        results["full_coverage_accuracies"].append(1 if all_found else 0)
        results["f1_scores"].append(f1)
        results["exact_match_scores"].append(exact_match)

        status = "✓✓ ALL" if all_found else ("✓ SOME" if any_found else "✗ NONE")
        print(f"[{idx}/{len(qa_pairs)}] QA {qa_id}: {status} found ({len(supporting_docs)} golden docs, mrr={mrr:.3f})")

    # Calculate aggregate metrics
    if results["mrr_scores"]:
        n = len(results["mrr_scores"])
        results["average_mrr"] = sum(results["mrr_scores"]) / n
        results["average_any_hit_accuracy"] = sum(results["any_hit_accuracies"]) / n
        results["any_hit_accuracy_percentage"] = results["average_any_hit_accuracy"] * 100
        results["average_full_coverage_accuracy"] = sum(results["full_coverage_accuracies"]) / n
        results["full_coverage_accuracy_percentage"] = results["average_full_coverage_accuracy"] * 100
        results["average_f1"] = sum(results["f1_scores"]) / len(results["f1_scores"]) if results["f1_scores"] else 0
        results["overall_exact_match"] = sum(results["exact_match_scores"]) / len(results["exact_match_scores"]) if results["exact_match_scores"] else 0
        results["overall_exact_match_percentage"] = results["overall_exact_match"] * 100
    else:
        results["average_mrr"] = 0
        results["average_any_hit_accuracy"] = 0
        results["any_hit_accuracy_percentage"] = 0
        results["average_full_coverage_accuracy"] = 0
        results["full_coverage_accuracy_percentage"] = 0
        results["average_f1"] = 0
        results["overall_exact_match"] = 0
        results["overall_exact_match_percentage"] = 0

    # Explicit, unambiguous top-level summary: the four headline metrics
    # (MRR, Retrieval Accuracy, F1, Exact Match), each with a single clear name.
    # "retrieval_accuracy" is aliased to the any-hit definition (>=1 golden doc
    # retrieved) since that's the conventional single-number retrieval accuracy;
    # full_coverage_accuracy remains available separately for the stricter
    # multi-hop "all golden docs retrieved" view.
    results["summary"] = {
        "mrr": results["average_mrr"],
        "retrieval_accuracy": results["average_any_hit_accuracy"],
        "retrieval_accuracy_percentage": results["any_hit_accuracy_percentage"],
        "full_coverage_retrieval_accuracy": results["average_full_coverage_accuracy"],
        "full_coverage_retrieval_accuracy_percentage": results["full_coverage_accuracy_percentage"],
        "f1": results["average_f1"],
        "exact_match": results["overall_exact_match"],
        "exact_match_percentage": results["overall_exact_match_percentage"],
    }

    return results


def print_results(results: Dict) -> None:
    """Print evaluation results in a readable format."""
    print("\n" + "="*80)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*80)

    print(f"\nDataset: {results['total_questions']} questions (HotpotQA format)")
    print(f"Top-K: {results['top_k']}")
    print(f"Query Decomposition: {'DISABLED' if results.get('decomposition_disabled') else 'enabled'}")

    print(f"\n{'HEADLINE METRICS':-^80}")
    s = results["summary"]
    print(f"  MRR (Mean Reciprocal Rank):      {s['mrr']:.4f}")
    print(f"  Retrieval Accuracy:               {s['retrieval_accuracy']:.4f} ({s['retrieval_accuracy_percentage']:.2f}%)")
    print(f"  F1 (answers):                     {s['f1']:.4f}")
    print(f"  Exact Match (answers):            {s['exact_match']:.4f} ({s['exact_match_percentage']:.2f}%)")

    print(f"\n{'AGGREGATE METRICS (DETAIL)':-^80}")
    print(f"  Average MRR (averaged across ALL golden docs per question): {results['average_mrr']:.4f}")
    print(f"  Any-Hit Accuracy: {results['average_any_hit_accuracy']:.4f} ({results['any_hit_accuracy_percentage']:.2f}%)")
    print(f"  (Any-Hit = % of questions where >=1 golden doc was retrieved)")
    print(f"  Full-Coverage Accuracy: {results['average_full_coverage_accuracy']:.4f} ({results['full_coverage_accuracy_percentage']:.2f}%)")
    print(f"  (Full-Coverage = % of questions where ALL golden docs were retrieved -- meaningful for multi-hop questions)")
    print(f"  Average F1 (answers): {results.get('average_f1', 0):.4f}")
    print(f"  Exact-match Accuracy (answers): {results.get('overall_exact_match', 0):.4f} ({results.get('overall_exact_match_percentage', 0):.2f}%)")

    # Count how many questions had their best-found golden doc at each rank
    rank_distribution = {}
    for q in results["questions"]:
        for rank in q.get("doc_ranks", {}).values():
            if rank > 0:
                rank_distribution[rank] = rank_distribution.get(rank, 0) + 1

    if rank_distribution:
        print(f"\n{'GOLDEN-DOC RANK DISTRIBUTION (counts per individual doc, not per question)':-^80}")
        for rank in sorted(rank_distribution.keys()):
            count = rank_distribution[rank]
            print(f"  Rank {rank:2d}: {count:3d} golden docs found")

    print(f"\n{'DETAILED RESULTS PER QUESTION':-^80}")
    for q in results["questions"]:
        if q.get("all_found"):
            status = "✓✓ ALL GOLDEN DOCS FOUND"
        elif q.get("any_found"):
            status = "✓ PARTIAL (some golden docs missing)"
        else:
            status = "✗ NONE FOUND"
        print(f"\n  QA {q['qa_id']}: {status}")
        print(f"    Question: {q['question']}")
        print(f"    Golden Docs & Ranks: {q.get('doc_ranks', {})}")
        print(f"    MRR: {q['mrr']:.4f}")
        if q.get("was_decomposed"):
            print(f"    Decomposed: yes")
        print(f"    F1: {q.get('f1', 0):.4f}  ExactMatch: {q.get('exact_match', 0)}")
        if q.get("retrieved_sources"):
            print(f"    Top 3 Retrieved: {q['retrieved_sources']}")
        if q.get("error"):
            print(f"    Error: {q['error']}")

    print("\n" + "="*80)


def save_results(results: Dict, output_path: str) -> None:
    """Save results to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval performance with MRR and accuracy metrics"
    )
    parser.add_argument(
        "qa_file",
        help="Path to the QA PDF file (e.g., output/qa/qa_pairs.pdf)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top retrieved documents to consider (default: 10)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file for results (default: results_mrr_{input_filename}.json)"
    )
    parser.add_argument(
        "--no-decomposition",
        action="store_true",
        help=(
            "Disable query decomposition (skip get_subqueries entirely, treat every "
            "question as a single query). Use this to run a controlled ablation against "
            "a normal run on the same QA file, to isolate decomposition's actual effect "
            "from the fact that it tends to fire on intrinsically harder questions."
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(args.qa_file):
        print(f"Error: QA file not found at {args.qa_file}")
        sys.exit(1)

    ensure_indexed()

    qa_pairs = load_qa_from_pdf(args.qa_file)

    results = evaluate_retrieval(
        qa_pairs,
        top_k=args.top_k,
        max_examples=args.max_examples,
        disable_decomposition=args.no_decomposition,
    )

    print_results(results)

    default_suffix = "_no_decomp" if args.no_decomposition else ""
    output_path = args.output or f"results_mrr_{os.path.basename(args.qa_file).replace('.pdf', '')}{default_suffix}.json"
    save_results(results, output_path)


if __name__ == "__main__":
    main()