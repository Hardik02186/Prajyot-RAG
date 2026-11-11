#!/usr/bin/env python3
"""Evaluate QA PDFs with the chatbot and report F1 metrics.

This script reads the QA files under `output/qa`, sends each question to the
model, and scores the returned answer against the reference answer using the
F1 metric.

Usage examples:
    python evaluate_qa.py
    python evaluate_qa.py --qa-dir output/qa --max-examples 50
    python evaluate_qa.py --model-only
"""

import argparse
import collections
import json
import os
import re
import sys
import time

# Ensure the repo root is on the import path so the local package imports work.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rag_chatbot.core.ollama_client import chat
from rag_chatbot.config import CHAT_MODEL

try:
    import fitz
except ImportError as exc:
    raise ImportError(
        "PyMuPDF is required to run evaluate_qa.py. Install it with `pip install pymupdf`."
    ) from exc


def get_retrieval_modules():
    """Import RAG helpers only when evaluation actually runs."""
    from rag_chatbot.services.chat_service import generate_answer
    from rag_chatbot.services.retreival_service import retrieve, vector_store
    from rag_chatbot.main import index_documents
    return generate_answer, retrieve, vector_store, index_documents


QA_ITEM_PATTERN = re.compile(
    r"QA ID:\s*(?P<id>\d+)\s*"
    r"Context ID:\s*(?P<context>.*?)\s*"
    r"Question:\s*(?P<question>.*?)\s*"
    r"Answer:\s*(?P<answer>.*?)(?=(?:QA ID:\s*\d+)|\Z)",
    re.S,
)


def normalize_text(text: str) -> str:
    """Normalize text for fair F1 matching."""
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[\s\n]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
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

    common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def load_qa_from_pdf(path: str) -> list[dict[str, str]]:
    """Load QA pairs from a PDF QA file."""
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    entries = []
    for match in QA_ITEM_PATTERN.finditer(text):
        entries.append(
            {
                "id": match.group("id").strip(),
                "context_id": match.group("context").strip(),
                "question": normalize_whitespace(match.group("question")),
                "answer": normalize_whitespace(match.group("answer")),
            }
        )
    return entries


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace while preserving readable content."""
    return re.sub(r"\s+", " ", text.strip())


def ask_question(question: str, model_only: bool = False) -> str:
    """Ask a question using the RAG pipeline or direct model chat."""
    if model_only:
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant. Answer the question concisely and precisely.",
            },
            {"role": "user", "content": question},
        ]
        return chat(CHAT_MODEL, messages)

    generate_answer, retrieve, _, _ = get_retrieval_modules()
    chunks = retrieve(question)
    if chunks:
        return generate_answer(question, chunks)

    # Fallback to direct model if retrieval produced no context.
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant. Answer the question concisely and precisely.",
        },
        {"role": "user", "content": question},
    ]
    return chat(CHAT_MODEL, messages)


def ensure_indexed() -> None:
    """Index documents if the vector store is empty."""
    _, _, vector_store, index_documents = get_retrieval_modules()
    if vector_store.index.ntotal > 0:
        return
    print("Indexing documents from the rag_chatbot data folder...")
    index_documents()
    if vector_store.index.ntotal == 0:
        raise RuntimeError(
            "No documents were indexed. Please ensure `data/documents/` contains PDFs and Ollama is running."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate QA PDFs with LLM answers and F1 metrics.")
    parser.add_argument("--qa-dir", default="output/qa", help="Folder containing QA PDF files.")
    parser.add_argument("--max-examples", type=int, default=0, help="Limit the number of evaluated questions per file.")
    parser.add_argument("--model-only", action="store_true", help="Skip retrieval and query the model directly.")
    parser.add_argument("--threshold", type=float, default=0.5, help="F1 threshold to count a question as correct.")
    parser.add_argument("--output", default="qa_evaluation.json", help="Optional JSON output summary file.")
    parser.add_argument("--verbose", action="store_true", help="Print per-question results.")
    args = parser.parse_args()

    qa_dir = os.path.abspath(args.qa_dir)
    if not os.path.isdir(qa_dir):
        raise FileNotFoundError(f"QA directory not found: {qa_dir}")

    if not args.model_only:
        ensure_indexed()

    all_results = []
    evaluations = []
    pdf_files = sorted(
        f for f in os.listdir(qa_dir) if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found under {qa_dir}")

    start_time = time.time()
    for pdf_name in pdf_files:
        pdf_path = os.path.join(qa_dir, pdf_name)
        questions = load_qa_from_pdf(pdf_path)
        if not questions:
            continue

        if args.max_examples > 0:
            questions = questions[: args.max_examples]

        file_results = []
        print(f"Evaluating {pdf_name} ({len(questions)} questions)...")

        for item in questions:
            question = item["question"]
            reference = item["answer"]
            try:
                response = ask_question(question, model_only=args.model_only)
            except Exception as exc:
                print(f"  ERROR for QA {item['id']}: {exc}")
                response = ""

            f1 = compute_f1(response, reference)
            correct = f1 >= args.threshold
            file_results.append(
                {
                    "qa_id": item["id"],
                    "question": question,
                    "reference": reference,
                    "response": response,
                    "f1_score": round(f1, 4),
                    "correct": correct,
                }
            )
            all_results.append(file_results[-1])
            if args.verbose:
                print(
                    f"  QA {item['id']}: F1={f1:.4f} | correct={correct} | question={question!r}"
                )

        average_f1 = sum(r["f1_score"] for r in file_results) / len(file_results)
        correct_count = sum(r["correct"] for r in file_results)
        evaluations.append(
            {
                "file": pdf_name,
                "questions": len(file_results),
                "average_f1": round(average_f1, 4),
                "correct_count": correct_count,
                "accuracy": round(correct_count / len(file_results), 4),
                "results": file_results,
            }
        )
        print(
            f"  {pdf_name}: average F1={average_f1:.4f}, accuracy={correct_count}/{len(file_results)}"
        )

    elapsed = time.time() - start_time
    summary = {
        "mode": "model-only" if args.model_only else "rag",
        "qa_dir": qa_dir,
        "num_files": len(evaluations),
        "num_questions": len(all_results),
        "threshold": args.threshold,
        "total_time_seconds": round(elapsed, 2),
        "average_f1": round(
            sum(r["f1_score"] for r in all_results) / len(all_results), 4
        )
        if all_results
        else 0.0,
        "accuracy": round(
            sum(r["correct"] for r in all_results) / len(all_results), 4
        )
        if all_results
        else 0.0,
        "files": evaluations,
    }

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("\nEvaluation complete.")
    print(f"  Total questions: {len(all_results)}")
    print(f"  Overall average F1: {summary['average_f1']:.4f}")
    print(f"  Overall accuracy (@{args.threshold}): {summary['accuracy']:.4f}")
    print(f"  Summary written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
