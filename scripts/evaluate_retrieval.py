"""Evaluate retrieval quality against labeled query examples.

Usage:
    python scripts/evaluate_retrieval.py \
        --kb-path data/processed/bible_study_chunks.jsonl \
        --examples data/eval/retrieval_queries.jsonl \
        --top-k 5
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from yaf_gpt.data.knowledge_base import KnowledgeBase
from yaf_gpt.data.study_chunk import StudyChunk


@dataclass
class RetrievalExample:
    """Single evaluation example with expected metadata."""

    query: str
    expected_doc_ids: list[str]
    expected_sections: list[str] | None = None
    notes: str | None = None


def load_examples(path: Path) -> list[RetrievalExample]:
    """Load evaluation examples from JSONL."""
    examples: list[RetrievalExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            examples.append(
                RetrievalExample(
                    query=payload["query"],
                    expected_doc_ids=payload.get("expected_doc_ids", []),
                    expected_sections=payload.get("expected_sections"),
                    notes=payload.get("notes"),
                )
            )
    return examples


def reciprocal_rank(
    retrieved: Iterable[tuple[StudyChunk, float]],
    expected_doc_ids: set[str],
) -> float:
    """Compute reciprocal rank for a single query."""
    for idx, (chunk, _) in enumerate(retrieved, start=1):
        if chunk.doc_id in expected_doc_ids:
            return 1.0 / idx
    return 0.0


def hit(retrieved: Iterable[tuple[StudyChunk, float]], expected_doc_ids: set[str]) -> bool:
    """Return True if any expected doc appears in retrieved results."""
    return any(chunk.doc_id in expected_doc_ids for chunk, _ in retrieved)


def evaluate_retriever(
    kb: KnowledgeBase,
    examples: list[RetrievalExample],
    top_k: int,
) -> dict[str, object]:
    """Run retrieval evaluation and return aggregate + per-query metrics."""
    per_query: list[dict[str, object]] = []
    mrr_scores: list[float] = []
    hits: list[bool] = []

    for example in examples:
        expected_doc_ids = set(example.expected_doc_ids)
        retrieved = kb.search(example.query, top_k=top_k)
        mrr = reciprocal_rank(retrieved, expected_doc_ids)
        is_hit = hit(retrieved, expected_doc_ids)

        per_query.append(
            {
                "query": example.query,
                "expected_doc_ids": example.expected_doc_ids,
                "retrieved_doc_ids": [chunk.doc_id for chunk, _ in retrieved],
                "mrr": mrr,
                "hit": is_hit,
                "notes": example.notes,
            }
        )
        mrr_scores.append(mrr)
        hits.append(is_hit)

    mrr_mean = float(statistics.fmean(mrr_scores)) if mrr_scores else 0.0
    hit_rate = float(sum(hits) / len(hits)) if hits else 0.0

    return {
        "per_query": per_query,
        "aggregate": {
            "num_examples": len(examples),
            "mrr_mean": mrr_mean,
            "mrr_median": float(statistics.median(mrr_scores)) if mrr_scores else 0.0,
            "hit_rate": hit_rate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=Path("data/processed/bible_study_chunks.jsonl"),
        help="Path to the processed knowledge base JSONL.",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("data/eval/retrieval_queries.jsonl"),
        help="Path to JSONL file containing evaluation examples.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieval results to consider per query.",
    )
    args = parser.parse_args()

    if not args.examples.exists():
        raise SystemExit(
            f"Example file {args.examples} not found. "
            "Create it with labeled queries before running evaluation."
        )

    kb = KnowledgeBase(args.kb_path)
    examples = load_examples(args.examples)
    results = evaluate_retriever(kb, examples, top_k=args.top_k)

    aggregate = results["aggregate"]
    print("=== Retrieval Evaluation ===")
    print(f"Examples      : {aggregate['num_examples']}")
    print(f"Hit@{args.top_k:<7}: {aggregate['hit_rate']:.3f}")
    print(f"MRR (mean)    : {aggregate['mrr_mean']:.3f}")
    print(f"MRR (median)  : {aggregate['mrr_median']:.3f}")

    print("\n--- Per-query details ---")
    for record in results["per_query"]:
        print(f"Query: {record['query']}")
        print(f"  Expected docs : {record['expected_doc_ids']}")
        print(f"  Retrieved docs: {record['retrieved_doc_ids']}")
        print(f"  Hit: {record['hit']} | MRR: {record['mrr']:.3f}")
        if record["notes"]:
            print(f"  Notes: {record['notes']}")
        print()


if __name__ == "__main__":
    main()
