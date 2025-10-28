"""CLI helper to inspect the processed knowledge base."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from textwrap import fill

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from yaf_gpt.data.StudyChunk import StudyChunk


def load_chunks(path: Path) -> list[StudyChunk]:
    chunks: list[StudyChunk] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            chunks.append(StudyChunk(**json.loads(line)))
    return chunks


def filter_chunks(chunks: list[StudyChunk], book: str | None) -> list[StudyChunk]:
    if book:
        return [chunk for chunk in chunks if chunk.book.lower() == book.lower()]
    return chunks


def print_summary(chunks: list[StudyChunk]) -> None:
    if not chunks:
        print("No chunks found for the given filter.")
        return

    lengths = [len(chunk.text) for chunk in chunks]
    books = Counter(chunk.book for chunk in chunks)
    docs = Counter(chunk.doc_id for chunk in chunks)

    print("=== Knowledge Base Summary ===")
    print(f"Total chunks      : {len(chunks)}")
    print(f"Unique books      : {len(books)}")
    print(f"Unique documents  : {len(docs)}")
    print(
        "Chunk length (chars) "
        f"min={min(lengths)}  "
        f"median={int(statistics.median(lengths))}  "
        f"mean={int(statistics.mean(lengths))}  "
        f"max={max(lengths)}"
    )
    print()
    print("Top books by chunk count:")
    for book, count in books.most_common(10):
        print(f"  {book:<25} {count:>5}")


def print_samples(chunks: list[StudyChunk], sample_size: int) -> None:
    if sample_size <= 0 or not chunks:
        return

    print("\n=== Sample Chunks ===")
    picks = random.sample(chunks, min(sample_size, len(chunks)))
    for idx, chunk in enumerate(picks, start=1):
        header = (
            f"[{idx}] {chunk.book} / {chunk.section} "
            f"(doc={chunk.doc_id}, chunk_index={chunk.chunk_index})"
        )
        border = "-" * len(header)
        print(border)
        print(header)
        print(border)
        preview = chunk.text if len(chunk.text) <= 600 else chunk.text[:600] + " …"
        print(fill(preview, width=90))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the processed knowledge base JSONL file."
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "data" / "processed" / "bible_study_chunks.jsonl",
        help="Path to the knowledge base JSONL file (defaults to processed output).",
    )
    parser.add_argument(
        "--book",
        type=str,
        default=None,
        help="Filter results to a specific book name (case-insensitive).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Print N random sample chunks for manual inspection.",
    )
    args = parser.parse_args()

    chunks = load_chunks(args.path)
    filtered = filter_chunks(chunks, args.book)

    print_summary(filtered)
    print_samples(filtered, args.sample)


if __name__ == "__main__":
    main()
