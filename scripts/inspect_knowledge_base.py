import json
from pathlib import Path
import argparse
import random
from collections import Counter

from yaf_gpt.data.StudyChunk import StudyChunk

# Rough skeleton
def load_chunks(path: Path) -> list[StudyChunk]:
    chunks = []
    with open(path) as fh:
        for line in fh:
            chunks.append(StudyChunk(**json.loads(line)))
    return chunks

def summarize(chunks: list[StudyChunk], book: str | None) -> dict:
    if book:
        chunks = [chunk for chunk in chunks if chunk.book == book]
    return {
        "total_chunks": len(chunks),
        "books": list(set(chunk.book for chunk in chunks)),
        "sections": list(set(chunk.section for chunk in chunks)),
    }

def sample_chunks(chunks: list[StudyChunk], n: int = 5) -> list[StudyChunk]:
    return random.sample(chunks, min(n, len(chunks)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect knowledge base chunks.")
    parser.add_argument("path", type=Path, help="Path to the knowledge base JSONL file.")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--book", type=str, default=None, help="Filter summary to a specific book.")
    args = parser.parse_args()

    chunks = load_chunks(args.path)
    if args.book:
        chunks = [chunk for chunk in chunks if chunk.book == args.book]
    summary = summarize(chunks, args.book)

    counter = Counter(chunk.book for chunk in chunks)
    print("Chunk counts by book:")
    for book, count in counter.items():
        print(f"  {book}: {count}")


    if args.sample > 0:
        chunks = sample_chunks(chunks, args.sample)

    print("Knowledge Base Summary:")
    print(f"Total Chunks: {summary['total_chunks']}")
    print(f"Books: {', '.join(summary['books'])}")
    print(f"Sections: {', '.join(summary['sections'])}")