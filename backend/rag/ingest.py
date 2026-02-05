import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


def extract_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: List[Tuple[int, str]] = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append((idx, text))
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        overlap = 0

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def iter_pdf_chunks(input_dir: Path, chunk_size: int, overlap: int) -> Iterable[Tuple[str, str, dict]]:
    for pdf_path in sorted(input_dir.rglob("*.pdf")):
        rel = pdf_path.relative_to(input_dir).as_posix().replace("/", "_")
        pages = extract_pdf_pages(pdf_path)
        for page_idx, page_text in pages:
            chunks = chunk_text(page_text, chunk_size, overlap)
            for chunk_idx, chunk in enumerate(chunks, start=1):
                doc_id = f"{rel}-p{page_idx + 1}-c{chunk_idx}"
                metadata = {
                    "source": str(pdf_path),
                    "page": page_idx + 1,
                    "chunk": chunk_idx,
                }
                yield doc_id, chunk, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Chroma with OpenAI embeddings.")
    parser.add_argument("--input", default="backend/data/pdfs", help="Folder containing PDF files")
    parser.add_argument("--persist", default="backend/data/chroma", help="Chroma persistence directory")
    parser.add_argument("--collection", default="documents", help="Chroma collection name")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    persist_dir = Path(args.persist)
    persist_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    chroma = chromadb.PersistentClient(path=str(persist_dir))
    collection = chroma.get_or_create_collection(name=args.collection)

    items = list(iter_pdf_chunks(input_dir, args.chunk_size, args.chunk_overlap))
    if not items:
        print("No PDF text found to ingest.")
        return

    total = len(items)
    for i in range(0, total, args.batch_size):
        batch = items[i : i + args.batch_size]
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metadatas = [item[2] for item in batch]

        response = client.embeddings.create(model=args.model, input=texts)
        embeddings = [item.embedding for item in response.data]

        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        print(f"Ingested {min(i + args.batch_size, total)}/{total}")


if __name__ == "__main__":
    main()
