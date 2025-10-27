"""Extract structured text chunks from raw .docx study guides.

The script walks the ``raw_data`` directory, parses each Word document,
and emits a JSONL corpus at ``data/processed/bible_study_chunks.jsonl``.
It uses no third-party dependencies so it can run inside restricted
environments.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable
from pathlib import Path
from xml.etree import ElementTree as ET


DOCX_BODY_PATH = "word/document.xml"
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
STYLE_ATTR = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"


def iter_paragraphs(docx_path: Path) -> Iterable[tuple[str | None, str]]:
    """Yield (style, text) pairs for each non-empty paragraph in the docx."""
    with zipfile.ZipFile(docx_path) as archive:
        body = archive.read(DOCX_BODY_PATH)

    root = ET.fromstring(body)

    for paragraph in root.findall(".//w:body/w:p", WORD_NS):
        texts = [
            node.text for node in paragraph.findall(".//w:t", WORD_NS) if node.text
        ]
        if not texts:
            continue

        style_element = paragraph.find("w:pPr/w:pStyle", WORD_NS)
        style = style_element.get(STYLE_ATTR) if style_element is not None else None

        # Join text with spaces to avoid words being stuck together when Word
        # splits them across runs.
        text = " ".join(texts).strip()
        if text:
            yield style, text


def chunk_document(docx_path: Path, default_heading: str) -> list[dict[str, str | int]]:
    """Split a document into heading-bounded chunks."""
    chunks: list[dict[str, str | int]] = []
    current_heading = default_heading
    buffer: list[str] = []
    chunk_index = 0

    for style, text in iter_paragraphs(docx_path):
        if style and style.lower().startswith("heading"):
            if buffer:
                chunks.append(
                    {
                        "section": current_heading or default_heading,
                        "chunk_index": chunk_index,
                        "text": "\n".join(buffer).strip(),
                    }
                )
                chunk_index += 1
                buffer = []
            current_heading = text
        else:
            buffer.append(text)

    if buffer:
        chunks.append(
            {
                "section": current_heading or default_heading,
                "chunk_index": chunk_index,
                "text": "\n".join(buffer).strip(),
            }
        )

    return chunks


def build_corpus(raw_root: Path, output_path: Path) -> None:
    """Parse all docx files and write a JSONL corpus."""
    records = []
    for docx_path in sorted(raw_root.rglob("*.docx")):
        relative = docx_path.relative_to(raw_root)
        book = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        doc_id = docx_path.stem
        default_heading = doc_id.replace("_", " ")

        for chunk in chunk_document(docx_path, default_heading=default_heading):
            records.append(
                {
                    "book": book,
                    "doc_id": doc_id,
                    "section": chunk["section"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "source_path": str(relative),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "raw_data" / "Bible_Study_Docs"
    output_path = project_root / "data" / "processed" / "bible_study_chunks.jsonl"

    if not raw_root.exists():
        raise SystemExit(f"Raw data directory not found: {raw_root}")

    build_corpus(raw_root=raw_root, output_path=output_path)
    print(f"Wrote knowledge base to {output_path}")


if __name__ == "__main__":
    main()
