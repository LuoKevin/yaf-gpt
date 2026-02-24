from __future__ import annotations

import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class LukeStyleGuide:
    section_frequency: dict[str, int]
    canonical_sections: list[str]

    def to_instruction_block(self) -> str:
        sections = [section for section in self.canonical_sections if section != "Leader Notes"]
        ordered = ", ".join(sections)
        frequencies = ", ".join(
            f"{section}: {count}"
            for section, count in sorted(self.section_frequency.items())
            if section != "Leader Notes"
        )
        return (
            "Use this section structure inspired by the Luke study docs:\n"
            f"- Preferred section order: {ordered}\n"
            f"- Observed heading frequency across Luke docs: {frequencies}\n"
            "- Always include Passage, Context, and Questions."
        )


_HEADING_ALIASES = {
    "passage": "Passage",
    "context": "Context",
    "background": "Context",
    "questions": "Questions",
    "notes": "Leader Notes",
    "care group": "Leader Notes",
    "pre-notes": "Leader Notes",
    "ice breaker": "Ice Breaker",
    "icebreaker": "Ice Breaker",
}

_REQUIRED_SECTIONS = ["Passage", "Context", "Questions"]


def _extract_lines_from_docx(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", "\n", xml)
    return [line.strip() for line in text.splitlines() if line.strip()]


def _normalize_heading(line: str) -> str | None:
    cleaned = re.sub(r"[:/]+$", "", line.strip().lower())
    return _HEADING_ALIASES.get(cleaned)


@lru_cache(maxsize=1)
def load_luke_style_guide(doc_root: str = "backend/data/study_docx/Luke") -> LukeStyleGuide:
    folder = Path(doc_root)
    counts: Counter[str] = Counter()
    first_positions: dict[str, list[int]] = defaultdict(list)

    for docx_path in sorted(folder.glob("*.docx")):
        try:
            lines = _extract_lines_from_docx(docx_path)
        except (FileNotFoundError, zipfile.BadZipFile, KeyError):
            continue

        seen_in_doc: set[str] = set()
        for idx, line in enumerate(lines):
            heading = _normalize_heading(line)
            if not heading:
                continue
            if heading not in seen_in_doc:
                counts[heading] += 1
                first_positions[heading].append(idx)
                seen_in_doc.add(heading)

    if not counts:
        return LukeStyleGuide(
            section_frequency={section: 0 for section in _REQUIRED_SECTIONS},
            canonical_sections=list(_REQUIRED_SECTIONS),
        )

    ordered_by_position = sorted(
        counts.keys(),
        key=lambda key: (
            sum(first_positions[key]) / len(first_positions[key]) if first_positions[key] else 10_000,
            key,
        ),
    )

    canonical_sections: list[str] = []
    for section in ordered_by_position:
        if section == "Leader Notes":
            continue
        if section not in canonical_sections:
            canonical_sections.append(section)
    for section in _REQUIRED_SECTIONS:
        if section not in canonical_sections:
            canonical_sections.append(section)

    return LukeStyleGuide(section_frequency=dict(counts), canonical_sections=canonical_sections)
