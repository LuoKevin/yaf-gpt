from __future__ import annotations

import html
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .style_guide import REQUIRED_STUDY_PLAN_SECTIONS, STUDY_DOC_HEADING_ALIASES

DEFAULT_LUKE_DOC_ROOT = "backend/data/study_docx/Luke"
DEFAULT_TOP_K = 3

_QUESTION_PROMPT_PREFIXES = (
    "discuss",
    "share",
    "reflect",
    "review",
    "consider",
    "compare",
    "talk about",
)
_EXPLICIT_REFERENCE_RE = re.compile(
    r"\bLuke\s+(\d+)\s*[:._]\s*(\d+)(?:\s*[-–]\s*(?:(\d+)\s*[:._]\s*)?(\d+))?\b",
    re.IGNORECASE,
)
_FILENAME_REFERENCE_RE = re.compile(
    r"luke\s*(\d+)\s*[_.:]\s*(\d+)\s*-\s*(?:(\d+)\s*[_.:]\s*)?(\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LukeStructureExample:
    source_path: str
    normalized_reference: Optional[str]
    start_chapter: Optional[int]
    start_verse: Optional[int]
    end_chapter: Optional[int]
    end_verse: Optional[int]
    section_order: list[str]
    question_count: int
    has_ice_breaker: bool
    has_leader_notes: bool
    context_points: list[str]
    discussion_questions: list[str]


@dataclass(frozen=True)
class LukeStructureContext:
    examples: list[LukeStructureExample]
    recommended_section_order: list[str]
    typical_question_count: int
    ice_breaker_common: bool

    @classmethod
    def from_examples(cls, examples: list[LukeStructureExample]) -> LukeStructureContext:
        question_counts = [example.question_count for example in examples if example.question_count > 0]
        if question_counts:
            typical_question_count = Counter(question_counts).most_common(1)[0][0]
        else:
            typical_question_count = 6

        ice_breaker_common = bool(examples) and sum(
            1 for example in examples if example.has_ice_breaker
        ) * 2 >= len(examples)

        section_positions: dict[str, list[int]] = defaultdict(list)
        for example in examples:
            for idx, section in enumerate(example.section_order):
                if section == "Leader Notes":
                    continue
                section_positions[section].append(idx)

        recommended_section_order = _ordered_sections(section_positions)
        return cls(
            examples=examples,
            recommended_section_order=recommended_section_order,
            typical_question_count=typical_question_count,
            ice_breaker_common=ice_breaker_common,
        )

    def to_instruction_block(self) -> str:
        exemplar_refs = ", ".join(
            example.normalized_reference or Path(example.source_path).stem for example in self.examples
        )
        lines = [
            "Use these nearby Luke study-doc structure exemplars:",
            f"- Retrieved exemplar references: {exemplar_refs}",
            f"- Recommended section order from nearby docs: {', '.join(self.recommended_section_order)}",
            f"- Typical discussion question count in nearby docs: {self.typical_question_count}",
            f"- Ice breaker common in nearby docs: {'yes' if self.ice_breaker_common else 'no'}",
        ]

        for example in self.examples:
            reference_label = example.normalized_reference or Path(example.source_path).stem
            if example.context_points:
                lines.append(
                    f"- {reference_label} context style sample: "
                    f"{_truncate_text(example.context_points[0], limit=140)}"
                )
            for idx, question in enumerate(example.discussion_questions[:2], start=1):
                lines.append(
                    f"- {reference_label} question style sample {idx}: "
                    f"{_truncate_text(question, limit=160)}"
                )

        lines.append(
            "- Use these as format/style examples only; do not copy wording or import content not grounded in the requested passage."
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class LukeStructureCorpus:
    examples: list[LukeStructureExample]
    section_frequency: dict[str, int]
    canonical_sections: list[str]


@dataclass(frozen=True)
class _LukeReference:
    normalized_reference: str
    start_chapter: int
    start_verse: int
    end_chapter: int
    end_verse: int

    @property
    def start_index(self) -> int:
        return self.start_chapter * 1000 + self.start_verse

    @property
    def end_index(self) -> int:
        return self.end_chapter * 1000 + self.end_verse


def parse_luke_reference(text: str) -> Optional[str]:
    parsed = _parse_luke_reference(text)
    return parsed.normalized_reference if parsed else None


def parse_luke_reference_from_filename(filename: str) -> Optional[str]:
    parsed = _parse_luke_reference_from_filename(filename)
    return parsed.normalized_reference if parsed else None


def parse_luke_structure_doc(path: Path) -> LukeStructureExample:
    lines = _extract_lines_from_docx(path)
    sections, section_order = _extract_sections(lines)
    reference = _first_explicit_reference(lines) or _parse_luke_reference_from_filename(path.name)
    context_points = _extract_context_points(sections.get("Context", []))
    discussion_questions = _extract_discussion_questions(sections.get("Questions", []))

    return LukeStructureExample(
        source_path=str(path),
        normalized_reference=reference.normalized_reference if reference else None,
        start_chapter=reference.start_chapter if reference else None,
        start_verse=reference.start_verse if reference else None,
        end_chapter=reference.end_chapter if reference else None,
        end_verse=reference.end_verse if reference else None,
        section_order=section_order,
        question_count=len(discussion_questions),
        has_ice_breaker="Ice Breaker" in section_order,
        has_leader_notes="Leader Notes" in section_order,
        context_points=context_points,
        discussion_questions=discussion_questions,
    )


@lru_cache(maxsize=1)
def load_luke_structure_corpus(doc_root: str = DEFAULT_LUKE_DOC_ROOT) -> LukeStructureCorpus:
    folder = Path(doc_root)
    examples: list[LukeStructureExample] = []
    section_frequency: Counter[str] = Counter()
    section_positions: dict[str, list[int]] = defaultdict(list)

    for docx_path in sorted(folder.glob("*.docx")):
        try:
            example = parse_luke_structure_doc(docx_path)
        except (FileNotFoundError, zipfile.BadZipFile, KeyError):
            continue

        examples.append(example)
        seen_sections: set[str] = set()
        for idx, section in enumerate(example.section_order):
            if section in seen_sections:
                continue
            section_frequency[section] += 1
            section_positions[section].append(idx)
            seen_sections.add(section)

    return LukeStructureCorpus(
        examples=examples,
        section_frequency=dict(section_frequency),
        canonical_sections=_ordered_sections(section_positions),
    )


class LukeStructureRetriever:
    def __init__(
        self,
        *,
        doc_root: str = DEFAULT_LUKE_DOC_ROOT,
        top_k: int = DEFAULT_TOP_K,
        examples: Optional[list[LukeStructureExample]] = None,
    ) -> None:
        self._doc_root = doc_root
        self._top_k = top_k
        self._examples = examples

    def retrieve(self, reference: str) -> Optional[LukeStructureContext]:
        query = _parse_luke_reference(reference)
        if not query:
            return None

        corpus = (
            _build_corpus_from_examples(self._examples)
            if self._examples is not None
            else load_luke_structure_corpus(self._doc_root)
        )
        ranked_examples = sorted(
            (
                (_score_example(example, query), example)
                for example in corpus.examples
                if example.normalized_reference is not None
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        top_examples = [example for _, example in ranked_examples[: self._top_k]]
        if not top_examples:
            return None
        return LukeStructureContext.from_examples(top_examples)


def _build_corpus_from_examples(examples: list[LukeStructureExample]) -> LukeStructureCorpus:
    section_frequency: Counter[str] = Counter()
    section_positions: dict[str, list[int]] = defaultdict(list)

    for example in examples:
        seen_sections: set[str] = set()
        for idx, section in enumerate(example.section_order):
            if section in seen_sections:
                continue
            section_frequency[section] += 1
            section_positions[section].append(idx)
            seen_sections.add(section)

    return LukeStructureCorpus(
        examples=examples,
        section_frequency=dict(section_frequency),
        canonical_sections=_ordered_sections(section_positions),
    )


def _extract_lines_from_docx(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", "\n", xml)
    lines: list[str] = []
    for raw_line in text.splitlines():
        normalized = _normalize_text(raw_line)
        if normalized:
            lines.append(normalized)
    return lines


def _normalize_heading(line: str) -> Optional[str]:
    cleaned = re.sub(r"[:/]+$", "", line.strip().lower())
    return STUDY_DOC_HEADING_ALIASES.get(cleaned)


def _extract_sections(lines: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    sections: dict[str, list[str]] = defaultdict(list)
    section_order: list[str] = []
    preamble: list[str] = []
    current_section: Optional[str] = None

    for line in lines:
        heading = _normalize_heading(line)
        if heading:
            current_section = heading
            if heading not in section_order:
                section_order.append(heading)
            continue

        if current_section is None:
            preamble.append(line)
            continue

        sections[current_section].append(line)

    if preamble:
        sections["Passage"] = preamble + sections.get("Passage", [])
        if "Passage" not in section_order:
            section_order.insert(0, "Passage")

    return dict(sections), section_order


def _extract_context_points(lines: list[str], *, limit: int = 6) -> list[str]:
    context_points: list[str] = []
    for line in lines:
        if len(context_points) >= limit:
            break
        if "?" in line:
            continue
        if len(line) < 5 or len(line) > 180:
            continue
        if sum(char.isalpha() for char in line) < 3:
            continue
        context_points.append(line)
    return context_points


def _extract_discussion_questions(lines: list[str], *, limit: int = 8) -> list[str]:
    questions: list[str] = []
    for line in lines:
        if len(questions) >= limit:
            break
        if _is_discussion_question(line) and line not in questions:
            questions.append(line)
    return questions


def _is_discussion_question(line: str) -> bool:
    lowered = line.lower()
    if lowered.startswith("ice breaker"):
        return False
    if "?" in line:
        return True
    if len(line) > 120:
        return False
    return lowered.startswith(_QUESTION_PROMPT_PREFIXES)


def _normalize_text(text: str) -> str:
    return " ".join(html.unescape(text).strip().split())


def _truncate_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _first_explicit_reference(lines: list[str]) -> Optional[_LukeReference]:
    for line in lines:
        parsed = _parse_luke_reference(line)
        if parsed:
            return parsed
    return None


def _parse_luke_reference(text: str) -> Optional[_LukeReference]:
    match = _EXPLICIT_REFERENCE_RE.search(text)
    if not match:
        return None
    return _reference_from_groups(match.groups())


def _parse_luke_reference_from_filename(filename: str) -> Optional[_LukeReference]:
    stem = Path(filename).stem
    match = _FILENAME_REFERENCE_RE.search(stem)
    if not match:
        return None
    return _reference_from_groups(match.groups())


def _reference_from_groups(groups: tuple[str | None, ...]) -> _LukeReference:
    start_chapter = int(groups[0])
    start_verse = int(groups[1])
    end_chapter = int(groups[2]) if groups[2] is not None else start_chapter
    end_verse = int(groups[3]) if groups[3] is not None else start_verse
    normalized_reference = _format_reference(
        start_chapter=start_chapter,
        start_verse=start_verse,
        end_chapter=end_chapter,
        end_verse=end_verse,
    )
    return _LukeReference(
        normalized_reference=normalized_reference,
        start_chapter=start_chapter,
        start_verse=start_verse,
        end_chapter=end_chapter,
        end_verse=end_verse,
    )


def _format_reference(
    *, start_chapter: int, start_verse: int, end_chapter: int, end_verse: int
) -> str:
    if start_chapter == end_chapter and start_verse == end_verse:
        return f"Luke {start_chapter}:{start_verse}"
    if start_chapter == end_chapter:
        return f"Luke {start_chapter}:{start_verse}-{end_verse}"
    return f"Luke {start_chapter}:{start_verse}-{end_chapter}:{end_verse}"


def _ordered_sections(section_positions: dict[str, list[int]]) -> list[str]:
    if not section_positions:
        return list(REQUIRED_STUDY_PLAN_SECTIONS)

    ordered = [
        section
        for section, _ in sorted(
            section_positions.items(),
            key=lambda item: (
                sum(item[1]) / len(item[1]) if item[1] else 10_000,
                item[0],
            ),
        )
        if section != "Leader Notes"
    ]
    for section in REQUIRED_STUDY_PLAN_SECTIONS:
        if section not in ordered:
            ordered.append(section)
    return ordered


def _score_example(example: LukeStructureExample, query: _LukeReference) -> tuple[int, int, int, int, int, int, int]:
    if (
        example.start_chapter is None
        or example.start_verse is None
        or example.end_chapter is None
        or example.end_verse is None
        or example.normalized_reference is None
    ):
        return (-1, -1, -1, -1, -999, -1, -999)

    exact_match = int(example.normalized_reference == query.normalized_reference)
    example_start = example.start_chapter * 1000 + example.start_verse
    example_end = example.end_chapter * 1000 + example.end_verse
    overlap_units = max(0, min(example_end, query.end_index) - max(example_start, query.start_index) + 1)
    has_overlap = int(overlap_units > 0)
    same_chapter = int(_shares_any_chapter(example, query))
    chapter_distance = _chapter_distance(example, query)
    content_bonus = int(bool(example.context_points and example.discussion_questions))
    question_count_bonus = -abs(example.question_count - 6)
    return (
        exact_match,
        has_overlap,
        overlap_units,
        same_chapter,
        -chapter_distance,
        content_bonus,
        question_count_bonus,
    )


def _shares_any_chapter(example: LukeStructureExample, query: _LukeReference) -> bool:
    if example.start_chapter is None or example.end_chapter is None:
        return False
    example_chapters = set(range(example.start_chapter, example.end_chapter + 1))
    query_chapters = set(range(query.start_chapter, query.end_chapter + 1))
    return bool(example_chapters & query_chapters)


def _chapter_distance(example: LukeStructureExample, query: _LukeReference) -> int:
    if example.start_chapter is None or example.end_chapter is None:
        return 999
    candidate_distances = [
        abs(example.start_chapter - query.start_chapter),
        abs(example.start_chapter - query.end_chapter),
        abs(example.end_chapter - query.start_chapter),
        abs(example.end_chapter - query.end_chapter),
    ]
    return min(candidate_distances)
