from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from backend.services.study_plan import (
    LukeStructureExample,
    LukeStructureRetriever,
    parse_luke_reference_from_filename,
    parse_luke_structure_doc,
)


def _write_docx(path: Path, lines: list[str]) -> None:
    xml_lines = "".join(f"<w:p><w:r><w:t>{line}</w:t></w:r></w:p>" for line in lines)
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{xml_lines}</w:body>"
        "</w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)


class StudyDocxStructureTests(unittest.TestCase):
    def test_parse_doc_extracts_sections_and_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_path = Path(tmpdir) / "Luke 21_5-28.docx"
            _write_docx(
                doc_path,
                [
                    "Luke 21:5-28",
                    "Some of his disciples were remarking about the temple.",
                    "Context",
                    "Temple was central to Jewish life.",
                    "Herod expanded the temple complex.",
                    "Questions",
                    "What does Jesus say will happen to the temple?",
                    "Discuss one way this passage encourages perseverance.",
                    "Notes",
                    "Leader note that should not appear in prompt snippets.",
                ],
            )

            example = parse_luke_structure_doc(doc_path)

        self.assertEqual(example.normalized_reference, "Luke 21:5-28")
        self.assertEqual(example.section_order, ["Passage", "Context", "Questions", "Leader Notes"])
        self.assertEqual(example.question_count, 2)
        self.assertFalse(example.has_ice_breaker)
        self.assertTrue(example.has_leader_notes)
        self.assertEqual(
            example.context_points,
            ["Temple was central to Jewish life.", "Herod expanded the temple complex."],
        )
        self.assertEqual(
            example.discussion_questions,
            [
                "What does Jesus say will happen to the temple?",
                "Discuss one way this passage encourages perseverance.",
            ],
        )

    def test_filename_reference_normalization_handles_cross_chapter_formats(self) -> None:
        self.assertEqual(
            parse_luke_reference_from_filename("Luke 19_45-20_8.docx"),
            "Luke 19:45-20:8",
        )
        self.assertEqual(
            parse_luke_reference_from_filename("Luke 5.33-6.5.docx"),
            "Luke 5:33-6:5",
        )

    def test_retriever_prefers_exact_match_then_nearby_examples(self) -> None:
        examples = [
            LukeStructureExample(
                source_path="Luke 21_5-28.docx",
                normalized_reference="Luke 21:5-28",
                start_chapter=21,
                start_verse=5,
                end_chapter=21,
                end_verse=28,
                section_order=["Passage", "Context", "Questions"],
                question_count=6,
                has_ice_breaker=False,
                has_leader_notes=True,
                context_points=["Temple background"],
                discussion_questions=["What does Jesus say will happen to the temple?"],
            ),
            LukeStructureExample(
                source_path="Luke 21_1-23.docx",
                normalized_reference="Luke 21:1-23",
                start_chapter=21,
                start_verse=1,
                end_chapter=21,
                end_verse=23,
                section_order=["Passage", "Context", "Questions"],
                question_count=6,
                has_ice_breaker=False,
                has_leader_notes=False,
                context_points=["Temple treasury context"],
                discussion_questions=["How do the disciples respond to Jesus?"],
            ),
            LukeStructureExample(
                source_path="Luke 20_9-26.docx",
                normalized_reference="Luke 20:9-26",
                start_chapter=20,
                start_verse=9,
                end_chapter=20,
                end_verse=26,
                section_order=["Passage", "Context", "Questions"],
                question_count=6,
                has_ice_breaker=True,
                has_leader_notes=False,
                context_points=["Conflict with religious leaders"],
                discussion_questions=["How does Jesus answer his opponents?"],
            ),
            LukeStructureExample(
                source_path="Luke 10_25-42.docx",
                normalized_reference="Luke 10:25-42",
                start_chapter=10,
                start_verse=25,
                end_chapter=10,
                end_verse=42,
                section_order=["Ice Breaker", "Passage", "Context", "Questions"],
                question_count=6,
                has_ice_breaker=True,
                has_leader_notes=True,
                context_points=["Samaritan background"],
                discussion_questions=["What does Jesus teach about neighbor love?"],
            ),
        ]
        retriever = LukeStructureRetriever(examples=examples, top_k=3)

        context = retriever.retrieve("Luke 21:5-28")

        self.assertIsNotNone(context)
        self.assertEqual(
            [example.normalized_reference for example in context.examples],
            ["Luke 21:5-28", "Luke 21:1-23", "Luke 20:9-26"],
        )

    def test_non_luke_reference_returns_none(self) -> None:
        retriever = LukeStructureRetriever(
            examples=[
                LukeStructureExample(
                    source_path="Luke 21_5-28.docx",
                    normalized_reference="Luke 21:5-28",
                    start_chapter=21,
                    start_verse=5,
                    end_chapter=21,
                    end_verse=28,
                    section_order=["Passage", "Context", "Questions"],
                    question_count=6,
                    has_ice_breaker=False,
                    has_leader_notes=False,
                    context_points=["Temple background"],
                    discussion_questions=["What does Jesus say will happen to the temple?"],
                )
            ]
        )

        self.assertIsNone(retriever.retrieve("John 3:16"))


if __name__ == "__main__":
    unittest.main()
