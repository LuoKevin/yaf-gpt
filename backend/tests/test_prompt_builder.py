from __future__ import annotations

import unittest

from backend.services.study_plan import (
    LukeStructureContext,
    LukeStructureExample,
    LukeStyleGuide,
    build_study_plan_messages,
)


class PromptBuilderTests(unittest.TestCase):
    def test_includes_required_sections_and_question_count(self) -> None:
        style = LukeStyleGuide(
            section_frequency={"Passage": 40, "Context": 39, "Questions": 39},
            canonical_sections=["Passage", "Context", "Questions"],
        )
        messages = build_study_plan_messages(
            reference="Luke 21:5-28",
            normalized_reference="Luke 21:5-28",
            translation="WEB",
            passage_text="Sample passage text.",
            style_guide=style,
            structure_context=None,
            goals=None,
            user_notes=None,
            include_question_notes=False,
        )

        self.assertEqual(len(messages), 2)
        user_msg = messages[1].content
        self.assertIn("Passage", user_msg)
        self.assertIn("Context", user_msg)
        self.assertIn("Questions", user_msg)
        self.assertIn("exactly 6 discussion questions", user_msg)
        self.assertIn("discussion_questions", user_msg)
        self.assertIn("reflection_questions", user_msg)
        self.assertIn("Include 1 to 3 reflection questions", user_msg)
        self.assertIn("60-minute study", user_msg)
        self.assertIn("3-5 participants plus 1 discussion leader", user_msg)
        self.assertIn("follow the passage flow from beginning to end", user_msg)
        self.assertIn("open-ended, text-anchored", user_msg)
        self.assertIn("plain question strings only", user_msg)
        self.assertIn("discussion_questions must focus solely on understanding and discussing the passage text", user_msg)
        self.assertIn("reflection_questions must be at the end", user_msg)
        self.assertIn("must directly anchor to this passage", user_msg)
        self.assertIn("Avoid generic reflection prompts", user_msg)
        self.assertIn("Omit discussion_question_notes and reflection_question_notes", user_msg)

    def test_includes_structure_rag_block_when_available(self) -> None:
        style = LukeStyleGuide(
            section_frequency={"Passage": 40, "Context": 39, "Questions": 39},
            canonical_sections=["Passage", "Context", "Questions"],
        )
        structure_context = LukeStructureContext.from_examples(
            [
                LukeStructureExample(
                    source_path="backend/data/study_docx/Luke/Luke 21_5-28.docx",
                    normalized_reference="Luke 21:5-28",
                    start_chapter=21,
                    start_verse=5,
                    end_chapter=21,
                    end_verse=28,
                    section_order=["Passage", "Context", "Questions"],
                    question_count=6,
                    has_ice_breaker=False,
                    has_leader_notes=True,
                    context_points=["Temple was central to Jewish life."],
                    discussion_questions=[
                        "What does Jesus say will happen to the temple?",
                        "How should disciples respond to turmoil?",
                    ],
                )
            ]
        )
        messages = build_study_plan_messages(
            reference="Luke 21:5-28",
            normalized_reference="Luke 21:5-28",
            translation="WEB",
            passage_text="Sample passage text.",
            style_guide=style,
            structure_context=structure_context,
            goals=None,
            user_notes=None,
            include_question_notes=False,
        )

        user_msg = messages[1].content
        self.assertIn("Retrieved exemplar references: Luke 21:5-28", user_msg)
        self.assertIn("Typical discussion question count in nearby docs: 6", user_msg)
        self.assertIn("Temple was central to Jewish life.", user_msg)
        self.assertIn("What does Jesus say will happen to the temple?", user_msg)
        self.assertIn("Use these as format/style examples only", user_msg)

    def test_includes_question_notes_instruction_when_enabled(self) -> None:
        style = LukeStyleGuide(
            section_frequency={"Passage": 40, "Context": 39, "Questions": 39},
            canonical_sections=["Passage", "Context", "Questions"],
        )
        messages = build_study_plan_messages(
            reference="Luke 21:5-28",
            normalized_reference="Luke 21:5-28",
            translation="WEB",
            passage_text="Sample passage text.",
            style_guide=style,
            structure_context=None,
            goals=None,
            user_notes=None,
            include_question_notes=True,
        )

        user_msg = messages[1].content
        self.assertIn("Include discussion_question_notes", user_msg)
        self.assertIn("Include reflection_question_notes", user_msg)
        self.assertIn("leader-facing facilitation hint", user_msg)


if __name__ == "__main__":
    unittest.main()
