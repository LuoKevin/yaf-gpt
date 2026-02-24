from __future__ import annotations

import unittest

from backend.services.prompt_builder import build_study_plan_messages
from backend.services.style_guide import LukeStyleGuide


class PromptBuilderTests(unittest.TestCase):
    def test_includes_required_sections_and_question_count(self) -> None:
        style = LukeStyleGuide(
            section_frequency={"Passage": 40, "Context": 39, "Questions": 39, "Leader Notes": 10},
            canonical_sections=["Passage", "Context", "Questions", "Leader Notes"],
        )
        messages = build_study_plan_messages(
            reference="Luke 21:5-28",
            normalized_reference="Luke 21:5-28",
            translation="WEB",
            passage_text="Sample passage text.",
            style_guide=style,
            goals=None,
            user_notes=None,
        )

        self.assertEqual(len(messages), 2)
        user_msg = messages[1].content
        self.assertIn("Passage", user_msg)
        self.assertIn("Context", user_msg)
        self.assertIn("Questions", user_msg)
        self.assertIn("Leader Notes", user_msg)
        self.assertIn("exactly 6 discussion questions", user_msg)
        self.assertIn("discussion_questions", user_msg)


if __name__ == "__main__":
    unittest.main()

