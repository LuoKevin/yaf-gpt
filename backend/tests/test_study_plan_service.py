from __future__ import annotations

import unittest

from backend.app.schemas import StudyPlanRequest
from backend.llm.provider import ChatResponse
from backend.services.bible_lookup import PassageData, PassageVerse
from backend.services.study_docx_structure import LukeStructureContext, LukeStructureExample
from backend.services.study_plan_service import StudyPlanService, StudyPlanValidationError


def _valid_output_json() -> str:
    questions = [f"Question {i}" for i in range(1, 7)]
    reflection = ["What is one lesson from this passage you need to apply this week?"]
    return (
        "{"
        '"passage_title":"Jesus Foretells Turmoil and Hope",'
        '"context_points":["Temple context","Audience context"],'
        f'"discussion_questions":{questions!r},'
        f'"reflection_questions":{reflection!r}'
        "}"
    ).replace("'", '"')


class _FakeBibleProvider:
    def get_passage(self, reference: str, translation: str = "WEB") -> PassageData:
        return PassageData(
            reference=reference,
            normalized_reference=reference,
            translation=translation,
            text="Sample passage",
            verses=[PassageVerse(book="Luke", chapter=21, verse=5, text="Sample")],
        )


class _FakeChatProvider:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs
        self.calls = 0
        self.last_messages = None

    def generate(self, messages, *, model, temperature=0.2, max_tokens=None):
        output = self._outputs[self.calls]
        self.calls += 1
        self.last_messages = messages
        return ChatResponse(
            content=output,
            model=model,
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )

    def stream(self, messages, *, model, temperature=0.2, max_tokens=None):
        return []


class _FakeStructureRetriever:
    def __init__(self, context: LukeStructureContext | None) -> None:
        self._context = context
        self.calls: list[str] = []

    def retrieve(self, reference: str) -> LukeStructureContext | None:
        self.calls.append(reference)
        return self._context


class _RaisingStructureRetriever:
    def retrieve(self, reference: str) -> LukeStructureContext | None:
        raise RuntimeError("boom")


class StudyPlanServiceTests(unittest.TestCase):
    def test_retries_once_then_succeeds(self) -> None:
        chat_provider = _FakeChatProvider(["not-json", _valid_output_json()])
        service = StudyPlanService(
            bible_provider=_FakeBibleProvider(),
            chat_provider=chat_provider,
            model="gpt-4o-mini",
        )
        payload = StudyPlanRequest(reference="Luke 21:5-28", translation="WEB")
        response = service.generate_study_plan(payload)

        self.assertEqual(response.reference, "Luke 21:5-28")
        self.assertEqual(len(response.discussion_questions), 6)
        self.assertLessEqual(len(response.reflection_questions), 3)
        self.assertEqual(response.model, "gpt-4o-mini")
        self.assertIsNotNone(response.usage)
        self.assertIsNotNone(chat_provider.last_messages)

    def test_fails_after_second_invalid_output(self) -> None:
        service = StudyPlanService(
            bible_provider=_FakeBibleProvider(),
            chat_provider=_FakeChatProvider(["bad", "still bad"]),
            model="gpt-4o-mini",
        )
        payload = StudyPlanRequest(reference="Luke 21:5-28", translation="WEB")
        with self.assertRaises(StudyPlanValidationError):
            service.generate_study_plan(payload)

    def test_includes_structure_context_in_prompt_when_available(self) -> None:
        chat_provider = _FakeChatProvider([_valid_output_json()])
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
        retriever = _FakeStructureRetriever(structure_context)
        service = StudyPlanService(
            bible_provider=_FakeBibleProvider(),
            chat_provider=chat_provider,
            structure_retriever=retriever,
            model="gpt-4o-mini",
        )

        response = service.generate_study_plan(StudyPlanRequest(reference="Luke 21:5-28", translation="WEB"))

        self.assertEqual(response.reference, "Luke 21:5-28")
        self.assertEqual(retriever.calls, ["Luke 21:5-28"])
        self.assertIsNotNone(chat_provider.last_messages)
        user_prompt = chat_provider.last_messages[1].content
        self.assertIn("Retrieved exemplar references: Luke 21:5-28", user_prompt)
        self.assertIn("Temple was central to Jewish life.", user_prompt)

    def test_structure_retrieval_failure_falls_back_without_breaking_generation(self) -> None:
        chat_provider = _FakeChatProvider([_valid_output_json()])
        service = StudyPlanService(
            bible_provider=_FakeBibleProvider(),
            chat_provider=chat_provider,
            structure_retriever=_RaisingStructureRetriever(),
            model="gpt-4o-mini",
        )

        with self.assertLogs("backend.services.study_plan_service", level="WARNING") as logs:
            response = service.generate_study_plan(
                StudyPlanRequest(reference="Luke 21:5-28", translation="WEB")
            )

        self.assertEqual(response.reference, "Luke 21:5-28")
        self.assertIsNotNone(chat_provider.last_messages)
        self.assertTrue(any("Failed to retrieve Luke structure exemplars" in line for line in logs.output))
        user_prompt = chat_provider.last_messages[1].content
        self.assertNotIn("Retrieved exemplar references:", user_prompt)


if __name__ == "__main__":
    unittest.main()
