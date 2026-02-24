from __future__ import annotations

import unittest

from backend.app.schemas import StudyPlanRequest
from backend.llm.provider import ChatResponse
from backend.services.bible_lookup import PassageData, PassageVerse
from backend.services.study_plan_service import StudyPlanService, StudyPlanValidationError


def _valid_output_json() -> str:
    questions = [
        {
            "question": f"Question {i}",
            "intent": "Drive observation and interpretation.",
            "follow_up": "What in the text supports your answer?",
        }
        for i in range(1, 7)
    ]
    return (
        "{"
        '"passage_title":"Jesus Foretells Turmoil and Hope",'
        '"context_points":["Temple context","Audience context"],'
        f'"discussion_questions":{questions!r}'
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

    def generate(self, messages, *, model, temperature=0.2, max_tokens=None):
        output = self._outputs[self.calls]
        self.calls += 1
        return ChatResponse(
            content=output,
            model=model,
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )

    def stream(self, messages, *, model, temperature=0.2, max_tokens=None):
        return []


class StudyPlanServiceTests(unittest.TestCase):
    def test_retries_once_then_succeeds(self) -> None:
        service = StudyPlanService(
            bible_provider=_FakeBibleProvider(),
            chat_provider=_FakeChatProvider(["not-json", _valid_output_json()]),
            model="gpt-4o-mini",
        )
        payload = StudyPlanRequest(reference="Luke 21:5-28", translation="WEB")
        response = service.generate_study_plan(payload)

        self.assertEqual(response.reference, "Luke 21:5-28")
        self.assertEqual(len(response.discussion_questions), 6)
        self.assertEqual(response.model, "gpt-4o-mini")
        self.assertIsNotNone(response.usage)

    def test_fails_after_second_invalid_output(self) -> None:
        service = StudyPlanService(
            bible_provider=_FakeBibleProvider(),
            chat_provider=_FakeChatProvider(["bad", "still bad"]),
            model="gpt-4o-mini",
        )
        payload = StudyPlanRequest(reference="Luke 21:5-28", translation="WEB")
        with self.assertRaises(StudyPlanValidationError):
            service.generate_study_plan(payload)


if __name__ == "__main__":
    unittest.main()
