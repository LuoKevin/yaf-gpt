from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routes.bible import get_bible_provider
from backend.app.routes.study_plan import get_study_plan_service
from backend.app.schemas import StudyPlanResponse, UsageMetrics
from backend.services.bible_lookup import (
    InvalidReferenceError,
    PassageData,
    PassageNotFoundError,
    PassageVerse,
)
from backend.services.study_plan_service import StudyPlanValidationError


class _BibleProviderStub:
    def get_passage(self, reference: str, translation: str = "WEB") -> PassageData:
        if reference == "NotARef":
            raise InvalidReferenceError("bad ref")
        if reference == "Missing 1:1":
            raise PassageNotFoundError("not found")
        return PassageData(
            reference=reference,
            normalized_reference=reference,
            translation=translation,
            text="Passage text",
            verses=[PassageVerse(book="John", chapter=3, verse=16, text="For God so loved...")],
        )


class _StudyPlanServiceStub:
    def generate_study_plan(self, payload):
        if payload.reference == "Bad 1:1":
            raise InvalidReferenceError("invalid ref")
        if payload.reference == "Missing 1:1":
            raise PassageNotFoundError("not found")
        if payload.reference == "Malformed 1:1":
            raise StudyPlanValidationError("invalid model output")
        return StudyPlanResponse(
            reference=payload.reference,
            normalized_reference=payload.reference,
            translation=payload.translation,
            passage_text="Passage text",
            passage_title="Sample Title",
            context_points=["Point 1"],
            discussion_questions=[f"Q{i}" for i in range(1, 7)],
            model="gpt-4o-mini",
            usage=UsageMetrics(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )


class APIRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[get_bible_provider] = lambda: _BibleProviderStub()
        app.dependency_overrides[get_study_plan_service] = lambda: _StudyPlanServiceStub()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_bible_passage_success(self) -> None:
        response = self.client.get("/api/bible/passage", params={"reference": "John 3:16-18"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["normalized_reference"], "John 3:16-18")
        self.assertGreaterEqual(len(body["verses"]), 1)

    def test_study_plan_success(self) -> None:
        response = self.client.post(
            "/api/study-plan",
            json={"reference": "Luke 21:5-28", "translation": "WEB"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["discussion_questions"]), 6)
        self.assertEqual(body["model"], "gpt-4o-mini")

    def test_study_plan_invalid_reference_maps_to_400(self) -> None:
        response = self.client.post(
            "/api/study-plan",
            json={"reference": "Bad 1:1", "translation": "WEB"},
        )
        self.assertEqual(response.status_code, 400)

    def test_study_plan_not_found_maps_to_404(self) -> None:
        response = self.client.post(
            "/api/study-plan",
            json={"reference": "Missing 1:1", "translation": "WEB"},
        )
        self.assertEqual(response.status_code, 404)

    def test_study_plan_validation_error_maps_to_502(self) -> None:
        response = self.client.post(
            "/api/study-plan",
            json={"reference": "Malformed 1:1", "translation": "WEB"},
        )
        self.assertEqual(response.status_code, 502)


if __name__ == "__main__":
    unittest.main()
