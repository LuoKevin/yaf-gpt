from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routes.bible import get_bible_provider
from backend.app.routes.chat import get_persona_chat_service
from backend.app.routes.hymn import get_hymn_service
from backend.app.routes.image import get_passage_image_service
from backend.app.routes.study_plan import get_study_plan_service
from backend.app.schemas import (
    HymnGenerateResponse,
    HymnJobResponse,
    HymnLyrics,
    HymnSection,
    PassageImageResponse,
    PersonaChatResponse,
    StudyPlanResponse,
    UsageMetrics,
)
from backend.services.bible_lookup import (
    InvalidReferenceError,
    PassageData,
    PassageNotFoundError,
    PassageVerse,
)
from backend.services.hymn_service import HymnJobNotFoundError, HymnValidationError
from backend.services.persona_chat_service import PersonaChatValidationError
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
            reflection_questions=["How should this passage shape your week?"],
            model="gpt-4o-mini",
            usage=UsageMetrics(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )


class _PassageImageServiceStub:
    def generate_passage_image(self, payload):
        if payload.reference == "Bad 1:1":
            raise InvalidReferenceError("bad ref")
        return PassageImageResponse(
            reference=payload.reference,
            translation=payload.translation,
            style=payload.style,
            prompt_used="Prompt",
            image_b64_or_url="https://example.com/image.png",
            alt_text="Sample alt",
        )


class _PersonaChatServiceStub:
    def create_reply(self, payload):
        if payload.messages and payload.messages[0].content == "bad":
            raise PersonaChatValidationError("invalid")
        return PersonaChatResponse(
            reply="Sample persona response",
            model="gpt-4o-mini",
            usage=UsageMetrics(prompt_tokens=5, completion_tokens=7, total_tokens=12),
        )


class _HymnServiceStub:
    def generate_hymn(self, payload):
        if payload.reference == "Bad 1:1":
            raise InvalidReferenceError("bad ref")
        if payload.reference == "Malformed 1:1":
            raise HymnValidationError("bad hymn output")
        return HymnGenerateResponse(
            reference=payload.reference,
            normalized_reference=payload.reference,
            translation=payload.translation,
            passage_text="Passage text",
            hymn=HymnLyrics(
                title="Hope in the Storm",
                theme="Trusting Christ through trial",
                scripture_references=["Luke 21:5-28"],
                sections=[
                    HymnSection(label="Verse 1", lyrics="Lift up your eyes in trial."),
                    HymnSection(label="Chorus", lyrics="Christ is our steadfast hope."),
                ],
            ),
            job_id="job-123",
            job_status="queued",
            provider="mock",
            model="gpt-4o-mini",
            usage=UsageMetrics(prompt_tokens=20, completion_tokens=40, total_tokens=60),
        )

    def get_job_status(self, job_id: str):
        if job_id == "missing":
            raise HymnJobNotFoundError("not found")
        return HymnJobResponse(
            job_id=job_id,
            status="completed",
            provider="mock",
            audio_url="https://example.com/mock.mp3",
            error=None,
        )


class APIRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[get_bible_provider] = lambda: _BibleProviderStub()
        app.dependency_overrides[get_study_plan_service] = lambda: _StudyPlanServiceStub()
        app.dependency_overrides[get_passage_image_service] = lambda: _PassageImageServiceStub()
        app.dependency_overrides[get_persona_chat_service] = lambda: _PersonaChatServiceStub()
        app.dependency_overrides[get_hymn_service] = lambda: _HymnServiceStub()
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
        self.assertLessEqual(len(body["reflection_questions"]), 3)
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

    def test_passage_image_success(self) -> None:
        response = self.client.post(
            "/api/passage-image",
            json={
                "reference": "Luke 21:5-28",
                "translation": "WEB",
                "style": "modern_editorial_illustration",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["style"], "modern_editorial_illustration")

    def test_passage_image_invalid_reference_maps_to_400(self) -> None:
        response = self.client.post(
            "/api/passage-image",
            json={
                "reference": "Bad 1:1",
                "translation": "WEB",
                "style": "modern_editorial_illustration",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_persona_chat_success(self) -> None:
        response = self.client.post(
            "/api/persona-chat",
            json={
                "messages": [{"role": "user", "content": "How should we apply this passage?"}],
                "reference_context": "Luke 21:5-28",
                "translation": "WEB",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("reply", body)

    def test_persona_chat_invalid_payload_maps_to_400(self) -> None:
        response = self.client.post(
            "/api/persona-chat",
            json={
                "messages": [{"role": "user", "content": "bad"}],
                "translation": "WEB",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_hymn_generate_success(self) -> None:
        response = self.client.post(
            "/api/hymn/generate",
            json={
                "reference": "Luke 21:5-28",
                "translation": "WEB",
                "style_hint": "modern worship hymn, acoustic",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["job_id"], "job-123")

    def test_hymn_generate_invalid_reference_maps_to_400(self) -> None:
        response = self.client.post(
            "/api/hymn/generate",
            json={
                "reference": "Bad 1:1",
                "translation": "WEB",
                "style_hint": "modern worship hymn, acoustic",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_hymn_generate_validation_error_maps_to_502(self) -> None:
        response = self.client.post(
            "/api/hymn/generate",
            json={
                "reference": "Malformed 1:1",
                "translation": "WEB",
                "style_hint": "modern worship hymn, acoustic",
            },
        )
        self.assertEqual(response.status_code, 502)

    def test_hymn_job_status_success(self) -> None:
        response = self.client.get("/api/hymn/jobs/job-123")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "completed")

    def test_hymn_job_not_found_maps_to_404(self) -> None:
        response = self.client.get("/api/hymn/jobs/missing")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
