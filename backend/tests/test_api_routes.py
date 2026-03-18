from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routes.bible import get_bible_provider
from backend.app.routes.chat import get_persona_chat_service
from backend.app.routes.image import get_passage_image_service
from backend.app.routes.music import get_music_generation_service
from backend.app.routes.study_plan import get_study_plan_service
from backend.app.routes.voice import (
    get_voice_generation_service,
    get_voice_realtime_session_service,
    get_voice_transcription_service,
)
from backend.app.schemas import (
    MusicGenerateResponse,
    MusicJobResponse,
    PassageImageResponse,
    PersonaChatResponse,
    StudyPlanResponse,
    UsageMetrics,
    VoiceGenerationResponse,
)
from backend.services.bible_lookup import (
    InvalidReferenceError,
    PassageData,
    PassageNotFoundError,
    PassageVerse,
)
from backend.services.persona_chat_service import PersonaChatProviderError, PersonaChatValidationError
from backend.services.study_plan_service import StudyPlanValidationError
from backend.services.voice_transcription import VoiceTranscriptionService


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

    def stream_reply(self, payload):
        if payload.messages and payload.messages[0].content == "bad":
            raise PersonaChatValidationError("invalid")
        if payload.messages and payload.messages[0].content == "provider":
            raise PersonaChatProviderError("provider down")
        return "gpt-4o-mini", iter(["Sample ", "persona response"])


class _VoiceTranscriptionServiceStub:
    model_name = "gpt-4o-mini-transcribe"

    def transcribe_base64(self, *, audio_base64: str, mime_type: str | None = None, file_name: str | None = None):
        if audio_base64 == "bad":
            raise ValueError("invalid")
        if audio_base64 == "provider":
            raise RuntimeError("provider down")
        return "transcribed question"


class _VoiceGenerationServiceStub:
    model_name = "gpt-4o-mini-tts"

    def generate_audio(
        self,
        *,
        input_text: str,
        voice: str = "alloy",
        instructions: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
    ):
        if input_text == "bad":
            raise ValueError("invalid")
        if input_text == "provider":
            raise RuntimeError("provider down")
        return type(
            "VoiceGenerationResultStub",
            (),
            {
                "audio_bytes": b"fake-audio",
                "mime_type": "audio/mpeg",
                "model": self.model_name,
                "voice": voice,
                "response_format": response_format,
            },
        )()


class _VoiceRealtimeSessionServiceStub:
    def create_realtime_session(
        self,
        *,
        reference_context: str | None = None,
        translation: str = "WEB",
        voice: str | None = None,
    ):
        if reference_context == "Bad 1:1":
            raise InvalidReferenceError("invalid")
        if reference_context == "Missing 1:1":
            raise PassageNotFoundError("not found")
        if reference_context == "provider":
            raise RuntimeError("provider down")
        return type(
            "VoiceRealtimeSessionStub",
            (),
            {
                "client_secret": "ephemeral-secret",
                "expires_at": 1_700_000_000,
                "model": "gpt-realtime-mini",
                "voice": voice or "cedar",
                "webrtc_url": "https://api.openai.com/v1/realtime/calls",
            },
        )()


class _MusicGenerationServiceStub:
    def generate_music(self, payload):
        if payload.prompt == "bad":
            raise ValueError("invalid")
        if payload.prompt == "provider":
            raise RuntimeError("provider down")
        return MusicGenerateResponse(
            job_id="music-job-1",
            status="queued",
            provider="mock",
            title=payload.title or "Generated Track",
            prompt=payload.prompt,
        )

    def get_job_status(self, job_id: str):
        if job_id == "bad":
            raise ValueError("invalid")
        if job_id == "missing":
            raise LookupError("not found")
        if job_id == "provider":
            raise RuntimeError("provider down")
        return MusicJobResponse(
            job_id=job_id,
            status="completed",
            provider="mock",
            audio_url="https://example.com/music.wav",
            error=None,
        )


class APIRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[get_bible_provider] = lambda: _BibleProviderStub()
        app.dependency_overrides[get_study_plan_service] = lambda: _StudyPlanServiceStub()
        app.dependency_overrides[get_passage_image_service] = lambda: _PassageImageServiceStub()
        app.dependency_overrides[get_persona_chat_service] = lambda: _PersonaChatServiceStub()
        app.dependency_overrides[get_voice_transcription_service] = lambda: _VoiceTranscriptionServiceStub()
        app.dependency_overrides[get_voice_generation_service] = lambda: _VoiceGenerationServiceStub()
        app.dependency_overrides[get_voice_realtime_session_service] = lambda: _VoiceRealtimeSessionServiceStub()
        app.dependency_overrides[get_music_generation_service] = lambda: _MusicGenerationServiceStub()
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

    def test_persona_chat_stream_success(self) -> None:
        response = self.client.post(
            "/api/persona-chat/stream",
            json={
                "messages": [{"role": "user", "content": "How should we apply this passage?"}],
                "reference_context": "Luke 21:5-28",
                "translation": "WEB",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "text/event-stream; charset=utf-8")
        body = response.text
        self.assertIn("event: meta", body)
        self.assertIn('"model": "gpt-4o-mini"', body)
        self.assertIn("event: chunk", body)
        self.assertIn('"delta": "Sample "', body)
        self.assertIn('"delta": "persona response"', body)
        self.assertIn("event: done", body)

    def test_persona_chat_stream_invalid_payload_maps_to_400(self) -> None:
        response = self.client.post(
            "/api/persona-chat/stream",
            json={
                "messages": [{"role": "user", "content": "bad"}],
                "translation": "WEB",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_persona_chat_stream_provider_error_maps_to_502(self) -> None:
        response = self.client.post(
            "/api/persona-chat/stream",
            json={
                "messages": [{"role": "user", "content": "provider"}],
                "translation": "WEB",
            },
        )
        self.assertEqual(response.status_code, 502)

    def test_voice_transcription_success(self) -> None:
        response = self.client.post(
            "/api/voice/transcribe",
            json={
                "audio_base64": "ZmFrZQ==",
                "mime_type": "audio/webm",
                "file_name": "recording.webm",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["transcript"], "transcribed question")
        self.assertEqual(body["model"], "gpt-4o-mini-transcribe")

    def test_voice_transcription_validation_error_maps_to_400(self) -> None:
        response = self.client.post(
            "/api/voice/transcribe",
            json={
                "audio_base64": "bad",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_voice_transcription_provider_error_maps_to_502(self) -> None:
        response = self.client.post(
            "/api/voice/transcribe",
            json={
                "audio_base64": "provider",
            },
        )
        self.assertEqual(response.status_code, 502)

    def test_voice_realtime_session_success(self) -> None:
        response = self.client.post(
            "/api/voice/realtime/session",
            json={
                "reference_context": "Luke 21:36",
                "translation": "WEB",
                "voice": "marin",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["client_secret"], "ephemeral-secret")
        self.assertEqual(body["model"], "gpt-realtime-mini")
        self.assertEqual(body["voice"], "marin")
        self.assertEqual(body["webrtc_url"], "https://api.openai.com/v1/realtime/calls")

    def test_voice_realtime_session_invalid_reference_maps_to_400(self) -> None:
        response = self.client.post(
            "/api/voice/realtime/session",
            json={
                "reference_context": "Bad 1:1",
                "translation": "WEB",
                "voice": "cedar",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_voice_realtime_session_provider_error_maps_to_502(self) -> None:
        response = self.client.post(
            "/api/voice/realtime/session",
            json={
                "reference_context": "provider",
                "translation": "WEB",
                "voice": "cedar",
            },
        )
        self.assertEqual(response.status_code, 502)

    def test_voice_generation_success(self) -> None:
        response = self.client.post(
            "/api/voice/generate",
            json={
                "input": "Speak this response",
                "voice": "alloy",
                "response_format": "mp3",
                "speed": 1.0,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model"], "gpt-4o-mini-tts")
        self.assertEqual(body["voice"], "alloy")
        self.assertEqual(body["response_format"], "mp3")
        self.assertEqual(body["mime_type"], "audio/mpeg")

    def test_voice_generation_validation_error_maps_to_400(self) -> None:
        response = self.client.post(
            "/api/voice/generate",
            json={
                "input": "bad",
                "voice": "alloy",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_voice_generation_provider_error_maps_to_502(self) -> None:
        response = self.client.post(
            "/api/voice/generate",
            json={
                "input": "provider",
                "voice": "alloy",
            },
        )
        self.assertEqual(response.status_code, 502)

    def test_music_generate_success(self) -> None:
        response = self.client.post(
            "/api/music/generate",
            json={
                "prompt": "hopeful worship track",
                "style_hint": "modern worship",
                "mood_hint": "hopeful",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["job_id"], "music-job-1")
        self.assertEqual(body["provider"], "mock")

    def test_music_generate_validation_error_maps_to_400(self) -> None:
        response = self.client.post(
            "/api/music/generate",
            json={
                "prompt": "bad",
                "style_hint": "modern worship",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_music_generate_provider_error_maps_to_502(self) -> None:
        response = self.client.post(
            "/api/music/generate",
            json={
                "prompt": "provider",
                "style_hint": "modern worship",
            },
        )
        self.assertEqual(response.status_code, 502)

    def test_music_job_status_success(self) -> None:
        response = self.client.get("/api/music/jobs/music-job-1")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "completed")

    def test_music_job_status_bad_request_maps_to_400(self) -> None:
        response = self.client.get("/api/music/jobs/bad")
        self.assertEqual(response.status_code, 400)

    def test_music_job_status_not_found_maps_to_404(self) -> None:
        response = self.client.get("/api/music/jobs/missing")
        self.assertEqual(response.status_code, 404)

    def test_music_job_status_provider_error_maps_to_502(self) -> None:
        response = self.client.get("/api/music/jobs/provider")
        self.assertEqual(response.status_code, 502)

if __name__ == "__main__":
    unittest.main()
