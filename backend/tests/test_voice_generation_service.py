from __future__ import annotations

import json
import unittest

from backend.services.voice_chat.generation import VoiceGenerationService
from backend.services.voice_chat.generation.providers import ModalVoiceGenerationProvider


class _FakeSpeechResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _FakeOpenAIClient:
    class _Speech:
        def __init__(self, parent: "_FakeOpenAIClient") -> None:
            self._parent = parent

        def create(self, **kwargs):
            self._parent.last_kwargs = kwargs
            if self._parent.error_message:
                raise RuntimeError(self._parent.error_message)
            return _FakeSpeechResponse(self._parent.audio_bytes)

    class _Audio:
        def __init__(self, parent: "_FakeOpenAIClient") -> None:
            self.speech = _FakeOpenAIClient._Speech(parent)

    def __init__(self, *, audio_bytes: bytes = b"voice-audio", error_message: str | None = None) -> None:
        self.audio_bytes = audio_bytes
        self.error_message = error_message
        self.last_kwargs = None
        self.audio = _FakeOpenAIClient._Audio(self)


class _FakeUrlopenResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class VoiceGenerationServiceTests(unittest.TestCase):
    def test_generates_audio_with_supported_options(self) -> None:
        client = _FakeOpenAIClient(audio_bytes=b"abc123")
        service = VoiceGenerationService(client=client, model="gpt-4o-mini-tts")

        result = service.generate_audio(
            input_text=" Speak this clearly. ",
            voice="coral",
            instructions=" cheerful and warm ",
            response_format="wav",
            speed=1.25,
        )

        self.assertEqual(result.audio_bytes, b"abc123")
        self.assertEqual(result.mime_type, "audio/wav")
        self.assertEqual(result.voice, "coral")
        self.assertEqual(result.response_format, "wav")
        self.assertEqual(result.model, "gpt-4o-mini-tts")
        self.assertIsNotNone(client.last_kwargs)
        self.assertEqual(client.last_kwargs["model"], "gpt-4o-mini-tts")
        self.assertEqual(client.last_kwargs["voice"], "coral")
        self.assertEqual(client.last_kwargs["input"], "Speak this clearly.")
        self.assertEqual(client.last_kwargs["instructions"], "cheerful and warm")

    def test_rejects_unsupported_voice(self) -> None:
        service = VoiceGenerationService(client=_FakeOpenAIClient(), model="gpt-4o-mini-tts")

        with self.assertRaises(ValueError):
            service.generate_audio(input_text="hello", voice="robot")

    def test_rejects_instructions_for_tts1_models(self) -> None:
        service = VoiceGenerationService(client=_FakeOpenAIClient(), model="tts-1")

        with self.assertRaises(ValueError):
            service.generate_audio(
                input_text="hello",
                voice="alloy",
                instructions="sound dramatic",
            )

    def test_maps_provider_errors(self) -> None:
        service = VoiceGenerationService(
            client=_FakeOpenAIClient(error_message="provider unavailable"),
            model="gpt-4o-mini-tts",
        )

        with self.assertRaises(RuntimeError):
            service.generate_audio(input_text="hello", voice="alloy")

    def test_modal_provider_posts_to_remote_endpoint(self) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=0):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeUrlopenResponse(b"modal-audio")

        provider = ModalVoiceGenerationProvider(
            base_url="https://modal.example/generate",
            model="chatterbox-turbo-modal",
            voice_prompt_name="Lucy.wav",
            bearer_token="secret-token",
            request_opener=fake_urlopen,
        )
        service = VoiceGenerationService(provider=provider)

        result = service.generate_audio(
            input_text=" Read this aloud. ",
            voice="alloy",
            response_format="wav",
        )

        self.assertEqual(result.audio_bytes, b"modal-audio")
        self.assertEqual(result.mime_type, "audio/wav")
        self.assertEqual(result.model, "chatterbox-turbo-modal")
        self.assertEqual(captured["url"], "https://modal.example/generate")
        self.assertEqual(captured["timeout"], 60)
        self.assertEqual(captured["body"], {
            "input_text": "Read this aloud.",
            "voice_prompt_name": "Lucy.wav",
            "voice": "alloy",
            "instructions": None,
            "speed": 1.0,
        })
        self.assertIn(("Authorization", "Bearer secret-token"), captured["headers"].items())

    def test_modal_provider_rejects_non_wav_output(self) -> None:
        provider = ModalVoiceGenerationProvider(
            base_url="https://modal.example/generate",
            request_opener=lambda *args, **kwargs: _FakeUrlopenResponse(b"unused"),
        )
        service = VoiceGenerationService(provider=provider)

        with self.assertRaises(RuntimeError):
            service.generate_audio(input_text="hello", voice="alloy", response_format="mp3")


if __name__ == "__main__":
    unittest.main()
