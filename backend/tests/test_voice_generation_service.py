from __future__ import annotations

import unittest

from backend.services.voice_generation import VoiceGenerationService


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


if __name__ == "__main__":
    unittest.main()
