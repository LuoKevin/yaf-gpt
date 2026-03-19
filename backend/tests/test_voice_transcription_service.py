from __future__ import annotations

import unittest

from backend.services.voice_chat.transcription import VoiceTranscriptionService


class _FakeTranscriptionResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeOpenAIClient:
    class _Transcriptions:
        def __init__(self, parent: "_FakeOpenAIClient") -> None:
            self._parent = parent

        def create(self, **kwargs):
            self._parent.last_kwargs = kwargs
            if self._parent.error_message:
                raise RuntimeError(self._parent.error_message)
            return _FakeTranscriptionResponse(self._parent.transcript_text)

    class _Audio:
        def __init__(self, parent: "_FakeOpenAIClient") -> None:
            self.transcriptions = _FakeOpenAIClient._Transcriptions(parent)

    def __init__(self, *, transcript_text: str = "hello world", error_message: str | None = None) -> None:
        self.transcript_text = transcript_text
        self.error_message = error_message
        self.last_kwargs = None
        self.audio = _FakeOpenAIClient._Audio(self)


class VoiceTranscriptionServiceTests(unittest.TestCase):
    def test_transcribes_base64_data_url(self) -> None:
        client = _FakeOpenAIClient(transcript_text="  hello   group  ")
        service = VoiceTranscriptionService(client=client, model="test-model")

        transcript = service.transcribe_base64(
            audio_base64="data:audio/webm;base64,ZmFrZQ==",
            mime_type="audio/webm",
            file_name="voice.webm",
        )

        self.assertEqual(transcript, "hello group")
        self.assertIsNotNone(client.last_kwargs)
        self.assertEqual(client.last_kwargs["model"], "test-model")

    def test_rejects_invalid_base64(self) -> None:
        service = VoiceTranscriptionService(client=_FakeOpenAIClient(), model="test-model")

        with self.assertRaises(ValueError):
            service.transcribe_base64(audio_base64="not-base64")

    def test_maps_provider_errors(self) -> None:
        service = VoiceTranscriptionService(
            client=_FakeOpenAIClient(error_message="provider unavailable"),
            model="test-model",
        )

        with self.assertRaises(RuntimeError):
            service.transcribe_base64(audio_base64="ZmFrZQ==")

    def test_rejects_empty_transcript(self) -> None:
        service = VoiceTranscriptionService(client=_FakeOpenAIClient(transcript_text="   "), model="test-model")

        with self.assertRaises(ValueError):
            service.transcribe_base64(audio_base64="ZmFrZQ==")


if __name__ == "__main__":
    unittest.main()
