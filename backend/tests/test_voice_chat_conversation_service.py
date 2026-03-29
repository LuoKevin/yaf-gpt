from __future__ import annotations

import unittest

from backend.services.voice_chat.conversation import VoiceChatConversationService


class _TranscriptionServiceStub:
    model_name = "gpt-4o-mini-transcribe"

    def transcribe_base64(self, *, audio_base64: str, mime_type: str | None = None, file_name: str | None = None) -> str:
        if audio_base64 == "bad":
            raise ValueError("invalid audio")
        return "How should I apply this passage?"


class _PersonaServiceStub:
    def create_reply(self, payload):
        self.last_payload = payload
        return type(
            "PersonaResponseStub",
            (),
            {
                "reply": "Stay watchful, faithful, and anchored in Christ.",
                "model": "gpt-4o-mini",
            },
        )()


class _AudioRenderingServiceStub:
    def render_audio(self, *, input_text: str, voice: str = "alloy", instructions: str | None = None, response_format: str | None = None, speed: float = 1.0):
        self.last_input_text = input_text
        return type(
            "AudioRenderResultStub",
            (),
            {
                "rendered": True,
                "audio_bytes": b"voice-bytes",
                "mime_type": "audio/wav",
                "model": "chatterbox-turbo-modal",
                "voice": "alloy",
                "response_format": "wav",
            },
        )()


class VoiceChatConversationServiceTests(unittest.TestCase):
    def test_create_turn_transcribes_replies_and_renders_audio(self) -> None:
        persona = _PersonaServiceStub()
        audio = _AudioRenderingServiceStub()
        service = VoiceChatConversationService(
            transcription_service=_TranscriptionServiceStub(),
            persona_service=persona,
            audio_rendering_service=audio,
        )

        result = service.create_turn_from_base64(
            audio_base64="ZmFrZQ==",
            mime_type="audio/webm",
            file_name="voice.webm",
            reference_context="Luke 21:36",
            translation="WEB",
        )

        self.assertEqual(result.transcript, "How should I apply this passage?")
        self.assertEqual(result.reply, "Stay watchful, faithful, and anchored in Christ.")
        self.assertEqual(result.audio_bytes, b"voice-bytes")
        self.assertEqual(result.audio_mime_type, "audio/wav")
        self.assertEqual(result.audio_model, "chatterbox-turbo-modal")
        self.assertEqual(result.audio_response_format, "wav")
        self.assertEqual(persona.last_payload.reference_context, "Luke 21:36")
        self.assertEqual(audio.last_input_text, "Stay watchful, faithful, and anchored in Christ.")


if __name__ == "__main__":
    unittest.main()
