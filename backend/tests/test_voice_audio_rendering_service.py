from __future__ import annotations

import unittest

from backend.services.voice_audio_rendering import (
    BackendTTSVoiceAudioRenderer,
    NativeRealtimeVoiceAudioRenderer,
    VoiceAudioRenderingService,
)
from backend.services.voice_generation.domain import VoiceGenerationResult


class _VoiceGenerationServiceStub:
    def __init__(self) -> None:
        self.last_kwargs = None

    def generate_audio(
        self,
        *,
        input_text: str,
        voice: str = "alloy",
        instructions: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> VoiceGenerationResult:
        self.last_kwargs = {
            "input_text": input_text,
            "voice": voice,
            "instructions": instructions,
            "response_format": response_format,
            "speed": speed,
        }
        return VoiceGenerationResult(
            audio_bytes=b"audio-payload",
            mime_type="audio/mpeg",
            model="gpt-4o-mini-tts",
            voice=voice,
            response_format=response_format,
        )


class VoiceAudioRenderingServiceTests(unittest.TestCase):
    def test_backend_tts_renderer_uses_voice_generation_service(self) -> None:
        generation_service = _VoiceGenerationServiceStub()
        renderer = BackendTTSVoiceAudioRenderer(voice_generation_service=generation_service)
        service = VoiceAudioRenderingService(renderer=renderer)

        result = service.render_audio(
            input_text=" Speak this response. ",
            voice="coral",
            instructions="warm and steady",
            response_format="mp3",
            speed=1.1,
        )

        self.assertEqual(service.renderer_name, "backend_tts")
        self.assertTrue(result.rendered)
        self.assertEqual(result.audio_bytes, b"audio-payload")
        self.assertEqual(result.mime_type, "audio/mpeg")
        self.assertEqual(result.model, "gpt-4o-mini-tts")
        self.assertEqual(result.voice, "coral")
        self.assertIsNotNone(generation_service.last_kwargs)
        self.assertEqual(generation_service.last_kwargs["input_text"], " Speak this response. ")

    def test_realtime_native_renderer_returns_metadata_without_audio(self) -> None:
        service = VoiceAudioRenderingService(renderer=NativeRealtimeVoiceAudioRenderer())

        result = service.render_audio(
            input_text="Any text here.",
            voice="cedar",
            response_format="mp3",
        )

        self.assertEqual(service.renderer_name, "realtime_native")
        self.assertFalse(result.rendered)
        self.assertIsNone(result.audio_bytes)
        self.assertEqual(result.voice, "cedar")
        self.assertEqual(result.response_format, "mp3")


if __name__ == "__main__":
    unittest.main()
