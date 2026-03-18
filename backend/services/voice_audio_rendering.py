from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from .voice_generation import VoiceGenerationService, VoiceGenerationResult

DEFAULT_VOICE_AUDIO_RENDERER = "backend_tts"

_ENV_LOADED = False


def _load_backend_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)
    _ENV_LOADED = True


@dataclass(frozen=True)
class VoiceAudioRenderResult:
    renderer: str
    rendered: bool
    audio_bytes: bytes | None
    mime_type: str | None
    model: str | None
    voice: str | None
    response_format: str | None


class BackendTTSVoiceAudioRenderer:
    renderer_name = "backend_tts"

    def __init__(self, *, voice_generation_service: VoiceGenerationService | None = None) -> None:
        self._voice_generation_service = voice_generation_service or VoiceGenerationService()

    def render_audio(
        self,
        *,
        input_text: str,
        voice: str = "alloy",
        instructions: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> VoiceAudioRenderResult:
        result: VoiceGenerationResult = self._voice_generation_service.generate_audio(
            input_text=input_text,
            voice=voice,
            instructions=instructions,
            response_format=response_format,
            speed=speed,
        )
        return VoiceAudioRenderResult(
            renderer=self.renderer_name,
            rendered=True,
            audio_bytes=result.audio_bytes,
            mime_type=result.mime_type,
            model=result.model,
            voice=result.voice,
            response_format=result.response_format,
        )


class NativeRealtimeVoiceAudioRenderer:
    renderer_name = "realtime_native"

    def render_audio(
        self,
        *,
        input_text: str,
        voice: str = "alloy",
        instructions: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> VoiceAudioRenderResult:
        return VoiceAudioRenderResult(
            renderer=self.renderer_name,
            rendered=False,
            audio_bytes=None,
            mime_type=None,
            model=None,
            voice=voice,
            response_format=response_format,
        )


class DisabledVoiceAudioRenderer:
    renderer_name = "disabled"

    def render_audio(
        self,
        *,
        input_text: str,
        voice: str = "alloy",
        instructions: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> VoiceAudioRenderResult:
        return VoiceAudioRenderResult(
            renderer=self.renderer_name,
            rendered=False,
            audio_bytes=None,
            mime_type=None,
            model=None,
            voice=voice,
            response_format=response_format,
        )


def build_voice_audio_renderer_from_env(
    *,
    voice_generation_service: VoiceGenerationService | None = None,
) -> object:
    _load_backend_env()
    renderer_name = (os.getenv("VOICE_AUDIO_RENDERER") or DEFAULT_VOICE_AUDIO_RENDERER).strip().lower()

    if renderer_name in {"backend_tts", "backend-tts", "openai_tts", "openai-tts"}:
        return BackendTTSVoiceAudioRenderer(voice_generation_service=voice_generation_service)
    if renderer_name in {"realtime_native", "realtime-native"}:
        return NativeRealtimeVoiceAudioRenderer()
    if renderer_name in {"disabled", "none"}:
        return DisabledVoiceAudioRenderer()

    raise RuntimeError(f"Unsupported VOICE_AUDIO_RENDERER '{renderer_name}'.")


class VoiceAudioRenderingService:
    def __init__(
        self,
        *,
        renderer: object | None = None,
        voice_generation_service: VoiceGenerationService | None = None,
    ) -> None:
        self._renderer = renderer or build_voice_audio_renderer_from_env(
            voice_generation_service=voice_generation_service
        )

    @property
    def renderer_name(self) -> str:
        return getattr(self._renderer, "renderer_name", "unknown")

    def render_audio(
        self,
        *,
        input_text: str,
        voice: str = "alloy",
        instructions: Optional[str] = None,
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> VoiceAudioRenderResult:
        return self._renderer.render_audio(
            input_text=input_text,
            voice=voice,
            instructions=instructions,
            response_format=response_format,
            speed=speed,
        )
