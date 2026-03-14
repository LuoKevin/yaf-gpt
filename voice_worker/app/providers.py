from __future__ import annotations

import base64
import io
import math
import os
import uuid
import wave
from dataclasses import dataclass
from typing import Protocol


class VoiceProviderError(RuntimeError):
    """Base class for voice provider failures."""


class VoiceProviderValidationError(VoiceProviderError):
    """Raised for bad client payloads."""


class VoiceProviderNotImplementedError(VoiceProviderError):
    """Raised when provider mode is not implemented."""


@dataclass(frozen=True)
class ClonedVoice:
    voice_id: str
    message: str | None = None


@dataclass(frozen=True)
class SynthesisAudio:
    voice_id: str | None
    audio_data_url: str
    duration_seconds: float
    sample_rate_hz: int = 16000
    mime_type: str = "audio/wav"


class VoiceProvider(Protocol):
    @property
    def name(self) -> str:
        ...

    def clone_voice(
        self,
        *,
        reference_audio: bytes,
        reference_text: str | None,
        voice_name: str | None,
    ) -> ClonedVoice:
        ...

    def synthesize(
        self,
        *,
        text: str,
        voice_id: str | None,
        speed: float,
    ) -> SynthesisAudio:
        ...


def _decode_data_url_or_base64(payload: str) -> bytes:
    cleaned = payload.strip()
    if cleaned.startswith("data:"):
        marker = cleaned.find(",")
        if marker < 0:
            raise VoiceProviderValidationError("Invalid data URL payload.")
        cleaned = cleaned[marker + 1 :]
    if not cleaned:
        raise VoiceProviderValidationError("Reference audio payload is empty.")
    try:
        return base64.b64decode(cleaned, validate=True)
    except Exception as exc:
        raise VoiceProviderValidationError("Reference audio must be valid base64.") from exc


def decode_reference_audio(payload: str) -> bytes:
    audio = _decode_data_url_or_base64(payload)
    if not audio:
        raise VoiceProviderValidationError("Reference audio payload is empty.")
    return audio


def _to_data_url_wav(audio_bytes: bytes) -> str:
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _render_sine_wave(text: str, *, speed: float, sample_rate: int = 16000) -> tuple[bytes, float]:
    duration_seconds = max(0.5, min(6.0, (len(text) / 22.0) / max(speed, 0.01)))
    frames = int(duration_seconds * sample_rate)
    amplitude = 11000
    frequency = 220.0

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(frames):
            value = int(amplitude * math.sin(2 * math.pi * frequency * (i / sample_rate)))
            wav_file.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(b"")
    return buffer.getvalue(), round(duration_seconds, 3)


class MockVoiceProvider:
    """In-memory provider for local development and scaffolding."""

    def __init__(self) -> None:
        self._known_voices: set[str] = set()

    @property
    def name(self) -> str:
        return "mock"

    def clone_voice(
        self,
        *,
        reference_audio: bytes,
        reference_text: str | None,
        voice_name: str | None,
    ) -> ClonedVoice:
        if len(reference_audio) < 2048:
            raise VoiceProviderValidationError("Reference audio is too short for cloning.")
        voice_id = f"voice_{uuid.uuid4().hex[:12]}"
        self._known_voices.add(voice_id)
        message = f"Voice profile '{voice_name.strip()}' ready." if voice_name and voice_name.strip() else None
        return ClonedVoice(voice_id=voice_id, message=message)

    def synthesize(
        self,
        *,
        text: str,
        voice_id: str | None,
        speed: float,
    ) -> SynthesisAudio:
        cleaned_voice_id = voice_id.strip() if voice_id else None
        if cleaned_voice_id and cleaned_voice_id not in self._known_voices:
            raise VoiceProviderValidationError(f"Unknown voice_id '{cleaned_voice_id}'.")
        wav_bytes, duration_seconds = _render_sine_wave(text, speed=speed)
        return SynthesisAudio(
            voice_id=cleaned_voice_id,
            audio_data_url=_to_data_url_wav(wav_bytes),
            duration_seconds=duration_seconds,
        )


class Qwen3StubVoiceProvider(MockVoiceProvider):
    """Drop-in scaffold for future Qwen3-TTS integration."""

    @property
    def name(self) -> str:
        return "qwen3_stub"


def build_provider_from_env() -> VoiceProvider:
    provider_name = os.getenv("VOICE_WORKER_PROVIDER", "mock").strip().lower()
    if provider_name == "mock":
        return MockVoiceProvider()
    if provider_name in {"qwen3", "qwen3_stub"}:
        return Qwen3StubVoiceProvider()
    raise VoiceProviderNotImplementedError(
        f"Unsupported VOICE_WORKER_PROVIDER '{provider_name}'. Supported: mock, qwen3_stub."
    )
