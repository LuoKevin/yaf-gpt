from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"


@dataclass(frozen=True)
class TranscribeVoiceCommand:
    audio_base64: str
    mime_type: Optional[str] = None
    file_name: Optional[str] = None


@dataclass(frozen=True)
class TranscribeAudioBytesCommand:
    audio_bytes: bytes
    mime_type: Optional[str] = None
    file_name: Optional[str] = None


@dataclass(frozen=True)
class VoiceTranscriptionResult:
    transcript: str
    model: str


def strip_data_url(audio_base64: str) -> str:
    cleaned = audio_base64.strip()
    if cleaned.startswith("data:"):
        marker = cleaned.find(",")
        if marker < 0:
            raise ValueError("Audio payload data URL is invalid.")
        return cleaned[marker + 1 :].strip()
    return cleaned


def decode_audio_base64(audio_base64: str) -> bytes:
    cleaned = strip_data_url(audio_base64)
    if not cleaned:
        raise ValueError("Audio payload is empty.")

    try:
        return base64.b64decode(cleaned, validate=True)
    except Exception as exc:
        raise ValueError("Audio payload must be valid base64.") from exc


def fallback_filename(mime_type: Optional[str]) -> str:
    normalized = (mime_type or "").split(";")[0].strip().lower()
    extension = {
        "audio/webm": "webm",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp4": "mp4",
        "audio/x-m4a": "m4a",
        "audio/ogg": "ogg",
    }.get(normalized, "webm")
    return f"voice_input.{extension}"


def normalize_transcript(value: str) -> str:
    return " ".join((value or "").split())
