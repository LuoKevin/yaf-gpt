from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"


class VoiceTranscriptionError(RuntimeError):
    """Base class for voice-transcription failures."""


class VoiceTranscriptionValidationError(VoiceTranscriptionError):
    """Raised when input audio is missing or malformed."""


class VoiceTranscriptionProviderError(VoiceTranscriptionError):
    """Raised when the transcription provider fails."""


def _strip_data_url(audio_base64: str) -> str:
    cleaned = audio_base64.strip()
    if cleaned.startswith("data:"):
        marker = cleaned.find(",")
        if marker < 0:
            raise VoiceTranscriptionValidationError("Audio payload data URL is invalid.")
        return cleaned[marker + 1 :].strip()
    return cleaned


def _decode_audio_base64(audio_base64: str) -> bytes:
    cleaned = _strip_data_url(audio_base64)
    if not cleaned:
        raise VoiceTranscriptionValidationError("Audio payload is empty.")

    try:
        return base64.b64decode(cleaned, validate=True)
    except Exception as exc:
        raise VoiceTranscriptionValidationError("Audio payload must be valid base64.") from exc


def _fallback_filename(mime_type: Optional[str]) -> str:
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


class VoiceTranscriptionService:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
    ) -> None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(env_path)

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key and client is None:
            raise VoiceTranscriptionProviderError("OPENAI_API_KEY is not set")

        self._client = client or OpenAI(api_key=resolved_key)
        self._model = model or os.getenv("VOICE_TRANSCRIPTION_MODEL") or DEFAULT_TRANSCRIPTION_MODEL

    @property
    def model_name(self) -> str:
        return self._model

    def transcribe_base64(
        self,
        *,
        audio_base64: str,
        mime_type: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> str:
        audio_bytes = _decode_audio_base64(audio_base64)
        return self.transcribe_bytes(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            file_name=file_name,
        )

    def transcribe_bytes(
        self,
        *,
        audio_bytes: bytes,
        mime_type: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> str:
        if not audio_bytes:
            raise VoiceTranscriptionValidationError("Audio payload is empty.")

        resolved_file_name = file_name.strip() if file_name else _fallback_filename(mime_type)
        if not resolved_file_name:
            resolved_file_name = _fallback_filename(mime_type)
        resolved_mime = (mime_type or "application/octet-stream").strip() or "application/octet-stream"

        try:
            response = self._client.audio.transcriptions.create(
                model=self._model,
                file=(resolved_file_name, audio_bytes, resolved_mime),
            )
        except Exception as exc:
            raise VoiceTranscriptionProviderError(str(exc)) from exc

        transcript = " ".join((getattr(response, "text", "") or "").split())
        if not transcript:
            raise VoiceTranscriptionValidationError("Transcription was empty.")
        return transcript
