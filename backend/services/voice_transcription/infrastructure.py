from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .domain import DEFAULT_TRANSCRIPTION_MODEL

if TYPE_CHECKING:
    from openai import OpenAI

_ENV_LOADED = False


def _load_backend_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path)
    _ENV_LOADED = True


class OpenAIVoiceTranscriptionGateway:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        client: Optional["OpenAI"] = None,
        model: Optional[str] = None,
    ) -> None:
        _load_backend_env()

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key and client is None:
            raise RuntimeError("OPENAI_API_KEY is not set")

        if client is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                raise RuntimeError("openai package is not installed") from exc
            self._client = OpenAI(api_key=resolved_key)
        else:
            self._client = client
        self._model = model or os.getenv("VOICE_TRANSCRIPTION_MODEL") or DEFAULT_TRANSCRIPTION_MODEL

    @property
    def model_name(self) -> str:
        return self._model

    def transcribe(self, *, audio_bytes: bytes, mime_type: str, file_name: str) -> str:
        try:
            response = self._client.audio.transcriptions.create(
                model=self._model,
                file=(file_name, audio_bytes, mime_type),
            )
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        return getattr(response, "text", "") or ""
