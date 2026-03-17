from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .domain import (
    DEFAULT_VOICE_GENERATION_MODEL,
    SUPPORTED_RESPONSE_FORMATS,
)

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


def _extract_audio_bytes(response: object) -> bytes:
    if isinstance(response, (bytes, bytearray)):
        return bytes(response)

    read = getattr(response, "read", None)
    if callable(read):
        payload = read()
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)

    content = getattr(response, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)

    getvalue = getattr(response, "getvalue", None)
    if callable(getvalue):
        payload = getvalue()
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)

    raise RuntimeError("Voice generation provider returned no audio bytes.")


class OpenAIVoiceGenerationGateway:
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

        self._model = model or os.getenv("VOICE_GENERATION_MODEL") or DEFAULT_VOICE_GENERATION_MODEL

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def mime_type_for_format(response_format: str) -> str:
        return SUPPORTED_RESPONSE_FORMATS[response_format]

    def generate_audio(
        self,
        *,
        input_text: str,
        voice: str,
        instructions: str | None,
        response_format: str,
        speed: float,
    ) -> bytes:
        params = {
            "model": self._model,
            "voice": voice,
            "input": input_text,
            "response_format": response_format,
            "speed": speed,
        }
        if instructions:
            params["instructions"] = instructions

        try:
            response = self._client.audio.speech.create(**params)
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        return _extract_audio_bytes(response)
