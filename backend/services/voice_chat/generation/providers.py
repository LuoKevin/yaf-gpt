from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from .domain import DEFAULT_VOICE_GENERATION_MODEL, SUPPORTED_RESPONSE_FORMATS

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
    env_path = Path(__file__).resolve().parents[3] / ".env"
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


def mime_type_for_format(response_format: str) -> str:
    return SUPPORTED_RESPONSE_FORMATS[response_format]


class OpenAIVoiceGenerationProvider:
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

        self.model_name = model or os.getenv("VOICE_GENERATION_MODEL") or DEFAULT_VOICE_GENERATION_MODEL
        self.default_response_format = "mp3"

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
            "model": self.model_name,
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


class SelfHostedVoiceGenerationProvider:
    def __init__(self, *, base_url: Optional[str] = None, model: Optional[str] = None) -> None:
        _load_backend_env()
        self._base_url = (base_url or os.getenv("SELF_HOSTED_VOICE_GENERATION_URL") or "").strip()
        self.model_name = model or os.getenv("VOICE_GENERATION_MODEL") or "self-hosted-voice"
        self.default_response_format = "wav"
        if not self._base_url:
            raise RuntimeError("SELF_HOSTED_VOICE_GENERATION_URL is not set")

    def generate_audio(
        self,
        *,
        input_text: str,
        voice: str,
        instructions: str | None,
        response_format: str,
        speed: float,
    ) -> bytes:
        raise RuntimeError("Self-hosted voice generation provider is not implemented yet.")


class ModalVoiceGenerationProvider:
    # Adjust this directly in code when tuning Chatterbox sampling.
    _TEMPERATURE = 1

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        voice_prompt_name: Optional[str] = None,
        bearer_token: Optional[str] = None,
        request_opener: Optional[Callable[..., object]] = None,
    ) -> None:
        _load_backend_env()
        self._base_url = (base_url or os.getenv("MODAL_VOICE_GENERATION_URL") or "").strip()
        self.model_name = model or os.getenv("MODAL_VOICE_GENERATION_MODEL") or "chatterbox-turbo-modal"
        self._voice_prompt_name = (
            voice_prompt_name
            or os.getenv("MODAL_VOICE_PROMPT_NAME")
            or "Lucy.wav"
        ).strip()
        self._bearer_token = (bearer_token or os.getenv("MODAL_VOICE_GENERATION_TOKEN") or "").strip()
        self._request_opener = request_opener or urllib_request.urlopen
        self.default_response_format = "wav"
        if not self._base_url:
            raise RuntimeError("MODAL_VOICE_GENERATION_URL is not set")

    def generate_audio(
        self,
        *,
        input_text: str,
        voice: str,
        instructions: str | None,
        response_format: str,
        speed: float,
    ) -> bytes:
        if response_format != "wav":
            raise RuntimeError("Modal voice generation currently supports only wav output.")

        payload = {
            "input_text": input_text,
            "voice_prompt_name": self._voice_prompt_name,
            "temperature": self._TEMPERATURE,
            # Keep these in the payload for future provider-side expansion.
            "voice": voice,
            "instructions": instructions,
            "speed": speed,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        }
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"

        request = urllib_request.Request(
            self._base_url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            response = self._request_opener(request, timeout=60)
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(detail or f"Modal voice generation failed with status {exc.code}.") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Unable to reach Modal voice generation endpoint: {exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        return _extract_audio_bytes(response)


def build_voice_generation_provider_from_env() -> object:
    _load_backend_env()
    provider = (os.getenv("VOICE_GENERATION_PROVIDER") or "openai").strip().lower()
    if provider == "openai":
        return OpenAIVoiceGenerationProvider()
    if provider == "modal":
        return ModalVoiceGenerationProvider()
    if provider in {"self_hosted", "self-hosted"}:
        return SelfHostedVoiceGenerationProvider()
    raise RuntimeError(f"Unsupported VOICE_GENERATION_PROVIDER '{provider}'.")
