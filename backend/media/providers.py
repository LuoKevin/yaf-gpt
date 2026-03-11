from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI

_ENV_LOADED = False


class ImageProviderError(RuntimeError):
    """Raised when image generation fails."""


class MusicProviderError(RuntimeError):
    """Raised when music generation fails."""


@dataclass(frozen=True)
class ImageGenerationResult:
    image_b64_or_url: str


class ImageProvider(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        model: str,
        size: str = "1024x1024",
    ) -> ImageGenerationResult:
        ...


@dataclass(frozen=True)
class MusicJob:
    job_id: str
    status: str
    provider: str
    audio_url: Optional[str] = None
    error: Optional[str] = None


class MusicProvider(Protocol):
    def create_job(
        self,
        *,
        title: str,
        lyrics: str,
        style_hint: str,
        mood_hint: Optional[str] = None,
    ) -> MusicJob:
        ...

    def get_job(self, job_id: str) -> Optional[MusicJob]:
        ...


class OpenAIImageProvider:
    def __init__(self, *, api_key: Optional[str] = None, client: Optional[OpenAI] = None) -> None:
        _load_backend_env()
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key and client is None:
            raise ImageProviderError("OPENAI_API_KEY is not set")
        self._client = client or OpenAI(api_key=resolved_key)

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        size: str = "1024x1024",
    ) -> ImageGenerationResult:
        try:
            response = self._client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
            )
            data = getattr(response, "data", None) or []
            if not data:
                raise ImageProviderError("Image provider returned no image payload.")

            first = data[0]
            url = getattr(first, "url", None)
            b64_json = getattr(first, "b64_json", None)

            if isinstance(url, str) and url.strip():
                return ImageGenerationResult(image_b64_or_url=url)

            if isinstance(b64_json, str) and b64_json.strip():
                return ImageGenerationResult(image_b64_or_url=f"data:image/png;base64,{b64_json}")

            raise ImageProviderError("Image provider response did not include url or base64 image data.")
        except ImageProviderError:
            raise
        except Exception as exc:  # pragma: no cover - provider/network behavior
            raise ImageProviderError(str(exc)) from exc


@dataclass
class _StoredMusicJob:
    job_id: str
    title: str
    lyrics: str
    style_hint: str
    mood_hint: Optional[str]
    created_at_mono: float
    status: str
    audio_url: Optional[str]
    error: Optional[str]


class MockMusicProvider:
    """A deterministic async-like job provider for local development and tests."""

    _IN_PROGRESS_AFTER_SECONDS = 1.5
    _COMPLETE_AFTER_SECONDS = 3.5
    _MOCK_AUDIO_BYTES = (
        b"RIFF$\x00\x00\x00WAVEfmt "
        b"\x10\x00\x00\x00\x01\x00\x01\x00"
        b"D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00"
        b"data\x00\x00\x00\x00"
    )

    def __init__(self) -> None:
        self._jobs: dict[str, _StoredMusicJob] = {}
        self._lock = threading.Lock()
        encoded = base64.b64encode(self._MOCK_AUDIO_BYTES).decode("ascii")
        self._audio_data_url = f"data:audio/wav;base64,{encoded}"

    def create_job(
        self,
        *,
        title: str,
        lyrics: str,
        style_hint: str,
        mood_hint: Optional[str] = None,
    ) -> MusicJob:
        with self._lock:
            job_id = uuid4().hex
            job = _StoredMusicJob(
                job_id=job_id,
                title=title,
                lyrics=lyrics,
                style_hint=style_hint,
                mood_hint=mood_hint,
                created_at_mono=time.monotonic(),
                status="queued",
                audio_url=None,
                error=None,
            )
            self._jobs[job_id] = job
            return self._snapshot(job)

    def get_job(self, job_id: str) -> Optional[MusicJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None

            if job.status not in {"completed", "failed"}:
                elapsed = time.monotonic() - job.created_at_mono
                if elapsed >= self._COMPLETE_AFTER_SECONDS:
                    job.status = "completed"
                    job.audio_url = self._audio_data_url
                elif elapsed >= self._IN_PROGRESS_AFTER_SECONDS:
                    job.status = "in_progress"

            return self._snapshot(job)

    @staticmethod
    def _snapshot(job: _StoredMusicJob) -> MusicJob:
        return MusicJob(
            job_id=job.job_id,
            status=job.status,
            provider="mock",
            audio_url=job.audio_url,
            error=job.error,
        )


class SunoMusicProvider:
    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        _load_backend_env()
        self._api_key = api_key or os.getenv("SUNO_API_KEY")
        self._base_url = (base_url or os.getenv("SUNO_BASE_URL") or "").strip()
        if not self._api_key:
            raise MusicProviderError("SUNO_API_KEY is not set")

    def create_job(
        self,
        *,
        title: str,
        lyrics: str,
        style_hint: str,
        mood_hint: Optional[str] = None,
    ) -> MusicJob:
        raise MusicProviderError("Suno-compatible music provider adapter is not implemented yet.")

    def get_job(self, job_id: str) -> Optional[MusicJob]:
        raise MusicProviderError("Suno-compatible music provider adapter is not implemented yet.")


def build_image_provider_from_env() -> ImageProvider:
    _load_backend_env()
    provider = (os.getenv("IMAGE_PROVIDER") or "openai").strip().lower()
    if provider == "openai":
        return OpenAIImageProvider()
    raise ImageProviderError(f"Unsupported IMAGE_PROVIDER '{provider}'.")


def build_music_provider_from_env() -> MusicProvider:
    _load_backend_env()
    provider = (os.getenv("MUSIC_PROVIDER") or "mock").strip().lower()
    if provider == "mock":
        return MockMusicProvider()
    if provider == "suno":
        return SunoMusicProvider()
    raise MusicProviderError(f"Unsupported MUSIC_PROVIDER '{provider}'.")


def _load_backend_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)
    _ENV_LOADED = True
