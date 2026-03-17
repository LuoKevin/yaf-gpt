from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.app.schemas import MusicGenerateRequest, MusicGenerateResponse, MusicJobResponse
from backend.media import MusicProvider, MusicProviderError, build_music_provider_from_env

DEFAULT_TRACK_TITLE = "Generated Track"


class MusicGenerationError(RuntimeError):
    """Base class for music generation failures."""


class MusicGenerationValidationError(MusicGenerationError):
    """Raised when request data is invalid."""


class MusicGenerationProviderError(MusicGenerationError):
    """Raised when music provider fails."""


class MusicGenerationJobNotFoundError(MusicGenerationError):
    """Raised when a music job id does not exist."""


@dataclass(frozen=True)
class _ResolvedGenerateRequest:
    title: str
    prompt: str
    style_hint: str
    mood_hint: Optional[str]


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _derive_title(prompt: str) -> str:
    words = prompt.split()
    if not words:
        return DEFAULT_TRACK_TITLE
    return " ".join(words[:6]).title()


class MusicGenerationService:
    def __init__(self, *, music_provider: Optional[MusicProvider] = None) -> None:
        self._music_provider = music_provider or build_music_provider_from_env()

    def generate_music(self, payload: MusicGenerateRequest) -> MusicGenerateResponse:
        resolved = self._resolve_generate_request(payload)
        try:
            job = self._music_provider.create_job(
                title=resolved.title,
                lyrics=resolved.prompt,
                style_hint=resolved.style_hint,
                mood_hint=resolved.mood_hint,
            )
        except MusicProviderError as exc:
            raise MusicGenerationProviderError(str(exc)) from exc

        return MusicGenerateResponse(
            job_id=job.job_id,
            status=job.status,  # type: ignore[arg-type]
            provider=job.provider,
            title=resolved.title,
            prompt=resolved.prompt,
        )

    def get_job_status(self, job_id: str) -> MusicJobResponse:
        cleaned = _normalize_text(job_id)
        if not cleaned:
            raise MusicGenerationValidationError("Job id cannot be empty.")

        try:
            job = self._music_provider.get_job(cleaned)
        except MusicProviderError as exc:
            raise MusicGenerationProviderError(str(exc)) from exc

        if job is None:
            raise MusicGenerationJobNotFoundError(f"No music generation job found for id '{cleaned}'.")

        return MusicJobResponse(
            job_id=job.job_id,
            status=job.status,  # type: ignore[arg-type]
            provider=job.provider,
            audio_url=job.audio_url,
            error=job.error,
        )

    def _resolve_generate_request(self, payload: MusicGenerateRequest) -> _ResolvedGenerateRequest:
        prompt = _normalize_text(payload.prompt)
        if not prompt:
            raise MusicGenerationValidationError("Prompt cannot be empty.")

        style_hint = _normalize_text(payload.style_hint)
        if not style_hint:
            raise MusicGenerationValidationError("Style hint cannot be empty.")

        mood_hint = _normalize_text(payload.mood_hint) or None
        title = _normalize_text(payload.title) or _derive_title(prompt)

        return _ResolvedGenerateRequest(
            title=title,
            prompt=prompt,
            style_hint=style_hint,
            mood_hint=mood_hint,
        )
