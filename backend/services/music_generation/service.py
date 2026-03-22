from __future__ import annotations

from typing import Optional

from backend.app.schemas import MusicGenerateRequest, MusicGenerateResponse, MusicJobResponse
from backend.media import MusicProvider, MusicProviderError, build_music_provider_from_env

from .domain import (
    GenerateMusicCommand,
    normalize_text,
    resolve_generate_music_command,
)


class MusicGenerationService:
    def __init__(
        self,
        *,
        music_provider: Optional[MusicProvider] = None,
    ) -> None:
        self._music_provider = music_provider or build_music_provider_from_env()

    def generate_music(self, payload: MusicGenerateRequest) -> MusicGenerateResponse:
        resolved = resolve_generate_music_command(
            GenerateMusicCommand(
                prompt=payload.prompt,
                style_hint=payload.style,
                mood_hint=payload.mood,
            )
        )
        try:
            job = self._music_provider.create_job(
                title=resolved.title,
                lyrics=resolved.prompt,
                style_hint=resolved.style_hint,
                mood_hint=resolved.mood_hint,
            )
        except MusicProviderError as exc:
            raise RuntimeError(str(exc)) from exc

        return MusicGenerateResponse(
            job_id=job.job_id,
            status=job.status,  # type: ignore[arg-type]
            provider=job.provider,
            title=resolved.title,
            prompt=resolved.prompt,
        )

    def get_job_status(self, job_id: str) -> MusicJobResponse:
        cleaned = normalize_text(job_id)
        if not cleaned:
            raise ValueError("Job id cannot be empty.")

        try:
            job = self._music_provider.get_job(cleaned)
        except MusicProviderError as exc:
            raise RuntimeError(str(exc)) from exc

        if job is None:
            raise LookupError(f"No music generation job found for id '{cleaned}'.")

        return MusicJobResponse(
            job_id=job.job_id,
            status=job.status,  # type: ignore[arg-type]
            provider=job.provider,
            audio_url=job.audio_url,
            error=job.error,
        )
