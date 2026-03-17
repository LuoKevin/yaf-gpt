from __future__ import annotations

from typing import Optional

from backend.app.schemas import MusicGenerateRequest, MusicGenerateResponse, MusicJobResponse
from backend.media import MusicProvider, MusicProviderError, build_music_provider_from_env

from .domain import GenerateMusicCommand, MusicGenerationProviderError, MusicJobSnapshot
from .ports import MusicGenerationGateway
from .use_cases import GenerateMusicUseCase, GetMusicJobStatusUseCase


class MediaMusicGenerationGateway:
    def __init__(self, provider: MusicProvider) -> None:
        self._provider = provider

    def create_job(
        self,
        *,
        title: str,
        lyrics: str,
        style_hint: str,
        mood_hint: Optional[str] = None,
    ) -> MusicJobSnapshot:
        try:
            job = self._provider.create_job(
                title=title,
                lyrics=lyrics,
                style_hint=style_hint,
                mood_hint=mood_hint,
            )
        except MusicProviderError as exc:
            raise MusicGenerationProviderError(str(exc)) from exc

        return MusicJobSnapshot(
            job_id=job.job_id,
            status=job.status,
            provider=job.provider,
            audio_url=job.audio_url,
            error=job.error,
        )

    def get_job(self, job_id: str) -> Optional[MusicJobSnapshot]:
        try:
            job = self._provider.get_job(job_id)
        except MusicProviderError as exc:
            raise MusicGenerationProviderError(str(exc)) from exc

        if job is None:
            return None

        return MusicJobSnapshot(
            job_id=job.job_id,
            status=job.status,
            provider=job.provider,
            audio_url=job.audio_url,
            error=job.error,
        )


class MusicGenerationService:
    def __init__(
        self,
        *,
        music_provider: Optional[MusicProvider] = None,
        gateway: Optional[MusicGenerationGateway] = None,
    ) -> None:
        resolved_gateway = gateway
        if resolved_gateway is None:
            provider = music_provider or build_music_provider_from_env()
            resolved_gateway = MediaMusicGenerationGateway(provider)

        self._generate_music = GenerateMusicUseCase(resolved_gateway)
        self._get_job_status = GetMusicJobStatusUseCase(resolved_gateway)

    def generate_music(self, payload: MusicGenerateRequest) -> MusicGenerateResponse:
        submission = self._generate_music.execute(
            GenerateMusicCommand(
                prompt=payload.prompt,
                style_hint=payload.style_hint,
                mood_hint=payload.mood_hint,
                title=payload.title,
            )
        )
        return MusicGenerateResponse(
            job_id=submission.job_id,
            status=submission.status,  # type: ignore[arg-type]
            provider=submission.provider,
            title=submission.title,
            prompt=submission.prompt,
        )

    def get_job_status(self, job_id: str) -> MusicJobResponse:
        job = self._get_job_status.execute(job_id)
        return MusicJobResponse(
            job_id=job.job_id,
            status=job.status,  # type: ignore[arg-type]
            provider=job.provider,
            audio_url=job.audio_url,
            error=job.error,
        )
