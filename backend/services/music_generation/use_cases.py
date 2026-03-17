from __future__ import annotations

from .domain import (
    GenerateMusicCommand,
    MusicGenerationJobNotFoundError,
    MusicGenerationSubmission,
    MusicGenerationValidationError,
    MusicJobSnapshot,
    normalize_text,
    resolve_generate_music_command,
)
from .ports import MusicGenerationGateway


class GenerateMusicUseCase:
    def __init__(self, gateway: MusicGenerationGateway) -> None:
        self._gateway = gateway

    def execute(self, command: GenerateMusicCommand) -> MusicGenerationSubmission:
        resolved = resolve_generate_music_command(command)
        job = self._gateway.create_job(
            title=resolved.title,
            lyrics=resolved.prompt,
            style_hint=resolved.style_hint,
            mood_hint=resolved.mood_hint,
        )
        return MusicGenerationSubmission(
            job_id=job.job_id,
            status=job.status,
            provider=job.provider,
            title=resolved.title,
            prompt=resolved.prompt,
        )


class GetMusicJobStatusUseCase:
    def __init__(self, gateway: MusicGenerationGateway) -> None:
        self._gateway = gateway

    def execute(self, job_id: str) -> MusicJobSnapshot:
        cleaned = normalize_text(job_id)
        if not cleaned:
            raise MusicGenerationValidationError("Job id cannot be empty.")

        job = self._gateway.get_job(cleaned)
        if job is None:
            raise MusicGenerationJobNotFoundError(f"No music generation job found for id '{cleaned}'.")
        return job
