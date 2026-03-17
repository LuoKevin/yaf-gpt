from __future__ import annotations

import unittest

from backend.app.schemas import MusicGenerateRequest
from backend.media import MusicJob, MusicProviderError
from backend.services.music_generation_service import (
    MusicGenerationJobNotFoundError,
    MusicGenerationProviderError,
    MusicGenerationService,
    MusicGenerationValidationError,
)


class _FakeMusicProvider:
    def __init__(self) -> None:
        self.created = []
        self.jobs: dict[str, MusicJob] = {
            "job-1": MusicJob(
                job_id="job-1",
                status="completed",
                provider="mock",
                audio_url="https://example.com/audio.wav",
                error=None,
            )
        }
        self.raise_create_error = False

    def create_job(self, *, title: str, lyrics: str, style_hint: str, mood_hint=None):
        if self.raise_create_error:
            raise MusicProviderError("provider down")
        self.created.append(
            {
                "title": title,
                "lyrics": lyrics,
                "style_hint": style_hint,
                "mood_hint": mood_hint,
            }
        )
        return MusicJob(
            job_id="job-1",
            status="queued",
            provider="mock",
            audio_url=None,
            error=None,
        )

    def get_job(self, job_id: str):
        return self.jobs.get(job_id)


class MusicGenerationServiceTests(unittest.TestCase):
    def test_generate_music_success(self) -> None:
        provider = _FakeMusicProvider()
        service = MusicGenerationService(music_provider=provider)

        response = service.generate_music(
            MusicGenerateRequest(
                prompt="  make a hopeful worship song about endurance  ",
                style_hint=" modern worship ",
                mood_hint=" hopeful ",
            )
        )

        self.assertEqual(response.job_id, "job-1")
        self.assertEqual(response.status, "queued")
        self.assertEqual(response.provider, "mock")
        self.assertEqual(response.prompt, "make a hopeful worship song about endurance")
        self.assertEqual(provider.created[0]["mood_hint"], "hopeful")

    def test_generate_music_rejects_blank_prompt_after_normalization(self) -> None:
        service = MusicGenerationService(music_provider=_FakeMusicProvider())

        with self.assertRaises(MusicGenerationValidationError):
            service.generate_music(
                MusicGenerateRequest(
                    prompt="   ",
                    style_hint="modern worship",
                )
            )

    def test_generate_music_maps_provider_error(self) -> None:
        provider = _FakeMusicProvider()
        provider.raise_create_error = True
        service = MusicGenerationService(music_provider=provider)

        with self.assertRaises(MusicGenerationProviderError):
            service.generate_music(
                MusicGenerateRequest(
                    prompt="generate a song",
                    style_hint="modern worship",
                )
            )

    def test_get_job_status_not_found(self) -> None:
        service = MusicGenerationService(music_provider=_FakeMusicProvider())

        with self.assertRaises(MusicGenerationJobNotFoundError):
            service.get_job_status("missing")


if __name__ == "__main__":
    unittest.main()
