from __future__ import annotations

import unittest

from backend.app.schemas import HymnGenerateRequest
from backend.llm.provider import ChatResponse
from backend.media import MusicJob
from backend.services.bible_lookup import PassageData, PassageVerse
from backend.services.hymn_service import HymnJobNotFoundError, HymnService, HymnValidationError


def _valid_hymn_json() -> str:
    return (
        "{"
        '"title":"Shelter Through the Storm",'
        '"theme":"Christ is faithful in trial",'
        '"scripture_references":["Luke 21:5-28"],'
        '"sections":['
        '{"label":"Verse 1","lyrics":"When earthly walls are shaken, Your promise still remains."},'
        '{"label":"Chorus","lyrics":"Christ our refuge, Christ our light, hold us through the night."},'
        '{"label":"Verse 2","lyrics":"Teach us patient endurance, until Your kingdom comes."}'
        "]"
        "}"
    )


class _FakeBibleProvider:
    def get_passage(self, reference: str, translation: str = "WEB") -> PassageData:
        return PassageData(
            reference=reference,
            normalized_reference="Luke 21:5-28",
            translation=translation,
            text="Jesus speaks of trials and endurance.",
            verses=[PassageVerse(book="Luke", chapter=21, verse=5, text="Jesus answered")],
        )


class _FakeChatProvider:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def generate(self, messages, *, model, temperature=0.2, max_tokens=None):
        output = self.outputs[self.calls]
        self.calls += 1
        return ChatResponse(
            content=output,
            model=model,
            prompt_tokens=120,
            completion_tokens=180,
            total_tokens=300,
        )

    def stream(self, messages, *, model, temperature=0.2, max_tokens=None):
        return []


class _FakeMusicProvider:
    def __init__(self) -> None:
        self.jobs: dict[str, MusicJob] = {
            "job-1": MusicJob(
                job_id="job-1",
                status="in_progress",
                provider="mock",
                audio_url=None,
                error=None,
            )
        }

    def create_job(self, *, title: str, lyrics: str, style_hint: str, mood_hint=None) -> MusicJob:
        return MusicJob(
            job_id="job-1",
            status="queued",
            provider="mock",
            audio_url=None,
            error=None,
        )

    def get_job(self, job_id: str):
        return self.jobs.get(job_id)


class HymnServiceTests(unittest.TestCase):
    def test_generate_hymn_retries_once_and_succeeds(self) -> None:
        service = HymnService(
            bible_provider=_FakeBibleProvider(),
            chat_provider=_FakeChatProvider(outputs=["not-json", _valid_hymn_json()]),
            music_provider=_FakeMusicProvider(),
            model="gpt-4o-mini",
        )

        response = service.generate_hymn(
            HymnGenerateRequest(
                reference="Luke 21:5-28",
                translation="WEB",
                style_hint="modern worship hymn, acoustic",
            )
        )

        self.assertEqual(response.job_id, "job-1")
        self.assertEqual(response.hymn.title, "Shelter Through the Storm")

    def test_generate_hymn_fails_after_second_invalid_output(self) -> None:
        service = HymnService(
            bible_provider=_FakeBibleProvider(),
            chat_provider=_FakeChatProvider(outputs=["bad", "still bad"]),
            music_provider=_FakeMusicProvider(),
            model="gpt-4o-mini",
        )

        with self.assertRaises(HymnValidationError):
            service.generate_hymn(
                HymnGenerateRequest(
                    reference="Luke 21:5-28",
                    translation="WEB",
                    style_hint="modern worship hymn, acoustic",
                )
            )

    def test_get_job_status_not_found(self) -> None:
        service = HymnService(
            bible_provider=_FakeBibleProvider(),
            chat_provider=_FakeChatProvider(outputs=[_valid_hymn_json()]),
            music_provider=_FakeMusicProvider(),
            model="gpt-4o-mini",
        )

        with self.assertRaises(HymnJobNotFoundError):
            service.get_job_status("missing")


if __name__ == "__main__":
    unittest.main()
