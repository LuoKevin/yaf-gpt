from __future__ import annotations

from typing import Optional, Protocol

from .domain import MusicJobSnapshot


class MusicGenerationGateway(Protocol):
    def create_job(
        self,
        *,
        title: str,
        lyrics: str,
        style_hint: str,
        mood_hint: str | None = None,
    ) -> MusicJobSnapshot:
        ...

    def get_job(self, job_id: str) -> Optional[MusicJobSnapshot]:
        ...
