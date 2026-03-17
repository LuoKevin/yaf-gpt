from __future__ import annotations

from typing import Protocol


class VoiceTranscriptionGateway(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def transcribe(self, *, audio_bytes: bytes, mime_type: str, file_name: str) -> str:
        ...
