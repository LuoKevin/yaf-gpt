from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .domain import TranscribeAudioBytesCommand, decode_audio_base64, fallback_filename, normalize_transcript
from .infrastructure import OpenAIVoiceTranscriptionGateway

if TYPE_CHECKING:
    from openai import OpenAI


class VoiceTranscriptionService:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        client: Optional["OpenAI"] = None,
        model: Optional[str] = None,
        gateway: Optional[OpenAIVoiceTranscriptionGateway] = None,
    ) -> None:
        self._gateway = gateway or OpenAIVoiceTranscriptionGateway(
            api_key=api_key,
            client=client,
            model=model,
        )

    @property
    def model_name(self) -> str:
        return self._gateway.model_name

    def transcribe_base64(
        self,
        *,
        audio_base64: str,
        mime_type: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> str:
        return self.transcribe_bytes(
            audio_bytes=decode_audio_base64(audio_base64),
            mime_type=mime_type,
            file_name=file_name,
        )

    def transcribe_bytes(
        self,
        *,
        audio_bytes: bytes,
        mime_type: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> str:
        command = TranscribeAudioBytesCommand(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            file_name=file_name,
        )
        if not command.audio_bytes:
            raise ValueError("Audio payload is empty.")

        resolved_file_name = command.file_name.strip() if command.file_name else fallback_filename(command.mime_type)
        if not resolved_file_name:
            resolved_file_name = fallback_filename(command.mime_type)
        resolved_mime = (command.mime_type or "application/octet-stream").strip() or "application/octet-stream"

        transcript = normalize_transcript(
            self._gateway.transcribe(
                audio_bytes=command.audio_bytes,
                mime_type=resolved_mime,
                file_name=resolved_file_name,
            )
        )
        if not transcript:
            raise ValueError("Transcription was empty.")
        return transcript
