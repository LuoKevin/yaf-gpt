from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .domain import TranscribeAudioBytesCommand, TranscribeVoiceCommand
from .infrastructure import OpenAIVoiceTranscriptionGateway
from .ports import VoiceTranscriptionGateway
from .use_cases import TranscribeVoiceBase64UseCase, TranscribeVoiceBytesUseCase

if TYPE_CHECKING:
    from openai import OpenAI


class VoiceTranscriptionService:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        client: Optional["OpenAI"] = None,
        model: Optional[str] = None,
        gateway: Optional[VoiceTranscriptionGateway] = None,
    ) -> None:
        resolved_gateway = gateway or OpenAIVoiceTranscriptionGateway(
            api_key=api_key,
            client=client,
            model=model,
        )
        self._gateway = resolved_gateway
        self._transcribe_bytes = TranscribeVoiceBytesUseCase(resolved_gateway)
        self._transcribe_base64 = TranscribeVoiceBase64UseCase(self._transcribe_bytes)

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
        result = self._transcribe_base64.execute(
            TranscribeVoiceCommand(
                audio_base64=audio_base64,
                mime_type=mime_type,
                file_name=file_name,
            )
        )
        return result.transcript

    def transcribe_bytes(
        self,
        *,
        audio_bytes: bytes,
        mime_type: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> str:
        result = self._transcribe_bytes.execute(
            TranscribeAudioBytesCommand(
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                file_name=file_name,
            )
        )
        return result.transcript
