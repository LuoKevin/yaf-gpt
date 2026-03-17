from __future__ import annotations

from .domain import (
    TranscribeAudioBytesCommand,
    TranscribeVoiceCommand,
    VoiceTranscriptionResult,
    VoiceTranscriptionValidationError,
    decode_audio_base64,
    fallback_filename,
    normalize_transcript,
)
from .ports import VoiceTranscriptionGateway


class TranscribeVoiceBytesUseCase:
    def __init__(self, gateway: VoiceTranscriptionGateway) -> None:
        self._gateway = gateway

    def execute(self, command: TranscribeAudioBytesCommand) -> VoiceTranscriptionResult:
        if not command.audio_bytes:
            raise VoiceTranscriptionValidationError("Audio payload is empty.")

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
            raise VoiceTranscriptionValidationError("Transcription was empty.")

        return VoiceTranscriptionResult(transcript=transcript, model=self._gateway.model_name)


class TranscribeVoiceBase64UseCase:
    def __init__(self, bytes_use_case: TranscribeVoiceBytesUseCase) -> None:
        self._bytes_use_case = bytes_use_case

    def execute(self, command: TranscribeVoiceCommand) -> VoiceTranscriptionResult:
        return self._bytes_use_case.execute(
            TranscribeAudioBytesCommand(
                audio_bytes=decode_audio_base64(command.audio_base64),
                mime_type=command.mime_type,
                file_name=command.file_name,
            )
        )
