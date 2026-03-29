from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.app.schemas import PersonaChatMessage, PersonaChatRequest, TranslationCode

from .audio_rendering import VoiceAudioRenderingService
from .persona import PersonaChatService
from .transcription import VoiceTranscriptionService


@dataclass(frozen=True)
class VoiceChatTurnResult:
    transcript: str
    transcript_model: str
    reply: str
    reply_model: str
    rendered: bool
    audio_bytes: bytes | None
    audio_mime_type: str | None
    audio_model: str | None
    audio_voice: str | None
    audio_response_format: str | None


class VoiceChatConversationService:
    def __init__(
        self,
        *,
        transcription_service: VoiceTranscriptionService | None = None,
        persona_service: PersonaChatService | None = None,
        audio_rendering_service: VoiceAudioRenderingService | None = None,
    ) -> None:
        self._transcription_service = transcription_service or VoiceTranscriptionService()
        self._persona_service = persona_service or PersonaChatService()
        self._audio_rendering_service = audio_rendering_service or VoiceAudioRenderingService()

    def create_turn_from_base64(
        self,
        *,
        audio_base64: str,
        mime_type: str | None = None,
        file_name: str | None = None,
        reference_context: str | None = None,
        translation: TranslationCode = "WEB",
    ) -> VoiceChatTurnResult:
        transcript = self._transcription_service.transcribe_base64(
            audio_base64=audio_base64,
            mime_type=mime_type,
            file_name=file_name,
        )

        persona_response = self._persona_service.create_reply(
            PersonaChatRequest(
                messages=[PersonaChatMessage(role="user", content=transcript)],
                reference_context=reference_context,
                translation=translation,
            )
        )

        audio_result = self._audio_rendering_service.render_audio(
            input_text=persona_response.reply,
        )

        return VoiceChatTurnResult(
            transcript=transcript,
            transcript_model=self._transcription_service.model_name,
            reply=persona_response.reply,
            reply_model=persona_response.model,
            rendered=audio_result.rendered,
            audio_bytes=audio_result.audio_bytes,
            audio_mime_type=audio_result.mime_type,
            audio_model=audio_result.model,
            audio_voice=audio_result.voice,
            audio_response_format=audio_result.response_format,
        )
