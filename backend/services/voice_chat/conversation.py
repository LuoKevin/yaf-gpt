from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.app.schemas import ChatRequest, ChatRequestMessage, TranslationCode

from .audio_rendering import VoiceAudioRenderingService
from .chat import ChatService
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
        chat_service: ChatService | None = None,
        persona_service: ChatService | None = None,
        audio_rendering_service: VoiceAudioRenderingService | None = None,
    ) -> None:
        self._transcription_service = transcription_service or VoiceTranscriptionService()
        self._chat_service = chat_service or persona_service or ChatService()
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

        chat_response = self._chat_service.create_reply(
            ChatRequest(
                messages=[ChatRequestMessage(role="user", content=transcript)],
                reference_context=reference_context,
                translation=translation,
            )
        )

        audio_result = self._audio_rendering_service.render_audio(
            input_text=chat_response.reply,
        )

        return VoiceChatTurnResult(
            transcript=transcript,
            transcript_model=self._transcription_service.model_name,
            reply=chat_response.reply,
            reply_model=chat_response.model,
            rendered=audio_result.rendered,
            audio_bytes=audio_result.audio_bytes,
            audio_mime_type=audio_result.mime_type,
            audio_model=audio_result.model,
            audio_voice=audio_result.voice,
            audio_response_format=audio_result.response_format,
        )
