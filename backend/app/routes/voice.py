from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ...services.voice_transcription_service import (
    VoiceTranscriptionProviderError,
    VoiceTranscriptionService,
    VoiceTranscriptionValidationError,
)
from ..schemas import APIErrorResponse, VoiceTranscriptionRequest, VoiceTranscriptionResponse

router = APIRouter(prefix="/api/voice", tags=["voice"])


def get_voice_transcription_service() -> VoiceTranscriptionService:
    return VoiceTranscriptionService()


@router.post(
    "/transcribe",
    response_model=VoiceTranscriptionResponse,
    responses={
        400: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def transcribe_voice(
    payload: VoiceTranscriptionRequest,
    service: VoiceTranscriptionService = Depends(get_voice_transcription_service),
) -> VoiceTranscriptionResponse:
    try:
        transcript = service.transcribe_base64(
            audio_base64=payload.audio_base64,
            mime_type=payload.mime_type,
            file_name=payload.file_name,
        )
        return VoiceTranscriptionResponse(
            transcript=transcript,
            model=service.model_name,
        )
    except VoiceTranscriptionValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except VoiceTranscriptionProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
