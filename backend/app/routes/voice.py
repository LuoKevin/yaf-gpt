from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, HTTPException, status

from ...services.voice_generation import VoiceGenerationService
from ...services.voice_transcription import VoiceTranscriptionService
from ..schemas import (
    APIErrorResponse,
    VoiceGenerationRequest,
    VoiceGenerationResponse,
    VoiceTranscriptionRequest,
    VoiceTranscriptionResponse,
)

router = APIRouter(prefix="/api/voice", tags=["voice"])


def get_voice_transcription_service() -> VoiceTranscriptionService:
    return VoiceTranscriptionService()


def get_voice_generation_service() -> VoiceGenerationService:
    return VoiceGenerationService()


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
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post(
    "/generate",
    response_model=VoiceGenerationResponse,
    responses={
        400: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def generate_voice(
    payload: VoiceGenerationRequest,
    service: VoiceGenerationService = Depends(get_voice_generation_service),
) -> VoiceGenerationResponse:
    try:
        result = service.generate_audio(
            input_text=payload.input,
            voice=payload.voice,
            instructions=payload.instructions,
            response_format=payload.response_format,
            speed=payload.speed,
        )
        return VoiceGenerationResponse(
            audio_base64=base64.b64encode(result.audio_bytes).decode("ascii"),
            mime_type=result.mime_type,
            model=result.model,
            voice=result.voice,  # type: ignore[arg-type]
            response_format=result.response_format,  # type: ignore[arg-type]
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
