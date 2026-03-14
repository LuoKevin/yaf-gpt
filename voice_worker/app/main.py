from __future__ import annotations

from fastapi import FastAPI, HTTPException, status

from .providers import (
    VoiceProvider,
    VoiceProviderError,
    VoiceProviderNotImplementedError,
    VoiceProviderValidationError,
    build_provider_from_env,
    decode_reference_audio,
)
from .schemas import (
    APIErrorResponse,
    CloneVoiceRequest,
    CloneVoiceResponse,
    HealthResponse,
    SynthesizeRequest,
    SynthesizeResponse,
)

app = FastAPI(title="yaf-gpt-voice-worker")
provider: VoiceProvider = build_provider_from_env()


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(status="ok", provider=provider.name)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", provider=provider.name)


@app.post(
    "/v1/voices/clone",
    response_model=CloneVoiceResponse,
    responses={
        400: {"model": APIErrorResponse},
        501: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def clone_voice(payload: CloneVoiceRequest) -> CloneVoiceResponse:
    try:
        reference_audio = decode_reference_audio(payload.reference_audio_base64)
        cloned = provider.clone_voice(
            reference_audio=reference_audio,
            reference_text=payload.reference_text,
            voice_name=payload.voice_name,
        )
        return CloneVoiceResponse(
            voice_id=cloned.voice_id,
            provider=provider.name,
            status="ready",
            message=cloned.message,
        )
    except VoiceProviderValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except VoiceProviderNotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
    except VoiceProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@app.post(
    "/v1/tts/synthesize",
    response_model=SynthesizeResponse,
    responses={
        400: {"model": APIErrorResponse},
        501: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def synthesize(payload: SynthesizeRequest) -> SynthesizeResponse:
    try:
        audio = provider.synthesize(
            text=payload.text.strip(),
            voice_id=payload.voice_id,
            speed=payload.speed,
        )
        return SynthesizeResponse(
            provider=provider.name,
            voice_id=audio.voice_id,
            mime_type=audio.mime_type,
            sample_rate_hz=audio.sample_rate_hz,
            duration_seconds=audio.duration_seconds,
            audio_base64=audio.audio_data_url,
        )
    except VoiceProviderValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except VoiceProviderNotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
    except VoiceProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
