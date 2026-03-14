from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class APIErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    provider: str


class CloneVoiceRequest(BaseModel):
    reference_audio_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded reference audio bytes, optionally as a data URL.",
    )
    reference_text: Optional[str] = Field(
        default=None,
        description="Optional transcript for the reference audio.",
    )
    voice_name: Optional[str] = Field(
        default=None,
        description="Optional display name for the cloned voice.",
    )


class CloneVoiceResponse(BaseModel):
    voice_id: str
    provider: str
    status: Literal["ready"]
    message: Optional[str] = None


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_id: Optional[str] = None
    response_format: Literal["wav"] = "wav"
    speed: float = Field(default=1.0, ge=0.5, le=1.5)


class SynthesizeResponse(BaseModel):
    provider: str
    voice_id: Optional[str] = None
    mime_type: str = "audio/wav"
    sample_rate_hz: int = 16000
    duration_seconds: float = Field(..., ge=0)
    audio_base64: str = Field(
        ...,
        description="Synthesized audio payload encoded as base64 data URL.",
    )
