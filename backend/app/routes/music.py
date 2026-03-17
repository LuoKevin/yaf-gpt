from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Path, status

from ...services.music_generation_service import (
    MusicGenerationJobNotFoundError,
    MusicGenerationProviderError,
    MusicGenerationService,
    MusicGenerationValidationError,
)
from ..schemas import APIErrorResponse, MusicGenerateRequest, MusicGenerateResponse, MusicJobResponse

router = APIRouter(prefix="/api/music", tags=["music"])


@lru_cache(maxsize=1)
def get_music_generation_service() -> MusicGenerationService:
    return MusicGenerationService()


@router.post(
    "/generate",
    response_model=MusicGenerateResponse,
    responses={
        400: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def generate_music(
    payload: MusicGenerateRequest,
    service: MusicGenerationService = Depends(get_music_generation_service),
) -> MusicGenerateResponse:
    try:
        return service.generate_music(payload)
    except MusicGenerationValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except MusicGenerationProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.get(
    "/jobs/{job_id}",
    response_model=MusicJobResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def get_music_job(
    job_id: str = Path(..., min_length=1),
    service: MusicGenerationService = Depends(get_music_generation_service),
) -> MusicJobResponse:
    try:
        return service.get_job_status(job_id)
    except MusicGenerationValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except MusicGenerationJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except MusicGenerationProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
