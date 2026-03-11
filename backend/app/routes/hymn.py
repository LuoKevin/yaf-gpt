from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Path, status

from ...services.bible_lookup import (
    InvalidReferenceError,
    PassageNotFoundError,
    PassageProviderError,
)
from ...services.hymn_service import (
    HymnJobNotFoundError,
    HymnProviderError,
    HymnService,
    HymnValidationError,
)
from ..schemas import (
    APIErrorResponse,
    HymnGenerateRequest,
    HymnGenerateResponse,
    HymnJobResponse,
)

router = APIRouter(prefix="/api/hymn", tags=["hymn"])


@lru_cache(maxsize=1)
def get_hymn_service() -> HymnService:
    return HymnService()


@router.post(
    "/generate",
    response_model=HymnGenerateResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def generate_hymn(
    payload: HymnGenerateRequest,
    service: HymnService = Depends(get_hymn_service),
) -> HymnGenerateResponse:
    try:
        return service.generate_hymn(payload)
    except InvalidReferenceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PassageNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (PassageProviderError, HymnValidationError, HymnProviderError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.get(
    "/jobs/{job_id}",
    response_model=HymnJobResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def get_hymn_job(
    job_id: str = Path(..., min_length=1),
    service: HymnService = Depends(get_hymn_service),
) -> HymnJobResponse:
    try:
        return service.get_job_status(job_id)
    except HymnValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except HymnJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except HymnProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
