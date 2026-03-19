from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ...services.study_plan.bible_lookup import (
    InvalidReferenceError,
    PassageNotFoundError,
    PassageProviderError,
)
from ...services.study_plan.passage_image_service import (
    PassageImageProviderError,
    PassageImageService,
)
from ..schemas import APIErrorResponse, PassageImageRequest, PassageImageResponse

router = APIRouter(prefix="/api", tags=["image"])


def get_passage_image_service() -> PassageImageService:
    return PassageImageService()


@router.post(
    "/passage-image",
    response_model=PassageImageResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def create_passage_image(
    payload: PassageImageRequest,
    service: PassageImageService = Depends(get_passage_image_service),
) -> PassageImageResponse:
    try:
        return service.generate_passage_image(payload)
    except InvalidReferenceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PassageNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (PassageProviderError, PassageImageProviderError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
