from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..schemas import APIErrorResponse, PassageImageRequest, PassageImageResponse

router = APIRouter(prefix="/api", tags=["image"])


@router.post(
    "/passage-image",
    response_model=PassageImageResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
        501: {"model": APIErrorResponse},
    },
)
def create_passage_image(payload: PassageImageRequest) -> PassageImageResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Passage image generation is not implemented yet.",
    )

