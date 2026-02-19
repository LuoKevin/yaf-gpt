from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from ..schemas import APIErrorResponse, BiblePassageResponse, TranslationCode

router = APIRouter(prefix="/api/bible", tags=["bible"])


@router.get(
    "/passage",
    response_model=BiblePassageResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
        501: {"model": APIErrorResponse},
    },
)
def get_passage(
    reference: str = Query(..., min_length=1, description="Bible reference, e.g. John 3:16-18"),
    translation: TranslationCode = Query(default="WEB"),
) -> BiblePassageResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Bible passage lookup is not implemented yet.",
    )

