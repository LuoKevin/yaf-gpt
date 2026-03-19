from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...services.study_plan.bible_lookup import (
    BibleAPIProvider,
    BibleProvider,
    InvalidReferenceError,
    PassageNotFoundError,
    PassageProviderError,
)
from ..schemas import APIErrorResponse, BiblePassageResponse, BibleVerse, TranslationCode

router = APIRouter(prefix="/api/bible", tags=["bible"])


def get_bible_provider() -> BibleProvider:
    return BibleAPIProvider()


@router.get(
    "/passage",
    response_model=BiblePassageResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def get_passage(
    reference: str = Query(..., min_length=1, description="Bible reference, e.g. John 3:16-18"),
    translation: TranslationCode = Query(default="WEB"),
    provider: BibleProvider = Depends(get_bible_provider),
) -> BiblePassageResponse:
    try:
        passage = provider.get_passage(reference=reference, translation=translation)
    except InvalidReferenceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PassageNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PassageProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return BiblePassageResponse(
        reference=passage.reference,
        translation=passage.translation,  # type: ignore[arg-type]
        normalized_reference=passage.normalized_reference,
        text=passage.text,
        verses=[
            BibleVerse(book=v.book, chapter=v.chapter, verse=v.verse, text=v.text)
            for v in passage.verses
        ],
    )
