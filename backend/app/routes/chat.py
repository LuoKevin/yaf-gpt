from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ...services.bible_lookup import (
    InvalidReferenceError,
    PassageNotFoundError,
    PassageProviderError,
)
from ...services.persona_chat_service import (
    PersonaChatProviderError,
    PersonaChatService,
    PersonaChatValidationError,
)
from ..schemas import APIErrorResponse, PersonaChatRequest, PersonaChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


def get_persona_chat_service() -> PersonaChatService:
    return PersonaChatService()


@router.post(
    "/persona-chat",
    response_model=PersonaChatResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def create_persona_chat(
    payload: PersonaChatRequest,
    service: PersonaChatService = Depends(get_persona_chat_service),
) -> PersonaChatResponse:
    try:
        return service.create_reply(payload)
    except (InvalidReferenceError, PersonaChatValidationError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PassageNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (PassageProviderError, PersonaChatProviderError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
