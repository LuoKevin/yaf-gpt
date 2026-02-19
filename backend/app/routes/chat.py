from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..schemas import APIErrorResponse, PersonaChatRequest, PersonaChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post(
    "/persona-chat",
    response_model=PersonaChatResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
        501: {"model": APIErrorResponse},
    },
)
def create_persona_chat(payload: PersonaChatRequest) -> PersonaChatResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Persona chat is not implemented yet.",
    )

