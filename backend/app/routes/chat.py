from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

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


def _sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


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


@router.post(
    "/persona-chat/stream",
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def stream_persona_chat(
    payload: PersonaChatRequest,
    service: PersonaChatService = Depends(get_persona_chat_service),
) -> StreamingResponse:
    try:
        model, deltas = service.stream_reply(payload)
    except (InvalidReferenceError, PersonaChatValidationError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PassageNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (PassageProviderError, PersonaChatProviderError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    def _event_stream():
        yield _sse_event("meta", {"model": model})
        try:
            for delta in deltas:
                yield _sse_event("chunk", {"delta": delta})
        except (PersonaChatValidationError, PersonaChatProviderError) as exc:
            yield _sse_event("error", {"detail": str(exc)})
            return
        yield _sse_event("done", {"done": True})

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
