from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ..rate_limit import limit_requests
from ...services.study_plan.bible_lookup import (
    InvalidReferenceError,
    PassageNotFoundError,
    PassageProviderError,
)
from ...services.voice_chat import (
    ChatProviderError,
    ChatService,
    ChatValidationError,
)
from ..schemas import APIErrorResponse, ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


def get_chat_service() -> ChatService:
    return ChatService()


get_persona_chat_service = get_chat_service


def _sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        429: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
    dependencies=[Depends(limit_requests(bucket="chat", max_requests=20))],
)
@router.post(
    "/persona-chat",
    response_model=ChatResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        429: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
    dependencies=[Depends(limit_requests(bucket="persona-chat", max_requests=20))],
)
def create_chat(
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    try:
        return service.create_reply(payload)
    except (InvalidReferenceError, ChatValidationError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PassageNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (PassageProviderError, ChatProviderError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post(
    "/chat/stream",
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        429: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
    dependencies=[Depends(limit_requests(bucket="chat-stream", max_requests=12))],
)
@router.post(
    "/persona-chat/stream",
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        429: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
    dependencies=[Depends(limit_requests(bucket="persona-chat-stream", max_requests=12))],
)
def stream_chat(
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    try:
        model, deltas = service.stream_reply(payload)
    except (InvalidReferenceError, ChatValidationError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PassageNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (PassageProviderError, ChatProviderError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    def _event_stream():
        yield _sse_event("meta", {"model": model})
        try:
            for delta in deltas:
                yield _sse_event("chunk", {"delta": delta})
        except (ChatValidationError, ChatProviderError) as exc:
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


create_persona_chat = create_chat
stream_persona_chat = stream_chat
