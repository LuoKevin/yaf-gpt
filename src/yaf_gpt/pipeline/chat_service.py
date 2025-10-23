"""Stub chat service used while wiring up the API surface."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """Single chat message with a role and content."""

    role: Role
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """Incoming request payload containing the conversation history."""

    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    """Response payload wrapping the assistant's reply."""

    message: ChatMessage


class ChatService:
    """Naive service that returns a deterministic assistant response."""

    def generate_reply(self, request: ChatRequest) -> ChatResponse:
        """Return a canned assistant message while the real model is pending."""
        if not request.messages:
            logger.warning("Received empty message list; returning default stub.")

        reply = ChatMessage(
            role="assistant",
            content="This is a stub response from yaf-gpt. Real model coming soon!",
        )
        return ChatResponse(message=reply)
