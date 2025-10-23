"""Chat service orchestrating prompts to LLM backends."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from yaf_gpt.config import Settings
from yaf_gpt.model.llm_client import LLMClient, OpenAIChatClient, StubLLMClient

logger = logging.getLogger(__name__)


Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """Single chat message with a role and content."""

    role: Role
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """Incoming request payload containing the conversation history."""

    messages: list[ChatMessage]

    @model_validator(mode="after")
    def ensure_user_message(self) -> "ChatRequest":
        if not self.messages:
            raise ValueError("At least one message is required to generate a reply.")
        return self


class ChatResponse(BaseModel):
    """Response payload wrapping the assistant's reply."""

    message: ChatMessage


class ChatService:
    """Route chat history through the configured LLM client."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or StubLLMClient()

    def generate_reply(self, request: ChatRequest) -> ChatResponse:
        """Return an assistant message produced by the LLM client."""
        logger.debug("Generating reply for %d messages", len(request.messages))
        completion = self._llm.complete(request.messages)
        reply = ChatMessage(role="assistant", content=completion)
        return ChatResponse(message=reply)
