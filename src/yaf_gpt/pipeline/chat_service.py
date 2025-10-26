"""Chat service orchestrating prompts to LLM backends."""

from __future__ import annotations
import time

import logging
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from yaf_gpt.config import ChatConfig
from yaf_gpt.model.llm_client import LLMClient, StubLLMClient

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

    def __init__(self, llm_client: LLMClient | None = None, config: ChatConfig | None = None) -> None:
        self._llm = llm_client or StubLLMClient()
        self._config = config or ChatConfig()

    def generate_reply(self, request: ChatRequest) -> ChatResponse:
        """Return an assistant message produced by the LLM client."""
        logger.debug("Generating reply for %d messages", len(request.messages))
        start_time = time.time()
        conversation = self._build_conversation(request.messages)
        completion = self._llm.complete(conversation)
        reply = ChatMessage(role="assistant", content=completion)
        elapsed_time = time.time() - start_time
        logger.info("chat_completion", extra={
            "latency_ms" : elapsed_time,
            "temperature": self._config.temperature,
            "max_tokens" : self._config.max_tokens
        })
        return ChatResponse(message=reply)

    def _build_conversation(self, user_messages: list[ChatMessage]) -> list[ChatMessage]:
        """Inject system prompt and return the full conversation for the LLM."""
        conversation: list[ChatMessage] = []
        if self._config.system_prompt:
            conversation.append(ChatMessage(role="system", content=self._config.system_prompt))

        conversation.extend(user_messages)
        return conversation
