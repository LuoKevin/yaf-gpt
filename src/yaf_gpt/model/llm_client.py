"""Clients responsible for communicating with large language models."""

from __future__ import annotations

import os
from typing import Protocol, Sequence, TYPE_CHECKING

from yaf_gpt.config import ChatConfig

if TYPE_CHECKING:
    from yaf_gpt.pipeline.chat_service import ChatMessage


class LLMClient(Protocol):
    """Interface for chat-completion style LLM backends."""

    def complete(self, messages: Sequence["ChatMessage"]) -> str:
        """Return the assistant reply text given the conversation history."""


class StubLLMClient:
    """Offline placeholder used during early development and tests."""

    def complete(self, messages: Sequence["ChatMessage"]) -> str:  # noqa: D401
        return "Stubbed response from the default LLM client."


class OpenAIChatClient:
    """Thin wrapper around the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        config: ChatConfig | None = None
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._config = config or ChatConfig()
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY must be provided to use OpenAIChatClient.")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError("openai package is required for OpenAIChatClient.") from exc

        self._client = OpenAI(api_key=self._api_key)

    def complete(self, messages: Sequence["ChatMessage"]) -> str:
        """Send conversation history to OpenAI and return the assistant text."""
        payload = [{"role": message.role, "content": message.content} for message in messages]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=payload,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature
        )
        message = response.choices[0].message
        return message.content or ""
