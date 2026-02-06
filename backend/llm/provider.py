from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatResponse:
    content: str
    model: str
    raw: Optional[object] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(frozen=True)
class ChatChunk:
    content_delta: str
    raw: Optional[object] = None


class ProviderError(RuntimeError):
    """Raised when the provider fails to generate a response."""


class ChatProvider(Protocol):
    """Minimal interface for a chat-completions provider adapter."""

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        ...

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Iterable[ChatChunk]:
        ...
