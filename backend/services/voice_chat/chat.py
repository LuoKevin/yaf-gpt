from __future__ import annotations

from collections.abc import Iterable, Iterator
import os

from backend.app.schemas import ChatRequest, ChatResponse, UsageMetrics
from backend.llm import (
    CALVINIST_BIBLE_STUDY,
    ChatChunk,
    ChatMessage,
    ChatProvider,
    OpenAIChatProvider,
    ProviderError,
)
from backend.llm.system_prompts import YOUNG_ADULT_COMMUNICATION

from ..study_plan.bible_lookup import BibleAPIProvider, BibleProvider, PassageData

DEFAULT_CHAT_MODEL = "gpt-4o-mini"


class ChatError(RuntimeError):
    """Base class for chat failures."""


class ChatValidationError(ChatError):
    """Raised when chat output is empty or malformed."""


class ChatProviderError(ChatError):
    """Raised when the provider fails."""


def _build_chat_system_prompt() -> str:
    return (
        f"{CALVINIST_BIBLE_STUDY} "
        f"{YOUNG_ADULT_COMMUNICATION} "
        "You are a mentor-style discussion partner for young adults. "
        "Balance biblical faithfulness with practical clarity. "
        "When using lists, format each item on its own new line for readability. "
        "Stay concise, ask one helpful follow-up question when useful, and avoid sounding preachy. "
        "If users ask for personal counseling or crisis help, encourage speaking with a pastor or trusted local leader."
    )


def _build_reference_context_block(passage: PassageData) -> str:
    excerpt = " ".join(passage.text.split())
    if len(excerpt) > 1200:
        excerpt = f"{excerpt[:1197].rstrip()}..."

    return (
        "Use this passage context when relevant:\n"
        f"Reference: {passage.normalized_reference} ({passage.translation})\n"
        f"Passage:\n{excerpt}"
    )


class ChatService:
    def __init__(
        self,
        *,
        bible_provider: BibleProvider | None = None,
        chat_provider: ChatProvider | None = None,
        model: str | None = None,
    ) -> None:
        self._bible_provider = bible_provider or BibleAPIProvider()
        self._chat_provider = chat_provider or OpenAIChatProvider()
        self._model = model or os.getenv("CHAT_MODEL") or os.getenv("PERSONA_MODEL") or DEFAULT_CHAT_MODEL

    def _build_messages(self, payload: ChatRequest) -> list[ChatMessage]:
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_build_chat_system_prompt())
        ]

        if payload.reference_context and payload.reference_context.strip():
            passage = self._bible_provider.get_passage(
                reference=payload.reference_context.strip(),
                translation=payload.translation,
            )
            messages.append(
                ChatMessage(
                    role="system",
                    content=_build_reference_context_block(passage),
                )
            )

        has_conversation_message = False
        for message in payload.messages:
            content = message.content.strip()
            if not content:
                continue
            has_conversation_message = True
            messages.append(ChatMessage(role=message.role, content=content))

        if not has_conversation_message:
            raise ChatValidationError("Chat requires at least one non-empty conversation message.")

        return messages

    def create_reply(self, payload: ChatRequest) -> ChatResponse:
        messages = self._build_messages(payload)

        try:
            response = self._chat_provider.generate(
                messages,
                model=self._model,
                temperature=0.5,
                max_tokens=700,
            )
        except ProviderError as exc:
            raise ChatProviderError(str(exc)) from exc

        reply = response.content.strip()
        if not reply:
            raise ChatValidationError("Chat response was empty.")

        return ChatResponse(
            reply=reply,
            model=response.model,
            usage=UsageMetrics(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
            ),
        )

    def stream_reply(self, payload: ChatRequest) -> tuple[str, Iterable[str]]:
        messages = self._build_messages(payload)

        try:
            stream_iter = iter(
                self._chat_provider.stream(
                    messages,
                    model=self._model,
                    temperature=0.5,
                    max_tokens=700,
                )
            )
        except ProviderError as exc:
            raise ChatProviderError(str(exc)) from exc

        first_delta: str | None = None
        try:
            while first_delta is None:
                chunk = next(stream_iter)
                if chunk.content_delta:
                    first_delta = chunk.content_delta
        except StopIteration as exc:
            raise ChatValidationError("Chat response was empty.") from exc
        except ProviderError as exc:
            raise ChatProviderError(str(exc)) from exc

        def _yield_deltas(first_chunk: str, remaining: Iterator[ChatChunk]) -> Iterator[str]:
            yield first_chunk
            try:
                for chunk in remaining:
                    if chunk.content_delta:
                        yield chunk.content_delta
            except ProviderError as exc:
                raise ChatProviderError(str(exc)) from exc

        return self._model, _yield_deltas(first_delta, stream_iter)
