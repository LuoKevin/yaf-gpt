from __future__ import annotations

import os

from backend.app.schemas import PersonaChatRequest, PersonaChatResponse, UsageMetrics
from backend.llm import (
    CALVINIST_BIBLE_STUDY,
    ChatMessage,
    ChatProvider,
    OpenAIChatProvider,
    ProviderError,
)
from backend.llm.system_prompts import YOUNG_ADULT_COMMUNICATION

from .bible_lookup import BibleAPIProvider, BibleProvider, PassageData

DEFAULT_PERSONA_MODEL = "gpt-4o-mini"


class PersonaChatError(RuntimeError):
    """Base class for persona-chat failures."""


class PersonaChatValidationError(PersonaChatError):
    """Raised when persona output is empty or malformed."""


class PersonaChatProviderError(PersonaChatError):
    """Raised when provider fails."""


def _build_persona_system_prompt() -> str:
    return (
        f"{CALVINIST_BIBLE_STUDY} "
        f"{YOUNG_ADULT_COMMUNICATION} "
        "You are a mentor-style discussion partner for young adults. "
        "Balance biblical faithfulness with practical clarity. "
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


class PersonaChatService:
    def __init__(
        self,
        *,
        bible_provider: BibleProvider | None = None,
        chat_provider: ChatProvider | None = None,
        model: str | None = None,
    ) -> None:
        self._bible_provider = bible_provider or BibleAPIProvider()
        self._chat_provider = chat_provider or OpenAIChatProvider()
        self._model = model or os.getenv("PERSONA_MODEL") or DEFAULT_PERSONA_MODEL

    def create_reply(self, payload: PersonaChatRequest) -> PersonaChatResponse:
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_build_persona_system_prompt())
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

        for message in payload.messages:
            content = message.content.strip()
            if not content:
                continue
            messages.append(ChatMessage(role=message.role, content=content))

        if len(messages) <= 1:
            raise PersonaChatValidationError("Persona chat requires at least one non-empty conversation message.")

        try:
            response = self._chat_provider.generate(
                messages,
                model=self._model,
                temperature=0.5,
                max_tokens=700,
            )
        except ProviderError as exc:
            raise PersonaChatProviderError(str(exc)) from exc

        reply = response.content.strip()
        if not reply:
            raise PersonaChatValidationError("Persona response was empty.")

        return PersonaChatResponse(
            reply=reply,
            model=response.model,
            usage=UsageMetrics(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
            ),
        )
