from __future__ import annotations

import os
from typing import Iterable, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .provider import ChatChunk, ChatMessage, ChatProvider, ChatResponse, ProviderError


def _to_input(messages: list[ChatMessage]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


class OpenAIChatProvider(ChatProvider):
    def __init__(self, *, api_key: Optional[str] = None, client: Optional[OpenAI] = None) -> None:
        load_dotenv()
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key and client is None:
            raise ProviderError("OPENAI_API_KEY is not set")
        self._client = client or OpenAI(api_key=resolved_key)

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        try:
            params = {
                "model": model,
                "input": _to_input(messages),
                "temperature": temperature,
            }
            if max_tokens is not None:
                params["max_output_tokens"] = max_tokens

            response = self._client.responses.create(**params)
            content = getattr(response, "output_text", None) or ""
            usage = getattr(response, "usage", None)

            return ChatResponse(
                content=content,
                model=model,
                raw=response,
                prompt_tokens=getattr(usage, "input_tokens", None),
                completion_tokens=getattr(usage, "output_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )
        except Exception as exc:  # pragma: no cover - provider/network error handling
            raise ProviderError(str(exc)) from exc

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Iterable[ChatChunk]:
        try:
            params = {
                "model": model,
                "input": _to_input(messages),
                "temperature": temperature,
                "stream": True,
            }
            if max_tokens is not None:
                params["max_output_tokens"] = max_tokens

            stream = self._client.responses.create(**params)
            for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        yield ChatChunk(content_delta=delta, raw=event)
                elif getattr(event, "type", None) == "error":
                    message = getattr(event, "message", "Provider error")
                    raise ProviderError(message)
        except Exception as exc:  # pragma: no cover - provider/network error handling
            raise ProviderError(str(exc)) from exc
