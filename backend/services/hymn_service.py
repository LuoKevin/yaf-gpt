from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from backend.app.schemas import (
    HymnGenerateRequest,
    HymnGenerateResponse,
    HymnJobResponse,
    HymnLyrics,
    UsageMetrics,
)
from backend.llm import ChatMessage, ChatProvider, OpenAIChatProvider, ProviderError
from backend.media import (
    MusicProvider,
    MusicProviderError,
    build_music_provider_from_env,
)

from .bible_lookup import (
    BibleAPIProvider,
    BibleProvider,
    InvalidReferenceError,
    PassageData,
)
from .hymn_prompt_builder import build_hymn_messages, build_hymn_repair_messages

DEFAULT_HYMN_LYRICS_MODEL = "gpt-4o-mini"


class HymnGenerationError(RuntimeError):
    """Base class for hymn generation failures."""


class HymnValidationError(HymnGenerationError):
    """Raised when generated lyrics fail schema validation."""


class HymnProviderError(HymnGenerationError):
    """Raised when LLM or music provider fails."""


class HymnJobNotFoundError(HymnGenerationError):
    """Raised when a music job id does not exist."""


@dataclass(frozen=True)
class _ResolvedPassage:
    normalized_reference: str
    translation: str
    passage_text: str


def _validate_hymn_output(data: dict) -> HymnLyrics:
    if hasattr(HymnLyrics, "model_validate"):
        return HymnLyrics.model_validate(data)
    return HymnLyrics.parse_obj(data)


def _extract_json_object(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _lyrics_to_text(hymn: HymnLyrics) -> str:
    lines = [hymn.title, ""]
    for section in hymn.sections:
        lines.append(f"[{section.label}]")
        lines.append(section.lyrics)
        lines.append("")
    return "\n".join(lines).strip()


class HymnService:
    def __init__(
        self,
        *,
        bible_provider: Optional[BibleProvider] = None,
        chat_provider: Optional[ChatProvider] = None,
        music_provider: Optional[MusicProvider] = None,
        model: str | None = None,
    ) -> None:
        self._bible_provider = bible_provider or BibleAPIProvider()
        self._chat_provider = chat_provider or OpenAIChatProvider()
        self._music_provider = music_provider or build_music_provider_from_env()
        self._model = model or os.getenv("HYMN_LYRICS_MODEL") or DEFAULT_HYMN_LYRICS_MODEL

    def generate_hymn(self, payload: HymnGenerateRequest) -> HymnGenerateResponse:
        passage = self._resolve_passage(payload)
        base_messages = build_hymn_messages(
            reference=payload.reference,
            normalized_reference=passage.normalized_reference,
            translation=passage.translation,
            passage_text=passage.passage_text,
            style_hint=payload.style_hint,
            mood_hint=payload.mood_hint,
            user_notes=payload.user_notes,
        )

        hymn, usage, model_name = self._generate_lyrics_with_retry(base_messages)

        try:
            job = self._music_provider.create_job(
                title=hymn.title,
                lyrics=_lyrics_to_text(hymn),
                style_hint=payload.style_hint,
                mood_hint=payload.mood_hint,
            )
        except MusicProviderError as exc:
            raise HymnProviderError(str(exc)) from exc

        return HymnGenerateResponse(
            reference=payload.reference.strip(),
            normalized_reference=passage.normalized_reference,
            translation=passage.translation,  # type: ignore[arg-type]
            passage_text=passage.passage_text,
            hymn=hymn,
            job_id=job.job_id,
            job_status=job.status,  # type: ignore[arg-type]
            provider=job.provider,
            model=model_name,
            usage=usage,
        )

    def get_job_status(self, job_id: str) -> HymnJobResponse:
        cleaned = job_id.strip()
        if not cleaned:
            raise HymnValidationError("Job id cannot be empty.")

        try:
            job = self._music_provider.get_job(cleaned)
        except MusicProviderError as exc:
            raise HymnProviderError(str(exc)) from exc

        if job is None:
            raise HymnJobNotFoundError(f"No hymn generation job found for id '{cleaned}'.")

        return HymnJobResponse(
            job_id=job.job_id,
            status=job.status,  # type: ignore[arg-type]
            provider=job.provider,
            audio_url=job.audio_url,
            error=job.error,
        )

    def _resolve_passage(self, payload: HymnGenerateRequest) -> _ResolvedPassage:
        if payload.passage_text and payload.passage_text.strip():
            return _ResolvedPassage(
                normalized_reference=payload.reference.strip(),
                translation=payload.translation,
                passage_text=payload.passage_text.strip(),
            )

        try:
            passage: PassageData = self._bible_provider.get_passage(
                reference=payload.reference,
                translation=payload.translation,
            )
        except InvalidReferenceError:
            raise

        return _ResolvedPassage(
            normalized_reference=passage.normalized_reference,
            translation=passage.translation,
            passage_text=passage.text,
        )

    def _generate_lyrics_with_retry(
        self,
        base_messages: list[ChatMessage],
    ) -> tuple[HymnLyrics, Optional[UsageMetrics], str]:
        messages = list(base_messages)

        for attempt in range(2):
            try:
                response = self._chat_provider.generate(
                    messages,
                    model=self._model,
                    temperature=0.7,
                    max_tokens=1400,
                )
            except ProviderError as exc:
                raise HymnProviderError(str(exc)) from exc

            try:
                data = _extract_json_object(response.content)
                hymn = _validate_hymn_output(data)
                usage = UsageMetrics(
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                )
                return hymn, usage, response.model
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                if attempt == 0:
                    messages = build_hymn_repair_messages(base_messages, response.content)
                    continue
                raise HymnValidationError("Model output did not match hymn schema.") from exc
            except Exception as exc:
                if attempt == 0:
                    messages = build_hymn_repair_messages(base_messages, response.content)
                    continue
                raise HymnValidationError("Model output did not match hymn schema.") from exc

        raise HymnValidationError("Failed to generate a valid hymn output.")
