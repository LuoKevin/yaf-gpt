from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from backend.app.schemas import (
    StudyPlanLLMOutput,
    StudyPlanRequest,
    StudyPlanResponse,
    UsageMetrics,
)
from backend.llm import ChatMessage, ChatProvider, OpenAIChatProvider, ProviderError

from .bible_lookup import (
    BibleAPIProvider,
    BibleProvider,
    InvalidReferenceError,
    PassageData,
)
from .prompt_builder import build_repair_messages, build_study_plan_messages
from .study_docx_structure import LukeStructureContext, LukeStructureRetriever
from .style_guide import load_luke_style_guide

DEFAULT_STUDY_PLAN_MODEL = "gpt-4o-mini"
MAX_QUESTION_NOTE_WORDS = 22
logger = logging.getLogger(__name__)


class StudyPlanGenerationError(RuntimeError):
    """Base class for study-plan generation failures."""


class StudyPlanValidationError(StudyPlanGenerationError):
    """Raised when model output cannot be parsed/validated."""


class StudyPlanProviderError(StudyPlanGenerationError):
    """Raised when model provider fails."""


@dataclass(frozen=True)
class _ResolvedPassage:
    normalized_reference: str
    translation: str
    passage_text: str


def _validate_model_output(data: dict) -> StudyPlanLLMOutput:
    if hasattr(StudyPlanLLMOutput, "model_validate"):
        return StudyPlanLLMOutput.model_validate(data)
    return StudyPlanLLMOutput.parse_obj(data)


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


def _normalize_note_list(values: list[str]) -> list[str]:
    return [" ".join(value.strip().split()) for value in values if value and value.strip()]


def _validate_optional_question_notes(
    parsed: StudyPlanLLMOutput,
    *,
    include_question_notes: bool,
) -> tuple[list[str] | None, list[str] | None]:
    if not include_question_notes:
        return None, None

    discussion_notes = parsed.discussion_question_notes
    reflection_notes = parsed.reflection_question_notes
    if discussion_notes is None or reflection_notes is None:
        raise ValueError("Question notes were requested but required note arrays are missing.")

    normalized_discussion = _normalize_note_list(discussion_notes)
    normalized_reflection = _normalize_note_list(reflection_notes)
    if len(normalized_discussion) != len(parsed.discussion_questions):
        raise ValueError("discussion_question_notes length must match discussion_questions.")
    if len(normalized_reflection) != len(parsed.reflection_questions):
        raise ValueError("reflection_question_notes length must match reflection_questions.")

    for note in normalized_discussion + normalized_reflection:
        if len(note.split()) > MAX_QUESTION_NOTE_WORDS:
            raise ValueError(
                f"Question note is too long; max allowed is {MAX_QUESTION_NOTE_WORDS} words."
            )

    return normalized_discussion, normalized_reflection


class StudyPlanService:
    def __init__(
        self,
        *,
        bible_provider: Optional[BibleProvider] = None,
        chat_provider: Optional[ChatProvider] = None,
        structure_retriever: Optional[LukeStructureRetriever] = None,
        model: str = DEFAULT_STUDY_PLAN_MODEL,
    ) -> None:
        self._bible_provider = bible_provider or BibleAPIProvider()
        self._chat_provider = chat_provider or OpenAIChatProvider()
        self._structure_retriever = structure_retriever or LukeStructureRetriever()
        self._model = model

    def generate_study_plan(self, payload: StudyPlanRequest) -> StudyPlanResponse:
        passage = self._resolve_passage(payload)
        style_guide = load_luke_style_guide()
        structure_context = self._retrieve_structure_context(passage.normalized_reference)
        base_messages = build_study_plan_messages(
            reference=payload.reference,
            normalized_reference=passage.normalized_reference,
            translation=passage.translation,
            passage_text=passage.passage_text,
            style_guide=style_guide,
            structure_context=structure_context,
            goals=payload.goals,
            user_notes=payload.user_notes,
            include_question_notes=payload.include_question_notes,
        )

        (
            llm_output,
            discussion_question_notes,
            reflection_question_notes,
            usage,
            model_name,
        ) = self._generate_with_retry(
            base_messages,
            include_question_notes=payload.include_question_notes,
        )
        return StudyPlanResponse(
            reference=payload.reference.strip(),
            normalized_reference=passage.normalized_reference,
            translation=passage.translation,  # type: ignore[arg-type]
            passage_text=passage.passage_text,
            passage_title=llm_output.passage_title,
            context_points=llm_output.context_points,
            discussion_questions=llm_output.discussion_questions,
            reflection_questions=llm_output.reflection_questions,
            include_question_notes=payload.include_question_notes,
            discussion_question_notes=discussion_question_notes,
            reflection_question_notes=reflection_question_notes,
            model=model_name,
            usage=usage,
        )

    def _resolve_passage(self, payload: StudyPlanRequest) -> _ResolvedPassage:
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

    def _retrieve_structure_context(
        self, normalized_reference: str
    ) -> Optional[LukeStructureContext]:
        try:
            structure_context = self._structure_retriever.retrieve(normalized_reference)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Failed to retrieve Luke structure exemplars for %s: %s",
                normalized_reference,
                exc,
            )
            return None

        if structure_context is None:
            logger.debug(
                "No Luke structure exemplars available for %s; using style-guide fallback only.",
                normalized_reference,
            )
            return None

        logger.debug(
            "Using Luke structure exemplars for %s: %s",
            normalized_reference,
            [example.normalized_reference for example in structure_context.examples],
        )
        return structure_context

    def _generate_with_retry(
        self,
        base_messages: list[ChatMessage],
        *,
        include_question_notes: bool,
    ) -> tuple[StudyPlanLLMOutput, list[str] | None, list[str] | None, Optional[UsageMetrics], str]:
        messages = list(base_messages)

        for attempt in range(2):
            try:
                response = self._chat_provider.generate(
                    messages,
                    model=self._model,
                    temperature=0.2,
                    max_tokens=1800,
                )
            except ProviderError as exc:
                raise StudyPlanProviderError(str(exc)) from exc

            try:
                raw_payload = _extract_json_object(response.content)
                parsed = _validate_model_output(raw_payload)
                (
                    discussion_question_notes,
                    reflection_question_notes,
                ) = _validate_optional_question_notes(
                    parsed,
                    include_question_notes=include_question_notes,
                )
                usage = UsageMetrics(
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                )
                return (
                    parsed,
                    discussion_question_notes,
                    reflection_question_notes,
                    usage,
                    response.model,
                )
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                if attempt == 0:
                    messages = build_repair_messages(
                        base_messages,
                        response.content,
                        include_question_notes=include_question_notes,
                    )
                    continue
                raise StudyPlanValidationError("Model output did not match study-plan schema.") from exc
            except Exception as exc:
                if attempt == 0:
                    messages = build_repair_messages(
                        base_messages,
                        response.content,
                        include_question_notes=include_question_notes,
                    )
                    continue
                raise StudyPlanValidationError("Model output did not match study-plan schema.") from exc

        raise StudyPlanValidationError("Failed to generate a valid study plan.")
