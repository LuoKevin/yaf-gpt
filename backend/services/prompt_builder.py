from __future__ import annotations

import json

from backend.app.schemas import StudyPlanLLMOutput
from backend.llm import ChatMessage
from backend.llm.system_prompts import CALVINIST_BIBLE_STUDY, YOUNG_ADULT_COMMUNICATION

from .style_guide import LukeStyleGuide

DEFAULT_QUESTION_COUNT = 6


def _schema_json() -> str:
    if hasattr(StudyPlanLLMOutput, "model_json_schema"):
        schema = StudyPlanLLMOutput.model_json_schema()
    else:
        schema = StudyPlanLLMOutput.schema()
    return json.dumps(schema, indent=2)


def build_study_plan_messages(
    *,
    reference: str,
    normalized_reference: str,
    translation: str,
    passage_text: str,
    style_guide: LukeStyleGuide,
    goals: str | None,
    user_notes: str | None,
) -> list[ChatMessage]:
    goals_text = goals.strip() if goals else "Not provided."
    notes_text = user_notes.strip() if user_notes else "Not provided."

    system_prompt = (
        f"{CALVINIST_BIBLE_STUDY} "
        f"{YOUNG_ADULT_COMMUNICATION} "
        "You generate Bible study plans for young-adult small groups. "
        "Produce concise, practical outputs grounded in Scripture. "
        "When doctrine differs, present the Reformed view first and briefly note other orthodox views. "
        "Do not include pastoral counseling; point users toward a pastor when personal guidance is needed."
    )

    user_prompt = (
        "Generate a structured Bible study plan using the exact JSON schema below.\n"
        f"Reference: {reference}\n"
        f"Normalized Reference: {normalized_reference}\n"
        f"Translation: {translation}\n"
        f"Passage Text:\n{passage_text}\n\n"
        f"Goals: {goals_text}\n"
        f"User Notes: {notes_text}\n\n"
        f"{style_guide.to_instruction_block()}\n"
        f"- Include exactly {DEFAULT_QUESTION_COUNT} discussion questions.\n"
        "- Keep context points historically and textually grounded.\n"
        "- Leader notes should include facilitation tips and theological caution where relevant.\n"
        "- Return valid JSON only. No markdown fences, no extra keys, no prose outside JSON.\n\n"
        f"JSON Schema:\n{_schema_json()}\n"
    )

    return [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]


def build_repair_messages(messages: list[ChatMessage], invalid_output: str) -> list[ChatMessage]:
    repair_instruction = (
        "Reformat your previous answer into valid JSON that strictly matches the required schema. "
        f"Include exactly {DEFAULT_QUESTION_COUNT} discussion questions. "
        "Return JSON only with no markdown fences."
    )
    return messages + [
        ChatMessage(role="assistant", content=invalid_output),
        ChatMessage(role="user", content=repair_instruction),
    ]

