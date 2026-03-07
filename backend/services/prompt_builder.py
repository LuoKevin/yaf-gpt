from __future__ import annotations

import json

from backend.app.schemas import StudyPlanLLMOutput
from backend.llm import ChatMessage
from backend.llm.system_prompts import CALVINIST_BIBLE_STUDY, YOUNG_ADULT_COMMUNICATION

from .style_guide import LukeStyleGuide

DEFAULT_QUESTION_COUNT = 6
MAX_REFLECTION_QUESTION_COUNT = 3
DEFAULT_GROUP_SIZE = "3-5 participants plus 1 discussion leader"
DEFAULT_SESSION_DURATION_MINUTES = 60


def _schema_json() -> str:
    if hasattr(StudyPlanLLMOutput, "model_json_schema"):
        schema = StudyPlanLLMOutput.model_json_schema()
    else:
        schema = StudyPlanLLMOutput.model_json_schema()
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
        f"- Include 1 to {MAX_REFLECTION_QUESTION_COUNT} reflection questions in a separate reflection_questions section.\n"
        f"- Design for a {DEFAULT_SESSION_DURATION_MINUTES}-minute study with {DEFAULT_GROUP_SIZE}.\n"
        "- Questions must follow the passage flow from beginning to end.\n"
        "- Questions should be plain question strings only. Do not include intent or follow-up fields.\n"
        "- discussion_questions must focus solely on understanding and discussing the passage text.\n"
        "- reflection_questions must be at the end and focus on personal reflection, application, and life lessons.\n"
        "- Every reflection question must directly anchor to this passage by naming a specific verse reference or phrase from the text.\n"
        "- Avoid generic reflection prompts that could fit any passage.\n"
        "- Each question should be open-ended, text-anchored, and promote active group discussion.\n"
        "- Keep context points historically and textually grounded.\n"
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
        f"Include 1 to {MAX_REFLECTION_QUESTION_COUNT} reflection questions in reflection_questions. "
        "Ensure questions move through the passage in order and are discussion-oriented for a 60-minute group. "
        "Use plain question strings only; no intent or follow-up fields. "
        "Keep discussion_questions passage-focused and reflection_questions application-focused. "
        "Each reflection question must explicitly reference the passage text (specific verse or phrase), not generic themes. "
        "Return JSON only with no markdown fences."
    )
    return messages + [
        ChatMessage(role="assistant", content=invalid_output),
        ChatMessage(role="user", content=repair_instruction),
    ]
