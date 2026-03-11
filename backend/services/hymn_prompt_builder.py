from __future__ import annotations

import json

from backend.app.schemas import HymnLyrics
from backend.llm import ChatMessage
from backend.llm.system_prompts import CALVINIST_BIBLE_STUDY, YOUNG_ADULT_COMMUNICATION


def _hymn_schema_json() -> str:
    if hasattr(HymnLyrics, "model_json_schema"):
        schema = HymnLyrics.model_json_schema()
    else:
        schema = HymnLyrics.schema()
    return json.dumps(schema, indent=2)


def build_hymn_messages(
    *,
    reference: str,
    normalized_reference: str,
    translation: str,
    passage_text: str,
    style_hint: str,
    mood_hint: str | None,
    user_notes: str | None,
) -> list[ChatMessage]:
    mood_text = mood_hint.strip() if mood_hint else "Not provided."
    notes_text = user_notes.strip() if user_notes else "Not provided."
    system_prompt = (
        f"{CALVINIST_BIBLE_STUDY} "
        f"{YOUNG_ADULT_COMMUNICATION} "
        "You write singable Christian hymn lyrics that are Scripture-grounded and theologically careful. "
        "Do not claim direct divine revelation. "
        "Avoid loaded language that implies salvation by works."
    )
    user_prompt = (
        "Create hymn lyrics from this passage and return JSON that strictly matches the schema.\n"
        f"Reference: {reference}\n"
        f"Normalized Reference: {normalized_reference}\n"
        f"Translation: {translation}\n"
        f"Style Hint: {style_hint}\n"
        f"Mood Hint: {mood_text}\n"
        f"User Notes: {notes_text}\n"
        f"Passage Text:\n{passage_text}\n\n"
        "Requirements:\n"
        "- Include a memorable title.\n"
        "- Include at least Verse 1, Chorus, and Verse 2 sections.\n"
        "- Keep lyrics singable and concrete, with imagery tied to the passage.\n"
        "- Scripture references must directly relate to this passage.\n"
        "- Return valid JSON only (no markdown fences).\n\n"
        f"JSON Schema:\n{_hymn_schema_json()}\n"
    )

    return [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]


def build_hymn_repair_messages(messages: list[ChatMessage], invalid_output: str) -> list[ChatMessage]:
    repair_prompt = (
        "Reformat your prior answer into valid JSON that strictly matches the schema. "
        "Keep the same intent, include Verse and Chorus sections, and return JSON only."
    )
    return messages + [
        ChatMessage(role="assistant", content=invalid_output),
        ChatMessage(role="user", content=repair_prompt),
    ]
