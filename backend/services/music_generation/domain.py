from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_TRACK_TITLE = "Generated Track"


@dataclass(frozen=True)
class GenerateMusicCommand:
    prompt: str
    style_hint: str
    mood_hint: Optional[str] = None
    title: Optional[str] = None


@dataclass(frozen=True)
class ResolvedGenerateMusicCommand:
    title: str
    prompt: str
    style_hint: str
    mood_hint: Optional[str]


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def derive_title(prompt: str) -> str:
    words = prompt.split()
    if not words:
        return DEFAULT_TRACK_TITLE
    return " ".join(words[:6]).title()


def resolve_generate_music_command(command: GenerateMusicCommand) -> ResolvedGenerateMusicCommand:
    prompt = normalize_text(command.prompt)
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    style_hint = normalize_text(command.style_hint)
    if not style_hint:
        raise ValueError("Style hint cannot be empty.")

    mood_hint = normalize_text(command.mood_hint) or None
    title = normalize_text(command.title) or derive_title(prompt)

    return ResolvedGenerateMusicCommand(
        title=title,
        prompt=prompt,
        style_hint=style_hint,
        mood_hint=mood_hint,
    )
