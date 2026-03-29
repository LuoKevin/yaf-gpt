from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_VOICE_GENERATION_MODEL = "gpt-4o-mini-tts"
SUPPORTED_VOICE_OPTIONS = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
}
SUPPORTED_RESPONSE_FORMATS = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}
MODELS_WITHOUT_INSTRUCTIONS = {"tts-1", "tts-1-hd"}
MAX_INPUT_LENGTH = 4096


@dataclass(frozen=True)
class GenerateVoiceCommand:
    input_text: str
    voice: str = "alloy"
    instructions: Optional[str] = None
    response_format: Optional[str] = None
    speed: float = 1.0


@dataclass(frozen=True)
class ResolvedGenerateVoiceCommand:
    input_text: str
    voice: str
    instructions: Optional[str]
    response_format: str
    speed: float


@dataclass(frozen=True)
class VoiceGenerationResult:
    audio_bytes: bytes
    mime_type: str
    model: str
    voice: str
    response_format: str


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def resolve_generate_voice_command(
    command: GenerateVoiceCommand,
    *,
    model_name: str,
    default_response_format: str = "mp3",
) -> ResolvedGenerateVoiceCommand:
    input_text = normalize_text(command.input_text)
    if not input_text:
        raise ValueError("Input text cannot be empty.")
    if len(input_text) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input text cannot exceed {MAX_INPUT_LENGTH} characters."
        )

    voice = normalize_text(command.voice).lower() or "alloy"
    if voice not in SUPPORTED_VOICE_OPTIONS:
        raise ValueError(f"Unsupported voice '{voice}'.")

    response_format = normalize_text(command.response_format).lower() or default_response_format
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise ValueError(f"Unsupported response format '{response_format}'.")

    instructions = normalize_text(command.instructions) or None
    if instructions and model_name in MODELS_WITHOUT_INSTRUCTIONS:
        raise ValueError(
            f"Model '{model_name}' does not support voice instructions."
        )

    speed = float(command.speed)
    if speed < 0.25 or speed > 4.0:
        raise ValueError("Voice generation speed must be between 0.25 and 4.0.")

    return ResolvedGenerateVoiceCommand(
        input_text=input_text,
        voice=voice,
        instructions=instructions,
        response_format=response_format,
        speed=speed,
    )
