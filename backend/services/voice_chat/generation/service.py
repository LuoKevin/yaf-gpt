from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .domain import GenerateVoiceCommand, VoiceGenerationResult, resolve_generate_voice_command
from .providers import (
    OpenAIVoiceGenerationProvider,
    build_voice_generation_provider_from_env,
    mime_type_for_format,
)

if TYPE_CHECKING:
    from openai import OpenAI


class VoiceGenerationService:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        client: Optional["OpenAI"] = None,
        model: Optional[str] = None,
        provider: Optional[object] = None,
    ) -> None:
        if provider is not None:
            self._provider = provider
        elif client is not None or api_key is not None or model is not None:
            self._provider = OpenAIVoiceGenerationProvider(
                api_key=api_key,
                client=client,
                model=model,
            )
        else:
            self._provider = build_voice_generation_provider_from_env()

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    def generate_audio(
        self,
        *,
        input_text: str,
        voice: str = "alloy",
        instructions: Optional[str] = None,
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> VoiceGenerationResult:
        resolved = resolve_generate_voice_command(
            GenerateVoiceCommand(
                input_text=input_text,
                voice=voice,
                instructions=instructions,
                response_format=response_format,
                speed=speed,
            ),
            model_name=self._provider.model_name,
        )
        audio_bytes = self._provider.generate_audio(
            input_text=resolved.input_text,
            voice=resolved.voice,
            instructions=resolved.instructions,
            response_format=resolved.response_format,
            speed=resolved.speed,
        )
        return VoiceGenerationResult(
            audio_bytes=audio_bytes,
            mime_type=mime_type_for_format(resolved.response_format),
            model=self._provider.model_name,
            voice=resolved.voice,
            response_format=resolved.response_format,
        )
