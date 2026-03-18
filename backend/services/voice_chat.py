from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .bible_lookup import BibleAPIProvider, BibleProvider

if TYPE_CHECKING:
    from openai import OpenAI


DEFAULT_VOICE_CHAT_MODEL = "gpt-realtime-mini"
DEFAULT_VOICE_CHAT_VOICE = "cedar"
DEFAULT_VOICE_CHAT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_VOICE_CHAT_SECRET_TTL_SECONDS = 60
DEFAULT_VOICE_CHAT_WEBRTC_URL = "https://api.openai.com/v1/realtime/calls"

CALVINIST_BIBLE_STUDY = (
    "You are a concise, respectful assistant specializing in Christian "
    "Bible study from a Reformed/Calvinist perspective. "
    "Use clear, plain language. "
    "Prioritize Scripture, and cite references when relevant. "
    "If interpretations differ across traditions, briefly note that and "
    "state the Reformed view. "
    "Do not claim certainty when unsure. "
    "Avoid pastoral counseling; encourage speaking with a pastor for "
    "personal guidance when appropriate."
)

YOUNG_ADULT_COMMUNICATION = (
    "Communicate in a warm, thoughtful, and relatable way for young adults. "
    "Use clear, contemporary language without slang overload. "
    "Ask gentle clarifying questions when needed. "
    "Keep responses concise and actionable. "
    "Be respectful and avoid condescension."
)

_ENV_LOADED = False


def _load_backend_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)
    _ENV_LOADED = True


def _parse_positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _build_persona_system_prompt() -> str:
    return (
        f"{CALVINIST_BIBLE_STUDY} "
        f"{YOUNG_ADULT_COMMUNICATION} "
        "You are a mentor-style discussion partner for young adults. "
        "Balance biblical faithfulness with practical clarity. "
        "When using lists, format each item on its own new line for readability. "
        "Stay concise, ask one helpful follow-up question when useful, and avoid sounding preachy. "
        "If users ask for personal counseling or crisis help, encourage speaking with a pastor or trusted local leader."
    )


def _build_reference_context_block(normalized_reference: str, translation: str, passage_text: str) -> str:
    excerpt = " ".join(passage_text.split())
    if len(excerpt) > 1200:
        excerpt = f"{excerpt[:1197].rstrip()}..."

    return (
        "Use this passage context when relevant:\n"
        f"Reference: {normalized_reference} ({translation})\n"
        f"Passage:\n{excerpt}"
    )


@dataclass(frozen=True)
class VoiceRealtimeSession:
    client_secret: str
    expires_at: int
    model: str
    voice: str
    webrtc_url: str


class VoiceChatService:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        client: Optional["OpenAI"] = None,
        bible_provider: BibleProvider | None = None,
        model: str | None = None,
        transcription_model: str | None = None,
        default_voice: str | None = None,
        webrtc_url: str | None = None,
        secret_ttl_seconds: int | None = None,
    ) -> None:
        _load_backend_env()

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if client is None:
            if not resolved_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                raise RuntimeError("openai package is not installed") from exc
            self._client = OpenAI(api_key=resolved_key)
        else:
            self._client = client

        self._bible_provider = bible_provider or BibleAPIProvider()
        self.model_name = model or os.getenv("VOICE_CHAT_REALTIME_MODEL") or DEFAULT_VOICE_CHAT_MODEL
        self.transcription_model = (
            transcription_model
            or os.getenv("VOICE_CHAT_TRANSCRIPTION_MODEL")
            or DEFAULT_VOICE_CHAT_TRANSCRIPTION_MODEL
        )
        self.default_voice = default_voice or os.getenv("VOICE_CHAT_VOICE") or DEFAULT_VOICE_CHAT_VOICE
        self.webrtc_url = webrtc_url or os.getenv("VOICE_CHAT_WEBRTC_URL") or DEFAULT_VOICE_CHAT_WEBRTC_URL
        env_ttl = _parse_positive_int(
            os.getenv("VOICE_CHAT_SECRET_TTL_SECONDS"),
            DEFAULT_VOICE_CHAT_SECRET_TTL_SECONDS,
        )
        self.secret_ttl_seconds = secret_ttl_seconds or env_ttl

    def _build_instructions(self, *, reference_context: str | None, translation: str) -> str:
        instructions = [
            _build_persona_system_prompt(),
            (
                "You are speaking in a live voice conversation. "
                "Sound natural, grounded, and conversational. "
                "Keep most replies concise unless the user asks for more depth. "
                "Avoid markdown, headings, or reading punctuation aloud."
            ),
        ]

        if reference_context and reference_context.strip():
            passage = self._bible_provider.get_passage(
                reference=reference_context.strip(),
                translation=translation,
            )
            instructions.append(
                _build_reference_context_block(
                    normalized_reference=passage.normalized_reference,
                    translation=passage.translation,
                    passage_text=passage.text,
                )
            )
            instructions.append("Ground your spoken answers in that passage when it is relevant.")

        return "\n\n".join(instructions)

    def create_realtime_session(
        self,
        *,
        reference_context: str | None = None,
        translation: str = "WEB",
        voice: str | None = None,
    ) -> VoiceRealtimeSession:
        resolved_voice = (voice or self.default_voice).strip()
        if not resolved_voice:
            raise ValueError("Voice is required.")

        instructions = self._build_instructions(
            reference_context=reference_context,
            translation=translation,
        )

        try:
            response = self._client.realtime.client_secrets.create(
                expires_after={
                    "anchor": "created_at",
                    "seconds": self.secret_ttl_seconds,
                },
                session={
                    "type": "realtime",
                    "model": self.model_name,
                    "instructions": instructions,
                    "max_output_tokens": 700,
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "noise_reduction": {"type": "near_field"},
                            "transcription": {
                                "language": "en",
                                "model": self.transcription_model,
                                "prompt": "Expect Bible references, theology terms, and Christian discipleship language.",
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "create_response": True,
                                "interrupt_response": True,
                                "prefix_padding_ms": 300,
                                "silence_duration_ms": 500,
                                "threshold": 0.5,
                            },
                        },
                        "output": {
                            "voice": resolved_voice,
                        },
                    },
                },
            )
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        session = response.session
        session_audio = getattr(session, "audio", None)
        session_output = getattr(session_audio, "output", None)
        session_voice = getattr(session_output, "voice", None) or resolved_voice

        return VoiceRealtimeSession(
            client_secret=response.value,
            expires_at=response.expires_at,
            model=getattr(session, "model", None) or self.model_name,
            voice=session_voice,
            webrtc_url=self.webrtc_url,
        )
