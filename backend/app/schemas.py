from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


TranslationCode = Literal["WEB", "KJV"]
ImageStyle = Literal["modern_editorial_illustration"]
ChatRole = Literal["user", "assistant"]
HymnJobStatus = Literal["queued", "in_progress", "completed", "failed"]
VoiceGenerationVoice = Literal[
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
]
VoiceGenerationFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
RealtimeVoice = Literal[
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "marin",
    "sage",
    "shimmer",
    "verse",
]


class APIErrorResponse(BaseModel):
    detail: str = Field(..., description="User-safe error message.")
    request_id: Optional[str] = Field(
        default=None,
        description="Optional request identifier for troubleshooting.",
    )


class BiblePassageQuery(BaseModel):
    reference: str = Field(..., min_length=1, description="Bible reference, e.g. John 3:16-18")
    translation: TranslationCode = Field(default="WEB")


class BibleVerse(BaseModel):
    book: str
    chapter: int
    verse: int
    text: str


class BiblePassageResponse(BaseModel):
    reference: str
    translation: TranslationCode = Field(default="WEB")
    normalized_reference: str
    text: str
    verses: list[BibleVerse]


class StudyPlanRequest(BaseModel):
    reference: str = Field(..., min_length=1, description="Bible reference, e.g. Romans 8:28-39")
    translation: TranslationCode = Field(default="WEB")
    passage_text: Optional[str] = Field(
        default=None,
        description="Optional passage text override. If absent, the API looks up text by reference.",
    )
    goals: Optional[str] = Field(
        default=None,
        description="Optional learner goals to tailor the study guide.",
    )
    user_notes: Optional[str] = Field(
        default=None,
        description="Optional context about the learner or study constraints.",
    )
    include_question_notes: bool = Field(
        default=False,
        description=(
            "If true, include short leader-note hints associated with each discussion/reflection "
            "question."
        ),
    )


class StudyPlanSection(BaseModel):
    heading: str
    content: str
    scripture_references: list[str] = Field(default_factory=list)


class StudyPlanLLMOutput(BaseModel):
    passage_title: str = Field(..., min_length=1)
    context_points: list[str] = Field(..., min_length=1)
    discussion_questions: list[str] = Field(..., min_length=6, max_length=6)
    reflection_questions: list[str] = Field(..., min_length=1, max_length=3)
    discussion_question_notes: Optional[list[str]] = Field(
        default=None,
        description="Optional short leader notes matching discussion_questions by index.",
    )
    reflection_question_notes: Optional[list[str]] = Field(
        default=None,
        description="Optional short leader notes matching reflection_questions by index.",
    )


class PassageImageRequest(BaseModel):
    reference: str = Field(..., min_length=1, description="Bible reference, e.g. Psalm 23:1-4")
    translation: TranslationCode = Field(default="WEB")
    style: ImageStyle = Field(default="modern_editorial_illustration")


class PassageImageResponse(BaseModel):
    reference: str
    translation: TranslationCode = Field(default="WEB")
    style: ImageStyle = Field(default="modern_editorial_illustration")
    prompt_used: str
    image_b64_or_url: str = Field(
        ...,
        description="Either a base64 payload or an image URL returned by the provider.",
    )
    alt_text: str


class PersonaChatMessage(BaseModel):
    role: ChatRole
    content: str = Field(..., min_length=1)


class PersonaChatRequest(BaseModel):
    messages: list[PersonaChatMessage] = Field(
        ...,
        min_length=1,
        description="Conversation history in order.",
    )
    reference_context: Optional[str] = Field(
        default=None,
        description="Optional passage reference to ground the response.",
    )
    translation: TranslationCode = Field(default="WEB")


class UsageMetrics(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class StudyPlanResponse(BaseModel):
    reference: str
    normalized_reference: str
    translation: TranslationCode = Field(default="WEB")
    passage_text: str
    passage_title: str
    context_points: list[str] = Field(..., min_length=1)
    discussion_questions: list[str] = Field(..., min_length=6, max_length=6)
    reflection_questions: list[str] = Field(..., min_length=1, max_length=3)
    include_question_notes: bool = False
    discussion_question_notes: Optional[list[str]] = None
    reflection_question_notes: Optional[list[str]] = None
    model: str
    usage: Optional[UsageMetrics] = None


class PersonaChatResponse(BaseModel):
    reply: str
    model: str
    usage: Optional[UsageMetrics] = None


class VoiceTranscriptionRequest(BaseModel):
    audio_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded audio bytes, optionally as a data URL.",
    )
    mime_type: Optional[str] = Field(
        default=None,
        description="Optional MIME type for the uploaded audio.",
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Optional file name hint for transcription provider compatibility.",
    )


class VoiceTranscriptionResponse(BaseModel):
    transcript: str
    model: str


class VoiceChatTurnRequest(BaseModel):
    audio_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded recorded audio bytes, optionally as a data URL.",
    )
    mime_type: Optional[str] = Field(
        default=None,
        description="Optional MIME type for the uploaded audio.",
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Optional file name hint for transcription provider compatibility.",
    )
    reference_context: Optional[str] = Field(
        default=None,
        description="Optional Bible reference used to ground the voice reply.",
    )
    translation: TranslationCode = Field(default="WEB")


class VoiceChatTurnResponse(BaseModel):
    transcript: str
    transcript_model: str
    reply: str
    reply_model: str
    audio_base64: Optional[str] = None
    audio_mime_type: Optional[str] = None
    audio_model: Optional[str] = None
    audio_voice: Optional[str] = None
    audio_response_format: Optional[str] = None


class VoiceGenerationRequest(BaseModel):
    input: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Text to synthesize into spoken audio.",
    )
    voice: VoiceGenerationVoice = Field(default="alloy")
    instructions: Optional[str] = Field(
        default=None,
        description="Optional speaking guidance such as tone or delivery style.",
    )
    response_format: Optional[VoiceGenerationFormat] = Field(
        default=None,
        description="Optional output format. If omitted, the active provider chooses its default.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Playback speed multiplier for synthesized audio.",
    )


class VoiceGenerationResponse(BaseModel):
    audio_base64: str
    mime_type: str
    model: str
    voice: VoiceGenerationVoice
    response_format: VoiceGenerationFormat


class VoiceRealtimeSessionRequest(BaseModel):
    reference_context: Optional[str] = Field(
        default=None,
        description="Optional Bible reference used to ground the live voice conversation.",
    )
    translation: TranslationCode = Field(default="WEB")
    voice: RealtimeVoice = Field(default="cedar")


class VoiceRealtimeSessionResponse(BaseModel):
    client_secret: str
    expires_at: int
    model: str
    voice: RealtimeVoice
    webrtc_url: str


class HymnSection(BaseModel):
    label: str = Field(..., min_length=1, description="Section label, for example Verse 1 or Chorus.")
    lyrics: str = Field(..., min_length=1)


class HymnLyrics(BaseModel):
    title: str = Field(..., min_length=1)
    theme: str = Field(..., min_length=1)
    scripture_references: list[str] = Field(..., min_length=1, max_length=6)
    sections: list[HymnSection] = Field(..., min_length=2, max_length=8)


class HymnGenerateRequest(BaseModel):
    reference: str = Field(..., min_length=1, description="Bible reference, e.g. Psalm 23:1-6")
    translation: TranslationCode = Field(default="WEB")
    passage_text: Optional[str] = Field(
        default=None,
        description="Optional passage text override. If absent, the API looks up text by reference.",
    )
    style_hint: str = Field(
        default="modern worship hymn, acoustic",
        min_length=3,
        description="High-level musical style hint used for lyric and music generation.",
    )
    mood_hint: Optional[str] = Field(
        default=None,
        description="Optional mood hint, for example hopeful, reflective, triumphant.",
    )
    user_notes: Optional[str] = Field(
        default=None,
        description="Optional constraints or context for lyric generation.",
    )


class HymnGenerateResponse(BaseModel):
    reference: str
    normalized_reference: str
    translation: TranslationCode = Field(default="WEB")
    passage_text: str
    hymn: HymnLyrics
    job_id: str
    job_status: HymnJobStatus
    provider: str
    model: str
    usage: Optional[UsageMetrics] = None


class HymnJobResponse(BaseModel):
    job_id: str
    status: HymnJobStatus
    provider: str
    audio_url: Optional[str] = None
    error: Optional[str] = None


class MusicGenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=3,
        description="User text prompt or lyrics seed for music generation.",
    )
    style: str = Field(
        default="modern worship, acoustic",
        min_length=3,
        description="High-level style direction sent to the music provider.",
    )
    mood: Optional[str] = Field(
        default=None,
        description="Optional mood direction, for example hopeful, reflective, triumphant.",
    )


class MusicGenerateResponse(BaseModel):
    job_id: str
    status: HymnJobStatus
    provider: str
    title: str
    prompt: str


class MusicJobResponse(BaseModel):
    job_id: str
    status: HymnJobStatus
    provider: str
    audio_url: Optional[str] = None
    error: Optional[str] = None
