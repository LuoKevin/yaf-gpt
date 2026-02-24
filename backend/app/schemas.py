from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


TranslationCode = Literal["WEB", "KJV"]
ImageStyle = Literal["modern_editorial_illustration"]
ChatRole = Literal["user", "assistant"]


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


class StudyPlanSection(BaseModel):
    heading: str
    content: str
    scripture_references: list[str] = Field(default_factory=list)


class StudyPlanLLMOutput(BaseModel):
    passage_title: str = Field(..., min_length=1)
    context_points: list[str] = Field(..., min_length=1)
    discussion_questions: list[str] = Field(..., min_length=6, max_length=6)


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
    model: str
    usage: Optional[UsageMetrics] = None


class PersonaChatResponse(BaseModel):
    reply: str
    model: str
    usage: Optional[UsageMetrics] = None
