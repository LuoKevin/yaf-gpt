"""FastAPI application wiring for yaf_gpt."""

from __future__ import annotations
import logging

from fastapi import FastAPI, Request
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from langchain_core.runnables import Runnable

from yaf_gpt.llm.bible_study_helper import BibleStudyHelper
from yaf_gpt.scripts.langchain.build_runnable import build_runnable
# from yaf_gpt.scripts.langchain.ingest_documents import ingest_documents
from yaf_gpt.core.config import Settings

class ChatMessage(BaseModel):
    """Single chat message with a role and content."""
    role: str = Field(..., description="Role of the message sender (e.g., 'user' or 'assistant')")
    content: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    """Incoming request payload containing the conversation history."""

    messages: list[ChatMessage]

    @model_validator(mode="after")
    def ensure_user_message(self) -> "ChatRequest":
        if not self.messages:
            raise ValueError("At least one message is required to generate a reply.")
        return self

class ChatResponse(BaseModel):
    """Response payload wrapping the assistant's reply."""

    message: ChatMessage

class StudyNotesRequest(BaseModel):
    """Incoming request payload for study notes generation."""
    reference: str = Field(..., description="Bible passage reference (e.g., 'John 3:16')")


def create_app(config: Settings | None = None) -> FastAPI:
    """Application factory with all routes registered."""
    settings = config if config else Settings()
    # runnable : Runnable = build_runnable(retriever=ingest_documents(config=settings), config=settings)
    openai = OpenAI(api_key=settings.OPENAI_API_KEY)
    study_helper: BibleStudyHelper = BibleStudyHelper(client=openai)


    app = FastAPI(title="yaf-gpt", version="0.0.2")

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    # @app.post("/chat", tags=["chat"], response_model=ChatResponse)
    # async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    #     """Accepts chat messages and returns the assistant reply."""
    #     return runnable.invoke({"question": request.message})

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger = logging.getLogger("yaf_gpt")
        logger.info(f"Incoming request: {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
   
    @app.get("/study_notes")
    async def get_study_notes(request: StudyNotesRequest):
        reference =request.reference
        ret_val = study_helper.study(reference=reference)
        return {"study_notes": ret_val.choices[0].message.content}
    
    return app

settings = Settings()

app = create_app()

