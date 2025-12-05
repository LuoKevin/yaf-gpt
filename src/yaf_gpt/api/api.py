"""FastAPI application wiring for yaf_gpt."""

from __future__ import annotations
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from langchain_core.runnables import Runnable

from src.yaf_gpt.client.bible_study_helper import BibleStudyHelper
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


    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["http://localhost:5173"],  # adjust as needed
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
    async def get_study_notes(reference: str):
        ret_val = study_helper.study(reference=reference)
        return {"study_notes": ret_val.choices[0].message.content}
        # return {"study_notes": "**PASSAGE**  \nLuke 13:20–21 (CSB)\n\n20 [Jesus:] \"To what is the kingdom of God like? To what can I compare it?\"  \n21 [Jesus:] \"It is like yeast that a woman took and hid in three measures of flour until the whole batch was leavened.\"\n\n###\n**ICE BREAKER**\nWhen have you seen a small, hidden thing make a big difference (a tiny habit, a comment, a gift)?\n\n###\n**CONTEXT**\n- Short parable about the Kingdom of God (paired with the mustard seed parable nearby).  \n- Yeast (leaven) was common household imagery: a little causes the whole dough to rise and transform.  \n- \"Three measures\" = a large batch of flour (enough for many loaves), so the leaven’s effect is widespread.  \n- Emphasis: quiet, internal, pervasive growth — not always flashy or immediate.\n\n###\n**QUESTIONS**\n\nWhat element of the kingdom does the yeast emphasize?  \n- Growth from within; small beginnings becoming shaping, pervasive influence.\n\nWhy a woman and why “hid” the yeast?  \n- Domestic setting makes the image relatable; “hid” suggests the kingdom often works subtly, unseen, inside communities and hearts.\n\nWhat does “the whole batch was leavened” tell us about the end result?  \n- The influence becomes complete and communal — not isolated; one small input changes the whole.\n\nHow does this parable contrast with expectations of power or spectacle?  \n- It downplays loud, obvious signs; God’s kingdom advances in patient, ordinary, and internal ways.\n\nWhere might you be expecting dramatic results instead of steady, hidden work?  \n- Personal growth, friendships, faith formation, workplace/college witness — notice impatience for quick fixes.\n\n###\n**LIFE APPLICATION**\n\n- Start one tiny spiritual habit this week (5 minutes of prayer, a daily gratitude note, reading one verse) and watch for slow change.  \n- Invest in one relationship with steady, ordinary presence instead of trying to \"make\" big outcomes.  \n- Serve quietly: small acts of kindness, listening, or consistent generosity can “leaven” your community.  \n- Practice patience: journal where you’ve seen small things grow over months — trust God’s hidden work."}
    
    return app

settings = Settings()

app = create_app()

