"""FastAPI application wiring for yaf_gpt."""

from __future__ import annotations
import logging

from fastapi import FastAPI, Request

from scripts.langchain.build_runnable import build_chain
from scripts.langchain.ingest_documents import ingest_documents
from yaf_gpt.config import Settings
from yaf_gpt.model.llm_client import OpenAIChatClient
from yaf_gpt.pipeline.chat_service import ChatRequest, ChatResponse, ChatService


def create_app(chat_service: ChatService | None = None, config: Settings | None = None) -> FastAPI:
    """Application factory with all routes registered."""
    settings = config if config else Settings()
    openai_key = settings.OPENAI_API_KEY
    llm_client = OpenAIChatClient(api_key=openai_key) if openai_key else None
    runnable = build_chain(retriever=ingest_documents(config=settings), config=settings)

    service: ChatService = chat_service or ChatService(llm_client=llm_client, config=settings.chat)
    app = FastAPI(title="yaf-gpt", version="0.0.2")

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", tags=["chat"], response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest) -> ChatResponse:
        """Accepts chat messages and returns the assistant reply."""
        return service.generate_reply(request)

    @app.post("/chat-langchain", tags=["chat"], response_model=ChatResponse)
    async def chat_langchain_endpoint(request: ChatRequest) -> ChatResponse:
        """Accepts chat messages and returns the assistant reply."""
        return runnable.invoke({"question": request.message})

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger = logging.getLogger("yaf_gpt")
        logger.info(f"Incoming request: {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response

    return app

settings = Settings()

app = create_app()

