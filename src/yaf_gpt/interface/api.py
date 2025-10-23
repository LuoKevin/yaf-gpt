"""FastAPI application wiring for yaf_gpt."""

from __future__ import annotations

from fastapi import FastAPI

from yaf_gpt.config import Settings
from yaf_gpt.model.llm_client import OpenAIChatClient
from yaf_gpt.pipeline.chat_service import ChatRequest, ChatResponse, ChatService


def create_app(chat_service: ChatService | None = None, config: Settings = Settings()) -> FastAPI:
    """Application factory with all routes registered."""
    openai_key = config.OPENAI_API_KEY
    openai_client = OpenAIChatClient(api_key=openai_key) if openai_key else None

    service: ChatService = chat_service or ChatService(llm_client=openai_client)
    app = FastAPI(title="yaf-gpt", version="0.0.1")

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", tags=["chat"], response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest) -> ChatResponse:
        """Accepts chat messages and returns the assistant reply."""
        return service.generate_reply(request)

    return app


app = create_app()
