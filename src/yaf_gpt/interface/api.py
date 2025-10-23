"""FastAPI application wiring for yaf_gpt."""

from fastapi import FastAPI

from yaf_gpt.pipeline.chat_service import ChatRequest, ChatResponse, ChatService


def create_app() -> FastAPI:
    """Application factory with all routes registered."""
    app = FastAPI(title="yaf-gpt", version="0.0.1")
    chat_service = ChatService()

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", tags=["chat"], response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest) -> ChatResponse:
        """Accepts chat messages and returns the assistant reply."""
        return chat_service.generate_reply(request)

    return app


app = create_app()
