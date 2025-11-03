"""Integration tests for FastAPI routes."""

from fastapi.testclient import TestClient

from yaf_gpt.api.api import create_app
from yaf_gpt.pipeline.chat_service import ChatMessage, ChatRequest, ChatResponse, ChatService


class FakeChatService(ChatService):
    """Overrides the LLM layer to stay offline for tests."""

    def __init__(self) -> None:
        super().__init__(llm_client=None)

    def generate_reply(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role="assistant", content="Offline response from test service.")
        )


def test_chat_endpoint_returns_assistant_message() -> None:
    """POST /chat should return a canned assistant reply."""
    app = create_app(chat_service=FakeChatService())
    client = TestClient(app)

    payload = {"messages": [{"role": "user", "content": "Who are you?"}]}

    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"] == "Offline response from test service."
