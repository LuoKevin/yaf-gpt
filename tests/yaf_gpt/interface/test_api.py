"""Integration tests for FastAPI routes."""

from fastapi.testclient import TestClient

from yaf_gpt.interface.api import create_app


def test_chat_endpoint_returns_assistant_message() -> None:
    """POST /chat should return a stubbed assistant reply."""
    app = create_app()
    client = TestClient(app)

    payload = {"messages": [{"role": "user", "content": "Who are you?"}]}

    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["message"]["role"] == "assistant"
    assert "stub" in body["message"]["content"].lower()
