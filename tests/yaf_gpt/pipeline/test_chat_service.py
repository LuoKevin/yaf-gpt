"""Tests for the chat service pipeline layer."""

from yaf_gpt.pipeline.chat_service import ChatMessage, ChatRequest, ChatService


def test_chat_service_returns_stubbed_response() -> None:
    """Service should produce a deterministic assistant message."""
    service = ChatService()
    request = ChatRequest(messages=[ChatMessage(role="user", content="Hello there!")])

    response = service.generate_reply(request)

    assert response.message.role == "assistant"
    assert "stub" in response.message.content.lower()
