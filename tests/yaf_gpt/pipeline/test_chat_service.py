"""Tests for the chat service pipeline layer."""
import logging

import pytest

from yaf_gpt.config import ChatConfig
from yaf_gpt.pipeline.chat_service import ChatMessage, ChatRequest, ChatService


class FakeLLMClient:
    """Minimal fake to capture the prompt and return canned text."""

    def __init__(self) -> None:
        self.captured_messages: list[ChatMessage] | None = None

    def complete(self, messages: list[ChatMessage]) -> str:
        self.captured_messages = messages
        return "Assistant reply from fake llm."


def test_chat_service_uses_llm_client() -> None:
    """Service should delegate to the provided LLM client."""
    llm = FakeLLMClient()
    service = ChatService(llm_client=llm)
    request = ChatRequest(messages=[ChatMessage(role="user", content="Hello there!")])
    response = service.generate_reply(request)

    assert llm.captured_messages is not None
    assert llm.captured_messages[-1].content == "Hello there!"
    assert response.message.role == "assistant"
    assert response.message.content == "Assistant reply from fake llm."

def test_chat_service_injects_system_prompt() -> None:
    """Service should prepend the configured system prompt."""
    llm = FakeLLMClient()
    system_prompt = "This is the system prompt"
    config = ChatConfig(system_prompt=system_prompt)
    service = ChatService(llm_client=llm, config=config)
    expected_role = "system"
    
    user_prompt = "Hello there!"

    request = ChatRequest(messages=[ChatMessage(role="user", content=user_prompt)])
    response = service.generate_reply(request)

    assert llm.captured_messages[0].role == expected_role
    assert llm.captured_messages[0].content == system_prompt
    assert llm.captured_messages[-1].content == "Hello there!"
    assert llm.captured_messages[-1].role == "user"


def test_chat_service_logs_metadata(caplog: pytest.LogCaptureFixture):
    llm = FakeLLMClient()
    expected_temp = 0.3
    expected_tokens = 256
    config = ChatConfig(system_prompt="...", temperature=expected_temp, max_tokens=expected_tokens)
    service = ChatService(llm_client=llm, config=config)

    with caplog.at_level(logging.INFO):
        service.generate_reply(ChatRequest(messages=[ChatMessage(role="user", content="Test message")]))

    # Now check caplog.records for the message you're going to add later.
    assert any(getattr(record, "latency_ms", None) is not None for record in caplog.records)
    assert any(getattr(record, "temperature", None) == expected_temp for record in caplog.records)
    assert any(getattr(record, "max_tokens", None) == expected_tokens for record in caplog.records)
