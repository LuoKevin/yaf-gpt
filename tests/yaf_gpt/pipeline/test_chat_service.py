"""Tests for the chat service pipeline layer."""

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
