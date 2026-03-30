"""Backward-compatible aliases for older persona-chat imports."""

from .chat import (
    DEFAULT_CHAT_MODEL,
    ChatProviderError,
    ChatService,
    ChatValidationError,
)

DEFAULT_PERSONA_MODEL = DEFAULT_CHAT_MODEL
PersonaChatProviderError = ChatProviderError
PersonaChatService = ChatService
PersonaChatValidationError = ChatValidationError
