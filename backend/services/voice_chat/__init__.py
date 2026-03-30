__all__ = [
    "BackendTTSVoiceAudioRenderer",
    "ChatProviderError",
    "ChatService",
    "ChatValidationError",
    "DisabledVoiceAudioRenderer",
    "NativeRealtimeVoiceAudioRenderer",
    "PersonaChatProviderError",
    "PersonaChatService",
    "PersonaChatValidationError",
    "VoiceChatConversationService",
    "VoiceAudioRenderResult",
    "VoiceAudioRenderingService",
    "VoiceChatService",
    "VoiceGenerationResult",
    "VoiceGenerationService",
    "VoiceTranscriptionService",
]


def __getattr__(name: str):
    if name in {
        "ChatProviderError",
        "ChatService",
        "ChatValidationError",
        "PersonaChatProviderError",
        "PersonaChatService",
        "PersonaChatValidationError",
    }:
        from .chat import ChatProviderError, ChatService, ChatValidationError
        from .persona import PersonaChatProviderError, PersonaChatService, PersonaChatValidationError

        exports = {
            "ChatProviderError": ChatProviderError,
            "ChatService": ChatService,
            "ChatValidationError": ChatValidationError,
            "PersonaChatProviderError": PersonaChatProviderError,
            "PersonaChatService": PersonaChatService,
            "PersonaChatValidationError": PersonaChatValidationError,
        }
        return exports[name]

    if name == "VoiceChatConversationService":
        from .conversation import VoiceChatConversationService

        return VoiceChatConversationService

    if name == "VoiceChatService":
        from .realtime import VoiceChatService

        return VoiceChatService

    if name in {
        "BackendTTSVoiceAudioRenderer",
        "DisabledVoiceAudioRenderer",
        "NativeRealtimeVoiceAudioRenderer",
        "VoiceAudioRenderResult",
        "VoiceAudioRenderingService",
    }:
        from .audio_rendering import (
            BackendTTSVoiceAudioRenderer,
            DisabledVoiceAudioRenderer,
            NativeRealtimeVoiceAudioRenderer,
            VoiceAudioRenderResult,
            VoiceAudioRenderingService,
        )

        exports = {
            "BackendTTSVoiceAudioRenderer": BackendTTSVoiceAudioRenderer,
            "DisabledVoiceAudioRenderer": DisabledVoiceAudioRenderer,
            "NativeRealtimeVoiceAudioRenderer": NativeRealtimeVoiceAudioRenderer,
            "VoiceAudioRenderResult": VoiceAudioRenderResult,
            "VoiceAudioRenderingService": VoiceAudioRenderingService,
        }
        return exports[name]

    if name in {"VoiceGenerationResult", "VoiceGenerationService"}:
        from .generation import VoiceGenerationResult, VoiceGenerationService

        exports = {
            "VoiceGenerationResult": VoiceGenerationResult,
            "VoiceGenerationService": VoiceGenerationService,
        }
        return exports[name]

    if name == "VoiceTranscriptionService":
        from .transcription import VoiceTranscriptionService

        return VoiceTranscriptionService

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
