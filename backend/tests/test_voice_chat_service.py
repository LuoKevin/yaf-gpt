from __future__ import annotations

import unittest

from backend.services.bible_lookup import PassageData, PassageVerse
from backend.services.voice_chat import VoiceChatService


class _FakeRealtimeSessionAudioOutput:
    def __init__(self, voice: str) -> None:
        self.voice = voice


class _FakeRealtimeSessionAudio:
    def __init__(self, voice: str) -> None:
        self.output = _FakeRealtimeSessionAudioOutput(voice)


class _FakeRealtimeSession:
    def __init__(self, model: str, voice: str) -> None:
        self.model = model
        self.audio = _FakeRealtimeSessionAudio(voice)


class _FakeClientSecretResponse:
    def __init__(self, *, value: str, expires_at: int, model: str, voice: str) -> None:
        self.value = value
        self.expires_at = expires_at
        self.session = _FakeRealtimeSession(model, voice)


class _FakeOpenAIClient:
    class _ClientSecrets:
        def __init__(self, parent: "_FakeOpenAIClient") -> None:
            self._parent = parent

        def create(self, **kwargs):
            self._parent.last_kwargs = kwargs
            if self._parent.error_message:
                raise RuntimeError(self._parent.error_message)
            return _FakeClientSecretResponse(
                value=self._parent.client_secret,
                expires_at=self._parent.expires_at,
                model=self._parent.model_name,
                voice=self._parent.voice,
            )

    class _Realtime:
        def __init__(self, parent: "_FakeOpenAIClient") -> None:
            self.client_secrets = _FakeOpenAIClient._ClientSecrets(parent)

    def __init__(
        self,
        *,
        client_secret: str = "secret-value",
        expires_at: int = 1_700_000_000,
        model_name: str = "gpt-realtime-mini",
        voice: str = "cedar",
        error_message: str | None = None,
    ) -> None:
        self.client_secret = client_secret
        self.expires_at = expires_at
        self.model_name = model_name
        self.voice = voice
        self.error_message = error_message
        self.last_kwargs = None
        self.realtime = _FakeOpenAIClient._Realtime(self)


class _BibleProviderStub:
    def get_passage(self, reference: str, translation: str = "WEB") -> PassageData:
        return PassageData(
            reference=reference,
            normalized_reference=reference,
            translation=translation,
            text="Jesus said, Stay awake and pray.",
            verses=[PassageVerse(book="Luke", chapter=21, verse=36, text="Stay awake at all times.")],
        )


class VoiceChatServiceTests(unittest.TestCase):
    def test_creates_realtime_session_with_reference_context(self) -> None:
        client = _FakeOpenAIClient()
        service = VoiceChatService(
            client=client,
            bible_provider=_BibleProviderStub(),
            model="gpt-realtime-mini",
            transcription_model="gpt-4o-mini-transcribe",
            default_voice="cedar",
            webrtc_url="https://api.openai.com/v1/realtime/calls",
            secret_ttl_seconds=90,
        )

        session = service.create_realtime_session(
            reference_context="Luke 21:36",
            translation="WEB",
            voice="marin",
        )

        self.assertEqual(session.client_secret, "secret-value")
        self.assertEqual(session.expires_at, 1_700_000_000)
        self.assertEqual(session.model, "gpt-realtime-mini")
        self.assertEqual(session.voice, "cedar")
        self.assertEqual(session.webrtc_url, "https://api.openai.com/v1/realtime/calls")
        self.assertIsNotNone(client.last_kwargs)
        self.assertEqual(client.last_kwargs["expires_after"]["seconds"], 90)
        self.assertEqual(client.last_kwargs["session"]["audio"]["output"]["voice"], "marin")
        self.assertIn("Luke 21:36", client.last_kwargs["session"]["instructions"])

    def test_maps_provider_errors(self) -> None:
        service = VoiceChatService(
            client=_FakeOpenAIClient(error_message="provider unavailable"),
            bible_provider=_BibleProviderStub(),
            model="gpt-realtime-mini",
        )

        with self.assertRaises(RuntimeError):
            service.create_realtime_session(reference_context="Luke 21:36", translation="WEB")


if __name__ == "__main__":
    unittest.main()
