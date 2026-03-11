from __future__ import annotations

import unittest

from backend.app.schemas import PersonaChatMessage, PersonaChatRequest
from backend.llm.provider import ChatResponse
from backend.services.bible_lookup import PassageData, PassageVerse
from backend.services.persona_chat_service import PersonaChatService, PersonaChatValidationError


class _FakeBibleProvider:
    def get_passage(self, reference: str, translation: str = "WEB") -> PassageData:
        return PassageData(
            reference=reference,
            normalized_reference="Luke 21:5-28",
            translation=translation,
            text="Jesus speaks about signs and endurance.",
            verses=[PassageVerse(book="Luke", chapter=21, verse=5, text="Jesus replied...")],
        )


class _FakeChatProvider:
    def __init__(self, content: str = "Focus on Christ's call to endurance.") -> None:
        self._content = content
        self.last_messages = None

    def generate(self, messages, *, model, temperature=0.2, max_tokens=None):
        self.last_messages = messages
        return ChatResponse(
            content=self._content,
            model=model,
            prompt_tokens=25,
            completion_tokens=35,
            total_tokens=60,
        )

    def stream(self, messages, *, model, temperature=0.2, max_tokens=None):
        return []


class PersonaChatServiceTests(unittest.TestCase):
    def test_generates_persona_reply_with_reference_context(self) -> None:
        chat_provider = _FakeChatProvider()
        service = PersonaChatService(
            bible_provider=_FakeBibleProvider(),
            chat_provider=chat_provider,
            model="gpt-4o-mini",
        )

        response = service.create_reply(
            PersonaChatRequest(
                messages=[PersonaChatMessage(role="user", content="What should we notice first?")],
                reference_context="Luke 21:5-28",
                translation="WEB",
            )
        )

        self.assertIn("endurance", response.reply.lower())
        self.assertEqual(response.model, "gpt-4o-mini")
        self.assertIsNotNone(chat_provider.last_messages)
        self.assertGreaterEqual(len(chat_provider.last_messages), 3)

    def test_rejects_empty_reply(self) -> None:
        service = PersonaChatService(
            bible_provider=_FakeBibleProvider(),
            chat_provider=_FakeChatProvider(content="   "),
            model="gpt-4o-mini",
        )

        with self.assertRaises(PersonaChatValidationError):
            service.create_reply(
                PersonaChatRequest(
                    messages=[PersonaChatMessage(role="user", content="hello")],
                    translation="WEB",
                )
            )


if __name__ == "__main__":
    unittest.main()
