from __future__ import annotations

import unittest

from backend.app.schemas import PassageImageRequest
from backend.media import ImageGenerationResult
from backend.services.study_plan.bible_lookup import PassageData, PassageVerse
from backend.services.study_plan.passage_image_service import PassageImageService


class _FakeBibleProvider:
    def get_passage(self, reference: str, translation: str = "WEB") -> PassageData:
        return PassageData(
            reference=reference,
            normalized_reference="Luke 21:5-28",
            translation=translation,
            text="Some of his disciples were remarking about how the temple was adorned.",
            verses=[PassageVerse(book="Luke", chapter=21, verse=5, text="Some of his disciples...")],
        )


class _FakeImageProvider:
    def __init__(self) -> None:
        self.last_prompt = None

    def generate(self, *, prompt: str, model: str, size: str = "1024x1024") -> ImageGenerationResult:
        self.last_prompt = prompt
        return ImageGenerationResult(image_b64_or_url="https://example.com/generated.png")


class PassageImageServiceTests(unittest.TestCase):
    def test_generates_passage_image_payload(self) -> None:
        image_provider = _FakeImageProvider()
        service = PassageImageService(
            bible_provider=_FakeBibleProvider(),
            image_provider=image_provider,
            image_model="gpt-image-1",
        )

        result = service.generate_passage_image(
            PassageImageRequest(
                reference="Luke 21:5-28",
                translation="WEB",
                style="modern_editorial_illustration",
            )
        )

        self.assertEqual(result.translation, "WEB")
        self.assertTrue(result.image_b64_or_url)
        self.assertIn("Luke 21:5-28", result.prompt_used)
        self.assertIsNotNone(image_provider.last_prompt)


if __name__ == "__main__":
    unittest.main()
