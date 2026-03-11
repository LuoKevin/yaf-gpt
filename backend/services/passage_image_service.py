from __future__ import annotations

import os

from backend.app.schemas import PassageImageRequest, PassageImageResponse
from backend.media import ImageProvider, ImageProviderError, build_image_provider_from_env

from .bible_lookup import BibleAPIProvider, BibleProvider, PassageData

DEFAULT_IMAGE_MODEL = "gpt-image-1"
DEFAULT_IMAGE_SIZE = "1024x1024"


class PassageImageGenerationError(RuntimeError):
    """Base class for passage image generation failures."""


class PassageImageProviderError(PassageImageGenerationError):
    """Raised when image provider fails."""


def _clean_excerpt(text: str, *, max_chars: int = 700) -> str:
    condensed = " ".join(text.split())
    if len(condensed) <= max_chars:
        return condensed
    return f"{condensed[: max_chars - 3].rstrip()}..."


def _build_image_prompt(
    *,
    reference: str,
    translation: str,
    style: str,
    passage_text: str,
) -> str:
    excerpt = _clean_excerpt(passage_text)
    return (
        "Create a respectful, non-idolatrous editorial illustration inspired by a Bible passage. "
        "Avoid text overlays and avoid depicting God the Father as a human figure. "
        "Use symbolic storytelling and natural lighting. "
        f"Style directive: {style}. "
        f"Passage reference: {reference} ({translation}). "
        f"Passage excerpt: {excerpt}"
    )


def _build_alt_text(*, reference: str) -> str:
    return f"Editorial illustration inspired by {reference}."


class PassageImageService:
    def __init__(
        self,
        *,
        bible_provider: BibleProvider | None = None,
        image_provider: ImageProvider | None = None,
        image_model: str | None = None,
        image_size: str | None = None,
    ) -> None:
        self._bible_provider = bible_provider or BibleAPIProvider()
        self._image_provider = image_provider or build_image_provider_from_env()
        self._image_model = image_model or os.getenv("IMAGE_MODEL") or DEFAULT_IMAGE_MODEL
        self._image_size = image_size or os.getenv("IMAGE_SIZE") or DEFAULT_IMAGE_SIZE

    def generate_passage_image(self, payload: PassageImageRequest) -> PassageImageResponse:
        passage: PassageData = self._bible_provider.get_passage(
            reference=payload.reference,
            translation=payload.translation,
        )
        prompt = _build_image_prompt(
            reference=passage.normalized_reference,
            translation=passage.translation,
            style=payload.style,
            passage_text=passage.text,
        )
        try:
            image = self._image_provider.generate(
                prompt=prompt,
                model=self._image_model,
                size=self._image_size,
            )
        except ImageProviderError as exc:
            raise PassageImageProviderError(str(exc)) from exc

        return PassageImageResponse(
            reference=payload.reference.strip(),
            translation=passage.translation,  # type: ignore[arg-type]
            style=payload.style,
            prompt_used=prompt,
            image_b64_or_url=image.image_b64_or_url,
            alt_text=_build_alt_text(reference=passage.normalized_reference),
        )
