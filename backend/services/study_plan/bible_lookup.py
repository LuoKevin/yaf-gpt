from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

MAX_VERSE_COUNT = 60
MAX_PASSAGE_TOKENS = 1000
DEFAULT_BASE_URL = "https://bible-api.com"
SUPPORTED_TRANSLATIONS = {"WEB": "web", "KJV": "kjv"}
REFERENCE_SUFFIX_PATTERN = re.compile(r"\d+(?::\d+)?(?:-\d+(?::\d+)?)?$")


class BibleLookupError(RuntimeError):
    """Base lookup error."""


class InvalidReferenceError(BibleLookupError):
    """Invalid or oversized reference input."""


class PassageTooLongError(InvalidReferenceError):
    """Passage exceeds the allowed token budget."""


class PassageNotFoundError(BibleLookupError):
    """Reference was valid, but no passage data exists."""


class PassageProviderError(BibleLookupError):
    """Lookup provider failed."""


@dataclass(frozen=True)
class PassageVerse:
    book: str
    chapter: int
    verse: int
    text: str


@dataclass(frozen=True)
class PassageData:
    reference: str
    normalized_reference: str
    translation: str
    text: str
    verses: list[PassageVerse]


class BibleProvider(Protocol):
    def get_passage(self, reference: str, translation: str = "WEB") -> PassageData:
        ...


def _normalize_reference(reference: str) -> str:
    return " ".join(reference.strip().split())


def _validate_reference_shape(reference: str) -> None:
    parts = reference.rsplit(" ", 1)
    if len(parts) != 2 or not REFERENCE_SUFFIX_PATTERN.fullmatch(parts[1]):
        raise InvalidReferenceError(
            "Reference must include a chapter or verse range, for example 'Luke 21', 'Luke 21-22', or 'Luke 21:5-28'."
        )


def _estimate_passage_tokens(text: str) -> int:
    normalized = " ".join(text.split())
    if not normalized:
        return 0

    # Conservative approximation for GPT-style tokenization on English prose.
    word_count = len(re.findall(r"\b\w+[’']?\w*\b", normalized))
    char_estimate = len(normalized) / 4
    word_estimate = word_count * 1.35
    return int(max(char_estimate, word_estimate) + 0.9999)


class BibleAPIProvider:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout_seconds: float = 10.0,
        max_verse_count: int = MAX_VERSE_COUNT,
        max_passage_tokens: int = MAX_PASSAGE_TOKENS,
    ) -> None:
        self._base_url = (base_url or os.getenv("BIBLE_API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_verse_count = max_verse_count
        self._max_passage_tokens = max_passage_tokens

    def get_passage(self, reference: str, translation: str = "WEB") -> PassageData:
        normalized_reference = _normalize_reference(reference)
        if not normalized_reference:
            raise InvalidReferenceError("Reference cannot be empty.")

        _validate_reference_shape(normalized_reference)
        translation_key = translation.upper()
        translation_id = SUPPORTED_TRANSLATIONS.get(translation_key)
        if not translation_id:
            raise InvalidReferenceError(f"Unsupported translation '{translation}'.")

        encoded_reference = quote(normalized_reference, safe=":-,")
        url = f"{self._base_url}/{encoded_reference}?translation={translation_id}"
        payload = self._fetch_json(url)

        if payload.get("error"):
            message = str(payload.get("error"))
            if "not found" in message.lower():
                raise PassageNotFoundError(message)
            raise InvalidReferenceError(message)

        raw_verses = payload.get("verses") or []
        verses = [
            PassageVerse(
                book=str(item.get("book_name") or ""),
                chapter=int(item.get("chapter")),
                verse=int(item.get("verse")),
                text=str(item.get("text") or "").strip(),
            )
            for item in raw_verses
            if item.get("chapter") is not None and item.get("verse") is not None
        ]

        if not verses:
            raise PassageNotFoundError(f"No verses found for '{normalized_reference}'.")

        if len(verses) > self._max_verse_count:
            raise InvalidReferenceError(
                f"Passage range is too large ({len(verses)} verses). "
                f"Maximum allowed is {self._max_verse_count} verses."
            )

        text = str(payload.get("text") or "").strip()
        if not text:
            text = " ".join(v.text for v in verses if v.text).strip()

        if not text:
            raise PassageNotFoundError(f"No passage text found for '{normalized_reference}'.")

        estimated_tokens = _estimate_passage_tokens(text)
        if estimated_tokens > self._max_passage_tokens:
            raise PassageTooLongError(
                "This passage is too long for a single study request. "
                "Please choose a shorter passage or split it into smaller sections."
            )

        response_reference = str(payload.get("reference") or normalized_reference)
        return PassageData(
            reference=normalized_reference,
            normalized_reference=response_reference,
            translation=translation_key,
            text=text,
            verses=verses,
        )

    def _fetch_json(self, url: str) -> dict:
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "yaf-gpt/1.0"})
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                payload = {}
            message = str(payload.get("error") or body or f"HTTP {exc.code}")
            if exc.code == 404:
                raise PassageNotFoundError(message) from exc
            if exc.code == 400:
                raise InvalidReferenceError(message) from exc
            raise PassageProviderError(message) from exc
        except URLError as exc:
            raise PassageProviderError(str(exc.reason)) from exc
        except json.JSONDecodeError as exc:
            raise PassageProviderError("Invalid JSON response from Bible provider.") from exc
