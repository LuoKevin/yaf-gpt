from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from backend.services.study_plan.bible_lookup import (
    BibleAPIProvider,
    InvalidReferenceError,
    PassageTooLongError,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class BibleLookupTests(unittest.TestCase):
    def test_rejects_reference_without_chapter_or_verse(self) -> None:
        provider = BibleAPIProvider()
        with self.assertRaises(InvalidReferenceError):
            provider.get_passage("Luke")

    @patch("backend.services.study_plan.bible_lookup.urlopen")
    def test_accepts_chapter_only_reference(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "reference": "Luke 21",
                "text": "Some spoke of the temple...",
                "verses": [
                    {"book_name": "Luke", "chapter": 21, "verse": 5, "text": "Some spoke of the temple..."},
                    {"book_name": "Luke", "chapter": 21, "verse": 6, "text": "Jesus said..."},
                ],
            }
        )

        provider = BibleAPIProvider()
        result = provider.get_passage("Luke 21", "WEB")
        self.assertEqual(result.normalized_reference, "Luke 21")
        self.assertEqual(len(result.verses), 2)

    @patch("backend.services.study_plan.bible_lookup.urlopen")
    def test_accepts_chapter_range_reference(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "reference": "Matthew 5-7",
                "text": "Blessed are the poor in spirit...",
                "verses": [
                    {"book_name": "Matthew", "chapter": 5, "verse": 1, "text": "Blessed are the poor in spirit..."},
                    {"book_name": "Matthew", "chapter": 7, "verse": 29, "text": "He taught as one having authority."},
                ],
            }
        )

        provider = BibleAPIProvider()
        result = provider.get_passage("Matthew 5-7", "WEB")
        self.assertEqual(result.normalized_reference, "Matthew 5-7")
        self.assertEqual(len(result.verses), 2)

    @patch("backend.services.study_plan.bible_lookup.urlopen")
    def test_parses_valid_provider_payload(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "reference": "John 3:16-17",
                "text": "For God so loved the world...",
                "verses": [
                    {"book_name": "John", "chapter": 3, "verse": 16, "text": "For God so loved..."},
                    {"book_name": "John", "chapter": 3, "verse": 17, "text": "For God did not send..."},
                ],
            }
        )

        provider = BibleAPIProvider()
        result = provider.get_passage("John 3:16-17", "WEB")
        self.assertEqual(result.normalized_reference, "John 3:16-17")
        self.assertEqual(len(result.verses), 2)
        self.assertTrue(result.text)

    @patch("backend.services.study_plan.bible_lookup.urlopen")
    def test_rejects_oversized_ranges(self, mock_urlopen) -> None:
        verses = [
            {"book_name": "Psalm", "chapter": 119, "verse": i, "text": f"Verse {i}"}
            for i in range(1, 62)
        ]
        mock_urlopen.return_value = _FakeHTTPResponse(
            {"reference": "Psalm 119:1-61", "text": "long", "verses": verses}
        )

        provider = BibleAPIProvider(max_verse_count=60)
        with self.assertRaises(InvalidReferenceError):
            provider.get_passage("Psalm 119:1-61", "WEB")

    @patch("backend.services.study_plan.bible_lookup.urlopen")
    def test_rejects_passages_over_token_limit(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "reference": "Luke 21:5-28",
                "text": " ".join(["word"] * 900),
                "verses": [
                    {"book_name": "Luke", "chapter": 21, "verse": 5, "text": " ".join(["word"] * 450)},
                    {"book_name": "Luke", "chapter": 21, "verse": 6, "text": " ".join(["word"] * 450)},
                ],
            }
        )

        provider = BibleAPIProvider(max_passage_tokens=1000)
        with self.assertRaises(PassageTooLongError) as context:
            provider.get_passage("Luke 21:5-28", "WEB")

        self.assertIn("too long for a single study request", str(context.exception))
        self.assertIn("Please choose a shorter passage", str(context.exception))


if __name__ == "__main__":
    unittest.main()
