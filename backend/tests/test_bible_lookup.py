from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from backend.services.study_plan.bible_lookup import BibleAPIProvider, InvalidReferenceError


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
    def test_rejects_reference_without_verse(self) -> None:
        provider = BibleAPIProvider()
        with self.assertRaises(InvalidReferenceError):
            provider.get_passage("Luke 21")

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


if __name__ == "__main__":
    unittest.main()
