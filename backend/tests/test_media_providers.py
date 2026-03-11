from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.media import MockMusicProvider


class MockMusicProviderTests(unittest.TestCase):
    def test_mock_job_transitions_from_queued_to_completed(self) -> None:
        provider = MockMusicProvider()

        with patch("backend.media.providers.time.monotonic", side_effect=[100.0, 100.4, 102.0, 104.0]):
            created = provider.create_job(
                title="Hope Song",
                lyrics="Sample lyrics",
                style_hint="acoustic",
            )
            self.assertEqual(created.status, "queued")

            queued = provider.get_job(created.job_id)
            self.assertIsNotNone(queued)
            self.assertEqual(queued.status, "queued")

            in_progress = provider.get_job(created.job_id)
            self.assertIsNotNone(in_progress)
            self.assertEqual(in_progress.status, "in_progress")

            completed = provider.get_job(created.job_id)
            self.assertIsNotNone(completed)
            self.assertEqual(completed.status, "completed")
            self.assertTrue(completed.audio_url)


if __name__ == "__main__":
    unittest.main()
