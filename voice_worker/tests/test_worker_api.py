from __future__ import annotations

import base64
import unittest

from fastapi.testclient import TestClient

import voice_worker.app.main as worker_main
from voice_worker.app.providers import MockVoiceProvider


def _reference_audio_payload() -> str:
    encoded = base64.b64encode(b"\x00" * 4096).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


class VoiceWorkerAPITests(unittest.TestCase):
    def setUp(self) -> None:
        worker_main.provider = MockVoiceProvider()
        self.client = TestClient(worker_main.app)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_clone_and_synthesize(self) -> None:
        clone_response = self.client.post(
            "/v1/voices/clone",
            json={
                "reference_audio_base64": _reference_audio_payload(),
                "voice_name": "leader",
            },
        )
        self.assertEqual(clone_response.status_code, 200)
        voice_id = clone_response.json()["voice_id"]

        synth_response = self.client.post(
            "/v1/tts/synthesize",
            json={"text": "Welcome to study group.", "voice_id": voice_id},
        )
        self.assertEqual(synth_response.status_code, 200)
        body = synth_response.json()
        self.assertTrue(body["audio_base64"].startswith("data:audio/wav;base64,"))
        self.assertGreater(body["duration_seconds"], 0)

    def test_synthesize_with_unknown_voice_id_returns_400(self) -> None:
        response = self.client.post(
            "/v1/tts/synthesize",
            json={"text": "hello", "voice_id": "voice_missing"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
