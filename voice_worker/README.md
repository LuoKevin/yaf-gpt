# Voice Worker (Scaffold)

Standalone TTS worker service intended for remote GPU deployment (for example Modal).

This scaffold provides:

- `POST /v1/voices/clone` for reference-audio cloning registration.
- `POST /v1/tts/synthesize` for speech generation.
- `GET /health` for liveness/provider checks.

By default it runs in `mock` mode and returns deterministic WAV audio so integration can be wired before real model inference.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r voice_worker/requirements.txt
uvicorn voice_worker.app.main:app --host 0.0.0.0 --port 8010 --reload
```

## Docker

```bash
docker build -f voice_worker/Dockerfile -t yaf-voice-worker .
docker run --rm -p 8010:8010 -e VOICE_WORKER_PROVIDER=mock yaf-voice-worker
```

## Provider Toggle

- `VOICE_WORKER_PROVIDER=mock` (default): in-memory cloned voice ids + generated sine-wave WAV.
- `VOICE_WORKER_PROVIDER=qwen3_stub`: same behavior, marker for future Qwen3 implementation.

## Example Requests

Clone voice:

```bash
curl -X POST http://127.0.0.1:8010/v1/voices/clone \
  -H "Content-Type: application/json" \
  -d '{"reference_audio_base64":"<data-url-or-base64-reference-audio>","voice_name":"leader"}'
```

Use a real clip (roughly 10-20 seconds recommended for cloning quality) so the request passes validation.

Synthesize:

```bash
curl -X POST http://127.0.0.1:8010/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello study group.","voice_id":"voice_123456789abc","speed":1.0}'
```

## Next Steps for Real Qwen3-TTS

1. Replace `Qwen3StubVoiceProvider` with a real inference adapter in `voice_worker/app/providers.py`.
2. Add persistent voice-profile storage (Redis or DB) instead of in-memory IDs.
3. Add streaming audio endpoint for lower playback latency.
