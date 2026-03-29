"""
Minimal Modal deployment for Chatterbox speech generation.

This exposes a single POST endpoint that matches the backend's Modal voice
provider contract:
  {
    "input_text": "...",
    "voice_prompt_name": "Lucy.wav",
    "voice": "alloy",
    "instructions": null,
    "speed": 1.0
  }

This script auto-loads `scripts/modal/.env` for local deploy-time settings.

Setup:
1. `pip install modal`
2. `modal setup`
3. `modal volume create chatterbox-voices`
4. Upload one or more prompt WAV files into the volume root.
5. Create a secret named `hf-token` with `HF_TOKEN=<token>`.
6. Create a secret named `chatterbox-api-token` with
   `CHATTERBOX_API_TOKEN=<strong-random-token>`.
7. Deploy with `modal deploy scripts/modal/chatterbox_simple.py`
"""

from __future__ import annotations

import io
import os
import secrets
from pathlib import Path

import modal

ENV_PATH = Path(__file__).with_name(".env")


def _load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


_load_local_env(ENV_PATH)

APP_NAME = os.getenv("MODAL_APP_NAME", "yaf-gpt-chatterbox")
VOICE_VOLUME_NAME = os.getenv("CHATTERBOX_VOICE_VOLUME", "chatterbox-voices")
VOICE_PROMPTS_DIR = "/voices"
DEFAULT_VOICE_PROMPT = os.getenv("CHATTERBOX_DEFAULT_VOICE_PROMPT", "Lucy.wav")
HF_SECRET_NAME = os.getenv("CHATTERBOX_HF_SECRET_NAME", "hf-token")
API_SECRET_NAME = os.getenv("CHATTERBOX_API_SECRET_NAME", "chatterbox-api-token")

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "chatterbox-tts==0.1.6",
    "fastapi[standard]==0.124.4",
    "peft==0.18.0",
    "pydantic==2.11.7",
)

app = modal.App(APP_NAME, image=image)
voice_prompts_volume = modal.Volume.from_name(VOICE_VOLUME_NAME)

with image.imports():
    import torchaudio as ta
    from chatterbox.tts_turbo import ChatterboxTurboTTS


@app.cls(
    gpu="a10g",
    scaledown_window=60 * 5,
    secrets=[
        modal.Secret.from_name(HF_SECRET_NAME),
        modal.Secret.from_name(API_SECRET_NAME),
    ],
    volumes={VOICE_PROMPTS_DIR: voice_prompts_volume},
)
class ChatterboxService:
    @modal.enter()
    def load(self) -> None:
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    def _prompt_path(self, prompt_name: str) -> str:
        resolved_name = Path(prompt_name).name
        prompt_path = Path(VOICE_PROMPTS_DIR) / resolved_name
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file '{resolved_name}' was not found in the Modal volume.")
        return str(prompt_path)

    @modal.method()
    def generate(self, *, input_text: str, voice_prompt_name: str = DEFAULT_VOICE_PROMPT) -> bytes:
        prompt_path = self._prompt_path(voice_prompt_name)
        wav = self.model.generate(input_text, audio_prompt_path=prompt_path)

        buffer = io.BytesIO()
        ta.save(buffer, wav, self.model.sr, format="wav")
        buffer.seek(0)
        return buffer.read()

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate_endpoint(self, payload: dict, request):
        from fastapi import HTTPException, Request
        from fastapi.responses import StreamingResponse

        if not isinstance(request, Request):
            raise HTTPException(status_code=500, detail="Request context is unavailable.")

        self._require_bearer_auth(request)

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object body is required.")

        input_text = payload.get("input_text")
        voice_prompt_name = payload.get("voice_prompt_name")

        if not isinstance(input_text, str) or not input_text.strip():
            raise HTTPException(status_code=400, detail="input_text is required.")

        resolved_prompt = (
            voice_prompt_name.strip()
            if isinstance(voice_prompt_name, str) and voice_prompt_name.strip()
            else DEFAULT_VOICE_PROMPT
        )

        try:
            audio_bytes = self.generate.local(
                input_text=input_text.strip(),
                voice_prompt_name=resolved_prompt,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - depends on remote runtime
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")

    @staticmethod
    def _require_bearer_auth(request) -> None:
        expected_token = (os.getenv("CHATTERBOX_API_TOKEN") or "").strip()
        if not expected_token:
            raise RuntimeError("CHATTERBOX_API_TOKEN is not set in the Modal runtime.")

        authorization = request.headers.get("authorization", "").strip()
        if not authorization.lower().startswith("bearer "):
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="Missing bearer token.")

        provided_token = authorization[7:].strip()
        if not provided_token or not secrets.compare_digest(provided_token, expected_token):
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="Invalid bearer token.")


@app.local_entrypoint()
def test(
    input_text: str = "Chatterbox running on Modal.",
    output_path: str = "tmp/chatterbox-modal-output.wav",
    voice_prompt_name: str = DEFAULT_VOICE_PROMPT,
) -> None:
    service = ChatterboxService()
    audio_bytes = service.generate.remote(
        input_text=input_text,
        voice_prompt_name=voice_prompt_name,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(audio_bytes)
    print(f"Saved audio to {destination}")
