"""
Deploy a Chatterbox Turbo TTS endpoint on Modal.

This is intended to become a remote provider target for the backend
voice-generation service, not a browser-facing endpoint.

Setup:
1. Install Modal locally: `pip install modal`
2. Authenticate: `modal setup`
3. Create a volume for voice prompts:
   `modal volume create chatterbox-tts-voices`
4. Upload unzipped prompt WAV files into that volume.
5. Create a secret named `hf-token` containing `HF_TOKEN=<token>`.
6. Deploy:
   `modal deploy scripts/modal/chatterbox_tts.py`
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import modal

APP_NAME = os.getenv("MODAL_APP_NAME", "example")
VOICE_VOLUME_NAME = os.getenv("CHATTERBOX_VOICE_VOLUME", "chatterbox-tts-voices")
VOICE_PROMPTS_DIR = "/chatterbox-tts-voices"
DEFAULT_VOICE_PROMPT = os.getenv("CHATTERBOX_DEFAULT_VOICE_PROMPT", "Lucy.wav")

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "chatterbox-tts==0.1.6",
    "fastapi[standard]==0.124.4",
    "peft==0.18.0",
    "pydantic==2.11.7",
)

voice_prompts_volume = modal.Volume.from_name(VOICE_VOLUME_NAME)
app = modal.App(APP_NAME, image=image)

with image.imports():
    import torchaudio as ta
    from chatterbox.tts_turbo import ChatterboxTurboTTS


@app.cls(
    gpu="a10g",
    scaledown_window=60 * 5,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={VOICE_PROMPTS_DIR: voice_prompts_volume},
)
@modal.concurrent(max_inputs=8)
class ChatterboxTurboService:
    @modal.enter()
    def load(self) -> None:
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    def _resolve_prompt_path(self, voice_prompt_name: str) -> str:
        prompt_name = Path(voice_prompt_name).name
        prompt_path = Path(VOICE_PROMPTS_DIR) / "chatterbox-tts-voices" / "prompts" / prompt_name
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Voice prompt '{prompt_name}' was not found in the Modal volume."
            )
        return str(prompt_path)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate_endpoint(self, payload: dict):
        from fastapi import HTTPException
        from fastapi.responses import StreamingResponse

        input_text = payload.get("input_text") if isinstance(payload, dict) else None
        voice_prompt_name = payload.get("voice_prompt_name") if isinstance(payload, dict) else None

        if not isinstance(input_text, str) or not input_text.strip():
            raise HTTPException(status_code=400, detail="input_text is required.")
        if len(input_text.strip()) > 4096:
            raise HTTPException(status_code=400, detail="input_text cannot exceed 4096 characters.")

        resolved_voice_prompt = (
            voice_prompt_name.strip()
            if isinstance(voice_prompt_name, str) and voice_prompt_name.strip()
            else DEFAULT_VOICE_PROMPT
        )

        try:
            audio_bytes = self.generate.local(
                input_text=input_text.strip(),
                voice_prompt_name=resolved_voice_prompt,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - remote inference behavior
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")

    @modal.method()
    def generate(self, *, input_text: str, voice_prompt_name: str = DEFAULT_VOICE_PROMPT) -> bytes:
        prompt_path = self._resolve_prompt_path(voice_prompt_name)
        wav = self.model.generate(
            input_text,
            audio_prompt_path=prompt_path,
        )

        buffer = io.BytesIO()
        ta.save(buffer, wav, self.model.sr, format="wav")
        buffer.seek(0)
        return buffer.read()


@app.local_entrypoint()
def test(
    input_text: str = "Chatterbox Turbo running on Modal for YAF-GPT.",
    output_path: str = "tmp/yaf-gpt-chatterbox-output.wav",
    voice_prompt_name: str = DEFAULT_VOICE_PROMPT,
) -> None:
    service = ChatterboxTurboService()
    audio_bytes = service.generate.remote(
        input_text=input_text,
        voice_prompt_name=voice_prompt_name,
    )

    from pathlib import Path

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(audio_bytes)
    print(f"Saved audio to {destination}")
