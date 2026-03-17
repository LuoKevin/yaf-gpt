# The Young Adult Fellowship - Generative Pre-trained Transformer (yaf-gpt)

Welcome to YAF-GPT, your all-in-one Spiritual Learning Operative Platform

## Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```

## Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

## Streamlit UI

```bash
streamlit run backend/streamlit_app.py
```

## Docker (hot-reload)

```bash
docker compose up --build
```

## Voice Worker (Scaffold)

```bash
uvicorn voice_worker.app.main:app --host 0.0.0.0 --port 8010 --reload
```

or with Docker:

```bash
docker build -f voice_worker/Dockerfile -t yaf-voice-worker .
docker run --rm -p 8010:8010 -e VOICE_WORKER_PROVIDER=mock yaf-voice-worker
```

Reference: `voice_worker/README.md`

## New API Endpoints

- `GET /api/bible/passage`
- `POST /api/study-plan`
- `POST /api/passage-image`
- `POST /api/persona-chat`
- `POST /api/music/generate`
- `GET /api/music/jobs/{job_id}`
- `POST /api/voice/transcribe`
- Voice worker: `POST /v1/voices/clone`, `POST /v1/tts/synthesize`

## Feature Environment Toggles

- `PERSONA_MODEL` (default: `gpt-4o-mini`)
- `IMAGE_PROVIDER` (default: `openai`)
- `IMAGE_MODEL` (default: `gpt-image-1`)
- `MUSIC_PROVIDER` (default: `mock`, scaffolded `suno`)
- `SUNO_API_KEY` and `SUNO_BASE_URL` (for future real music adapter)
- `VOICE_TRANSCRIPTION_MODEL` (default: `gpt-4o-mini-transcribe`)
- `VOICE_WORKER_PROVIDER` (voice worker only; default: `mock`)

## Workflow Game (Implementation Process)

- Spec: `docs/workflow_game.md`
- Scorecard template: `docs/workflow_scorecard_template.md`
- Quest checks: `bash scripts/workflow_game_checks.sh <quest-id>`

## RAG ingest (PDFs)

1. Put PDFs in `backend/data/pdfs/`.
2. Set `OPENAI_API_KEY` in `backend/.env`.
3. Run:

```bash
python -m backend.rag.ingest \
  --input backend/data/pdfs \
  --persist backend/data/chroma \
  --collection documents
```

## Reference Materials (For RAG?)

Bible Project (very big picture summary): Book of Luke | Guide with Key Information and Resources (bibleproject.com)
https://bibleproject.com/guides/book-of-luke/

Intervarsity Sample Questions: Luke Bible Study_final (intervarsity.org)
https://intervarsity.org/sites/default/files/Luke%20and%20Acts%20Bible%20Study.pdf

Desiring God Labs (Select Luke Passages): Labs on Luke | Desiring God":
https://www.desiringgod.org/scripture/luke/labs

The Gospel Coalition Luke study: TGC Course | Knowing the Bible: Luke | 12-week Bible Study (thegospelcoalition.org):
https://www.thegospelcoalition.org/course/knowing-bible-luke/#week-1-overview
