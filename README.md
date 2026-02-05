# The Young Adult Fellowship - Generative Pre-trained Transformer (yaf-gpt)

Minimal FastAPI backend + simple RAG ingest for PDFs.

## Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

## RAG ingest (PDFs)

1. Put PDFs in `backend/data/pdfs/`.
2. Set `OPENAI_API_KEY` in `.env`.
3. Run:

```bash
python -m backend.rag.ingest \
  --input backend/data/pdfs \
  --persist backend/data/chroma \
  --collection documents
```
