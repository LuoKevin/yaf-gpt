# The Young Adult Fellowship - Generative Pre-trained Transformer (yaf-gpt)

Welcome to YAF-GPT, your all-in-one Spiritual Learning Operative Platform

## Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```

## Streamlit UI

```bash
streamlit run backend/streamlit_app.py
```

## Docker (hot-reload)

```bash
docker compose up --build
```

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
