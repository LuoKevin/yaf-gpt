import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.bible import router as bible_router
from .routes.chat import router as chat_router
from .routes.image import router as image_router
from .routes.music import router as music_router
from .routes.study_plan import router as study_plan_router
from .routes.voice import router as voice_router

app = FastAPI(title="yaf-gpt")


def _build_allowed_origins() -> list[str]:
    # Keep local dev working by default, and layer production origins via env.
    allowed = {
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    }
    configured = os.getenv("CORS_ORIGINS", "")
    for origin in configured.split(","):
        cleaned = origin.strip()
        if cleaned:
            allowed.add(cleaned)
    return sorted(allowed)


app.add_middleware(
    CORSMiddleware,
    allow_origins=_build_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {"message": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(bible_router)
app.include_router(study_plan_router)
app.include_router(image_router)
app.include_router(chat_router)
app.include_router(music_router)
app.include_router(voice_router)
