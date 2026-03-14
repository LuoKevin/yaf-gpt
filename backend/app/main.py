from fastapi import FastAPI

from .routes.bible import router as bible_router
from .routes.chat import router as chat_router
from .routes.hymn import router as hymn_router
from .routes.image import router as image_router
from .routes.study_plan import router as study_plan_router
from .routes.voice import router as voice_router

app = FastAPI(title="yaf-gpt")


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
app.include_router(hymn_router)
app.include_router(voice_router)
