from fastapi import FastAPI

from .routes.bible import router as bible_router
from .routes.study_plan import router as study_plan_router

app = FastAPI(title="yaf-gpt")


@app.get("/")
def root() -> dict:
    return {"message": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(bible_router)
app.include_router(study_plan_router)
