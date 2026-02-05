from fastapi import FastAPI

app = FastAPI(title="yaf-gpt")


@app.get("/")
def root() -> dict:
    return {"message": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
