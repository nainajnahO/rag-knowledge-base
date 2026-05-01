from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    db.open_pool()
    try:
        yield
    finally:
        db.close_pool()


app = FastAPI(title="RAG Knowledge Base", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
