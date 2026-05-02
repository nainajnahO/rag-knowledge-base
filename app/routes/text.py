import json
from hashlib import sha256

from fastapi import APIRouter, HTTPException

from app.chunking import chunk_text
from app.db import ConnDep
from app.ingest import (
    embed_with_error_mapping,
    find_existing_by_hash,
    insert_document_with_chunks,
)
from app.models import IngestResponse, IngestTextRequest

router = APIRouter()


@router.post("/text", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest, conn: ConnDep) -> IngestResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty after stripping whitespace")

    content_hash = sha256(text.encode("utf-8")).hexdigest()

    existing = find_existing_by_hash(conn, content_hash)
    if existing is not None:
        return existing

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="text produced no chunks")

    embeddings = embed_with_error_mapping(chunks)

    return insert_document_with_chunks(
        conn,
        title=req.title,
        author=req.author,
        published_date=req.published_date,
        metadata=json.dumps(req.metadata),
        text=text,
        content_hash=content_hash,
        chunks=chunks,
        embeddings=embeddings,
    )
