import json
from hashlib import sha256

import psycopg
import voyageai
from fastapi import APIRouter, HTTPException

from app.chunking import chunk_text
from app.db import ConnDep
from app.embeddings import embed_chunks
from app.models import IngestTextRequest, IngestTextResponse

router = APIRouter()


@router.post("/text", response_model=IngestTextResponse)
def ingest_text(req: IngestTextRequest, conn: ConnDep) -> IngestTextResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty after stripping whitespace")

    content_hash = sha256(text.encode("utf-8")).hexdigest()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, (SELECT count(*) FROM chunks WHERE document_id = documents.id)
            FROM documents WHERE content_hash = %s
            """,
            (content_hash,),
        )
        row = cur.fetchone()
        if row is not None:
            existing_id, n_chunks = row
            return IngestTextResponse(document_id=existing_id, n_chunks=n_chunks)

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="text produced no chunks")

    try:
        embeddings = embed_chunks(chunks)
    except voyageai.error.AuthenticationError as exc:
        # Server misconfig (missing/invalid VOYAGE_API_KEY) — surface as 500 so
        # operators don't mistake it for a transient upstream issue.
        raise HTTPException(status_code=500, detail=f"embedding auth failure: {exc}") from exc
    except voyageai.error.RateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"embedding rate limited: {exc}") from exc
    except voyageai.error.InvalidRequestError as exc:
        raise HTTPException(status_code=400, detail=f"embedding rejected input: {exc}") from exc
    except voyageai.error.VoyageError as exc:
        # Timeout / ServiceUnavailable / generic upstream — transient, 503.
        raise HTTPException(status_code=503, detail=f"embedding upstream failure: {exc}") from exc

    try:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (title, author, published_date, metadata, raw_text, content_hash)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                    RETURNING id
                    """,
                    (
                        req.title,
                        req.author,
                        req.published_date,
                        json.dumps(req.metadata),
                        text,
                        content_hash,
                    ),
                )
                document_id = cur.fetchone()[0]
                cur.executemany(
                    "INSERT INTO chunks (document_id, ordinal, text, token_count, embedding) VALUES (%s, %s, %s, %s, %s)",
                    [
                        (document_id, c.ordinal, c.text, c.token_count, e)
                        for c, e in zip(chunks, embeddings)
                    ],
                )
    except psycopg.errors.UniqueViolation:
        # Race: another concurrent request inserted the same content_hash between
        # our pre-check and this transaction. The transaction context manager has
        # already rolled back; re-query for the existing document_id and return it
        # (idempotent — DECISIONS.md §12).
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, (SELECT count(*) FROM chunks WHERE document_id = documents.id)
                FROM documents WHERE content_hash = %s
                """,
                (content_hash,),
            )
            existing_id, n_chunks = cur.fetchone()
        return IngestTextResponse(document_id=existing_id, n_chunks=n_chunks)

    return IngestTextResponse(document_id=document_id, n_chunks=len(chunks))
