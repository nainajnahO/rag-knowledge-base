import json
from hashlib import sha256

import psycopg
import voyageai
from fastapi import APIRouter, Depends, HTTPException
from pgvector.psycopg import register_vector

from app.chunking import chunk_text
from app.db import get_conn
from app.embeddings import embed_chunks
from app.models import IngestTextRequest, IngestTextResponse

router = APIRouter()


@router.post("/text", response_model=IngestTextResponse)
def ingest_text(req: IngestTextRequest, conn: psycopg.Connection = Depends(get_conn)) -> IngestTextResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty after stripping whitespace")

    content_hash = sha256(text.encode("utf-8")).hexdigest()

    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM documents WHERE content_hash = %s", (content_hash,))
        row = cur.fetchone()
        if row is not None:
            existing_id = row[0]
            cur.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (existing_id,))
            n_chunks = cur.fetchone()[0]
            return IngestTextResponse(document_id=existing_id, n_chunks=n_chunks)

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="text produced no chunks")

    try:
        embeddings = embed_chunks(chunks)
    except voyageai.error.VoyageError as exc:
        raise HTTPException(status_code=503, detail=f"embedding upstream failure: {exc}") from exc

    with conn.transaction():
        with conn.cursor() as cur:
            try:
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
            except psycopg.errors.UniqueViolation:
                # Race: another concurrent request inserted the same content_hash
                # between our pre-check and this insert. Roll back and return the
                # existing document_id (idempotent — DECISIONS.md §12).
                conn.rollback()
                with conn.cursor() as cur2:
                    cur2.execute("SELECT id FROM documents WHERE content_hash = %s", (content_hash,))
                    existing_id = cur2.fetchone()[0]
                    cur2.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (existing_id,))
                    n_chunks = cur2.fetchone()[0]
                return IngestTextResponse(document_id=existing_id, n_chunks=n_chunks)

            cur.executemany(
                "INSERT INTO chunks (document_id, ordinal, text, token_count, embedding) VALUES (%s, %s, %s, %s, %s)",
                [
                    (document_id, c.ordinal, c.text, c.token_count, e)
                    for c, e in zip(chunks, embeddings)
                ],
            )

    return IngestTextResponse(document_id=document_id, n_chunks=len(chunks))
