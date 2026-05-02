"""Shared persistence + embedding glue for /text and /document.

Both ingestion routes share the same SHA-256 dedupe pre-check, the same Voyage
error mapping (per DECISIONS.md §11), and the same atomic document + chunks
insert with a UniqueViolation race fallback (per DECISIONS.md §12). This
module owns those pieces so the routes can stay focused on input validation
and content shape (text vs PDF).
"""

from datetime import date

import psycopg
import voyageai
from fastapi import HTTPException
from psycopg import Connection

from app.embeddings import embed_chunks
from app.models import Chunk, IngestResponse


def embed_with_error_mapping(chunks: list[Chunk]) -> list[list[float]]:
    """Embed chunks via Voyage, translating SDK errors to HTTPException per §11.

    AuthenticationError -> 500 (server misconfig — missing/invalid VOYAGE_API_KEY;
        surfaced as 500 rather than a 5xx-transient code so operators don't
        mistake it for an upstream blip and wait it out).
    RateLimitError -> 429.
    InvalidRequestError -> 400 (payload Voyage refused).
    VoyageError catch-all -> 503 (timeout / ServiceUnavailable / generic upstream;
        transient, the client may retry).
    """
    try:
        return embed_chunks(chunks)
    except voyageai.error.AuthenticationError as exc:
        raise HTTPException(status_code=500, detail=f"embedding auth failure: {exc}") from exc
    except voyageai.error.RateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"embedding rate limited: {exc}") from exc
    except voyageai.error.InvalidRequestError as exc:
        raise HTTPException(status_code=400, detail=f"embedding rejected input: {exc}") from exc
    except voyageai.error.VoyageError as exc:
        raise HTTPException(status_code=503, detail=f"embedding upstream failure: {exc}") from exc


def find_existing_by_hash(conn: Connection, content_hash: str) -> IngestResponse | None:
    """Return the existing document for `content_hash`, or None if no row matches.

    Used both for the cheap pre-check (skip Voyage entirely) and for the
    UniqueViolation race-fallback inside `insert_document_with_chunks`.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, (SELECT count(*) FROM chunks WHERE document_id = documents.id)
            FROM documents WHERE content_hash = %s
            """,
            (content_hash,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    existing_id, n_chunks = row
    return IngestResponse(document_id=existing_id, n_chunks=n_chunks)


def insert_document_with_chunks(
    conn: Connection,
    *,
    title: str,
    author: str | None,
    published_date: date | None,
    metadata: str,
    text: str,
    content_hash: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> IngestResponse:
    """Persist the document and all its chunks in one transaction.

    On UniqueViolation (a concurrent request inserted the same content_hash
    between our caller's pre-check and this insert), re-query and return the
    existing document_id — idempotent per §12. `metadata` is the JSON-encoded
    string the route will store as `jsonb`; the caller is responsible for
    JSON-encoding before passing it in.
    """
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (title, author, published_date, metadata, raw_text, content_hash)
                VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                RETURNING id
                """,
                (title, author, published_date, metadata, text, content_hash),
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
        existing = find_existing_by_hash(conn, content_hash)
        if existing is None:
            # Shouldn't happen — the only UNIQUE on this transaction is
            # documents.content_hash, so a violation implies a hash row exists.
            # Re-raise rather than silently dropping the request.
            raise
        return existing

    return IngestResponse(document_id=document_id, n_chunks=len(chunks))
