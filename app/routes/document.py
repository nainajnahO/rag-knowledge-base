import json
from hashlib import sha256
from typing import Annotated

import psycopg
import pymupdf
import voyageai
from fastapi import APIRouter, Form, HTTPException
from pydantic import TypeAdapter, ValidationError

from app.chunking import chunk_text
from app.db import ConnDep
from app.embeddings import embed_chunks
from app.extraction import extract_text
from app.limits import MaxText
from app.models import IngestDocumentRequest, IngestResponse

router = APIRouter()

_max_text_adapter: TypeAdapter[str] = TypeAdapter(MaxText)


@router.post("/document", response_model=IngestResponse)
def ingest_document(
    body: Annotated[IngestDocumentRequest, Form()],
    conn: ConnDep,
) -> IngestResponse:
    pdf_bytes = body.file.file.read()

    # Magic-byte check (DECISIONS.md §17). The Content-Type header is
    # client-controlled and untrustworthy; the %PDF- prefix is a property
    # of the file itself.
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="file is not a PDF (missing %PDF- header)")

    try:
        text = extract_text(pdf_bytes)
    except pymupdf.FileDataError as exc:
        raise HTTPException(status_code=400, detail=f"malformed PDF: {exc}") from exc

    # PyMuPDF can emit \x00 for unmapped glyphs and certain malformed text
    # streams. Postgres TEXT/JSONB reject NULs, which would otherwise surface
    # here as an opaque 500 from psycopg.
    text = text.replace("\x00", "").strip()
    if not text:
        # Image-only / scanned PDFs without an OCR'd text layer hit this.
        # Per §17 this is a 400 (mirrors /text's empty-after-strip), not 422.
        raise HTTPException(
            status_code=400,
            detail="PDF contains no extractable text (possibly an image-only or scanned document; OCR is not supported)",
        )

    # Re-validate against the 3M-char cap (DECISIONS.md §17). Reuses the
    # same MaxText alias /text uses, via TypeAdapter so we don't have to
    # construct an IngestTextRequest just to validate one field.
    try:
        _max_text_adapter.validate_python(text)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    # Parse the metadata JSON string (Q1: multipart sends metadata as JSON).
    try:
        metadata = json.loads(body.metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"metadata field is not valid JSON: {exc}",
        ) from exc
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")

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
            return IngestResponse(document_id=existing_id, n_chunks=n_chunks)

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="text produced no chunks")

    try:
        embeddings = embed_chunks(chunks)
    except voyageai.error.AuthenticationError as exc:
        raise HTTPException(status_code=500, detail=f"embedding auth failure: {exc}") from exc
    except voyageai.error.RateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"embedding rate limited: {exc}") from exc
    except voyageai.error.InvalidRequestError as exc:
        raise HTTPException(status_code=400, detail=f"embedding rejected input: {exc}") from exc
    except voyageai.error.VoyageError as exc:
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
                        body.title,
                        body.author,
                        body.published_date,
                        json.dumps(metadata),
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
        # Race: another concurrent request inserted the same content_hash
        # between our pre-check and this transaction. The transaction context
        # manager has already rolled back; re-query for the existing
        # document_id (idempotent — DECISIONS.md §12).
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, (SELECT count(*) FROM chunks WHERE document_id = documents.id)
                FROM documents WHERE content_hash = %s
                """,
                (content_hash,),
            )
            existing_id, n_chunks = cur.fetchone()
        return IngestResponse(document_id=existing_id, n_chunks=n_chunks)

    return IngestResponse(document_id=document_id, n_chunks=len(chunks))
