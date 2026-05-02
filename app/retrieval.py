"""Vector retrieval against the chunks index (DECISIONS.md §3 / §6 / §8).

One function — `retrieve(conn, query, k, filters) -> list[RetrievedChunk]`.
Endpoints don't write SQL; they call this. Step 5 ships the dense-only path;
Step 7 (hybrid) replaces the SQL body without changing the signature.
"""

import json
from datetime import date

from psycopg import Connection
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from app.embeddings import embed_query_with_error_mapping
from app.models import RetrievedChunk


class Filters(BaseModel):
    """Structured filter input to `retrieve`. Built by route layer from query params."""

    author: str | None = None
    published_after: date | None = None
    published_before: date | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


# `SET LOCAL hnsw.iterative_scan = 'strict_order'` keeps HNSW walking the
# graph past its initial candidate batch until enough rows pass the WHERE
# predicate to fill LIMIT, while preserving similarity-order in results.
# Requires pgvector ≥ 0.8.0; the GUC is registered lazily by the extension,
# so it's set inside an explicit transaction.
_SQL = """
SELECT
    c.id              AS chunk_id,
    c.ordinal         AS ordinal,
    c.text            AS text,
    d.id              AS document_id,
    d.title           AS document_title,
    d.author          AS author,
    d.published_date  AS published_date,
    d.metadata        AS metadata,
    1 - (c.embedding <=> %(query)s::vector) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE  (%(author)s::text IS NULL OR d.author = %(author)s::text)
  AND  (%(after)s::date  IS NULL OR d.published_date >= %(after)s::date)
  AND  (%(before)s::date IS NULL OR d.published_date <= %(before)s::date)
  AND  (%(meta)s::jsonb  IS NULL OR d.metadata @> %(meta)s::jsonb)
ORDER BY c.embedding <=> %(query)s::vector
LIMIT %(k)s
"""


def retrieve(
    conn: Connection,
    query: str,
    k: int,
    filters: Filters,
) -> list[RetrievedChunk]:
    """Embed `query` and return the top-k matching chunks under `filters`."""
    query_vec = embed_query_with_error_mapping(query)
    meta_json = json.dumps(filters.metadata) if filters.metadata else None

    params = {
        "query": query_vec,
        "author": filters.author,
        "after": filters.published_after,
        "before": filters.published_before,
        "meta": meta_json,
        "k": k,
    }

    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
        cur.execute(_SQL, params)
        rows = cur.fetchall()

    return [RetrievedChunk(**row) for row in rows]
