-- Database schema for the RAG knowledge base.
-- Applied automatically on first container init via docker-compose.yml's
-- /docker-entrypoint-initdb.d mount. Reset with `docker compose down -v && docker compose up -d`.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT        NOT NULL,
    author          TEXT,
    published_date  DATE,
    metadata        JSONB       NOT NULL DEFAULT '{}',
    raw_text        TEXT        NOT NULL,
    content_hash    TEXT        NOT NULL UNIQUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_documents_published_date ON documents (published_date);
CREATE INDEX idx_documents_metadata       ON documents USING GIN (metadata);

CREATE TABLE chunks (
    id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID         NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INT          NOT NULL,
    text         TEXT         NOT NULL,
    token_count  INT          NOT NULL,
    embedding    vector(1024) NOT NULL,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
