# RAG Knowledge Base

A small retrieval-augmented-generation (RAG) backend: upload documents (plain text or PDF), index them, search them, and chat against them with verifiable source citations.

Design decisions and the alternatives considered are documented in [`DECISIONS.md`](./DECISIONS.md).

> **Status:** scaffolding in place; feature endpoints land PR-by-PR. See the build sequence in DECISIONS.md §16.

---

## How to run

```bash
# 1. Start Postgres + pgvector (fresh-init applies the schema)
docker compose down -v && docker compose up -d

# 2. Install Python deps (uv reads pyproject.toml + .python-version)
uv sync

# 3. Create a .env file at the project root with:
#      DATABASE_URL=postgresql://ahody:ahody@localhost:5432/ahody
#      VOYAGE_API_KEY=<your key>          # https://www.voyageai.com
#      ANTHROPIC_API_KEY=<your key>       # https://console.anthropic.com (needed for /chat)

# 4. Run the API
uv run uvicorn app.main:app --reload

# 5. Sanity-check
curl http://localhost:8000/health
# → {"status":"ok"}

# 6. Ingest a document
curl -X POST http://localhost:8000/text \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Q3 Revenue Memo",
    "text": "Revenue grew 12% in Q3 to $4.2M, driven by enterprise contracts.",
    "author": "Finance Team",
    "published_date": "2025-10-15",
    "metadata": {"type": "memo", "department": "finance"}
  }'
# → {"document_id": "...", "n_chunks": 1}
```

Requires Docker, `uv`, and Python 3.14 (uv will manage Python automatically if you don't have it).

> **Dedupe behavior (`POST /text`).** If the request's `text` matches an already-ingested document (by SHA-256 content hash, applied after stripping leading/trailing whitespace), the existing `document_id` is returned and **any new `title`, `author`, `published_date`, or `metadata` in the request is ignored** — the stored row keeps its original values. This is intentional idempotent behavior (`DECISIONS.md` §12). If you need to update metadata on an existing document, that's out of scope for this endpoint.

---

## Architecture & tradeoffs

> _Filled in as PRs land._ See [`DECISIONS.md`](./DECISIONS.md) for the full record of decisions, alternatives, and migration paths.

---

## Data model

Two tables: `documents` (one row per uploaded source — title, author, published date, content hash for dedupe, plus a JSONB `metadata` column for arbitrary categorical tagging) and `chunks` (one row per retrievable text chunk, each with its 1024-dim embedding from voyage-4 and a foreign key back to its document).

The full schema lives in [`sql/schema.sql`](./sql/schema.sql) and is applied automatically by Postgres on first container init (mounted into `/docker-entrypoint-initdb.d/`). See [`DECISIONS.md`](./DECISIONS.md) §4 for the rationale and tradeoffs, and §7 for the lexical-ranking column added later in the hybrid-search step.

To reset the database during development (drops all data and re-applies the schema):

```bash
docker compose down -v && docker compose up -d
```

---

## Endpoints (status)

| Method | Path        | Status     |
|--------|-------------|------------|
| GET    | `/health`   | live       |
| POST   | `/text`     | live       |
| POST   | `/document` | PR 4       |
| GET    | `/search`   | PR 5       |
| POST   | `/chat`     | PR 6       |

> Curl examples and a Postman collection land alongside the endpoints.

---

## AI collaboration notes

> _Updated continuously as PRs land. The point: where I delegated to AI, where I corrected it, where I disagreed._

---

## What I left out and why

> See [`DECISIONS.md`](./DECISIONS.md) §15 — table of deliberate cuts with effort estimates for adding each later. Each cut is also a GitHub Issue.

---

## Known limitations

> _Filled in as the system takes shape._
