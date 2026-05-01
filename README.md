# RAG Knowledge Base

A small retrieval-augmented-generation (RAG) backend: upload documents (plain text or PDF), index them, search them, and chat against them with verifiable source citations.

Design decisions and the alternatives considered are documented in [`DECISIONS.md`](./DECISIONS.md).

> **Status:** scaffolding in place; feature endpoints land PR-by-PR. See the build sequence in DECISIONS.md §16.

---

## How to run

```bash
# 1. Start Postgres + pgvector
docker compose up -d

# 2. Install Python deps (uv reads pyproject.toml + .python-version)
uv sync

# 3. Copy env template and fill in API keys
cp .env.example .env
# edit .env: VOYAGE_API_KEY, ANTHROPIC_API_KEY

# 4. Run the API
uv run uvicorn app.main:app --reload

# 5. Sanity-check
curl http://localhost:8000/health
# → {"status":"ok"}
```

Requires Docker, `uv`, and Python 3.14 (uv will manage Python automatically if you don't have it).

---

## Architecture & tradeoffs

> _Filled in as PRs land._ See [`DECISIONS.md`](./DECISIONS.md) for the full record of decisions, alternatives, and migration paths.

---

## Data model

> _Filled in PR 2._ See [`DECISIONS.md`](./DECISIONS.md) §4 for the schema.

---

## Endpoints (status)

| Method | Path        | Status     |
|--------|-------------|------------|
| GET    | `/health`   | live       |
| POST   | `/text`     | PR 3       |
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
