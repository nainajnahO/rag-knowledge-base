# Design Decisions

This document captures the design decisions made before implementation, the alternatives considered for each, and why we chose what we chose. Where a decision deliberately keeps a future upgrade path open, the migration notes are recorded so the next person (or future me) can follow it without re-deriving the analysis.

This is a working artifact — written for the reviewer and for ourselves.

---

## 1. Stack

**Decision:** Python 3.14 + FastAPI + Pydantic v2 + `uv` for env/deps. Postgres 16 + pgvector via Docker Compose (`pgvector/pgvector:pg16`).

**Alternatives considered:**
- Node/TypeScript + Postgres+pgvector
- Python/FastAPI + dedicated vector DB (Qdrant, Weaviate)
- Python 3.12 instead of 3.14

**Why this choice:**
- PDF text extraction, embedding/LLM SDKs, tokenizers, and RAG tooling are all Python-first ecosystems. Node would mean fighting the ecosystem on a take-home.
- pgvector handles 100k+ chunks comfortably and gives us *one* database for documents, chunks, embeddings, metadata, *and* BM25 (via tsvector). A dedicated vector DB would force two stores in sync for no benefit at this scale.
- Python 3.14 because "default to the latest stable" is the rule. All deps in our list (`fastapi`, `pydantic`, `psycopg`, `voyageai`, `anthropic`, `pymupdf`) ship 3.14 wheels.
- Docker Compose because it's reproducible by the reviewer with one command and avoids hosted-service signup friction. Free-tier hosted Postgres (Supabase/Neon) was rejected: pausing free tiers, shared credentials, and using ~5% of a managed BaaS would be exactly the unnecessary abstraction the brief warns against.

**Migration notes:** None — this is a load-bearing decision. Don't change the language or vector store later.

---

## 2. Embedding model

**Decision:** Voyage `voyage-3`, 1024 dimensions, cosine distance.

**Alternatives considered:**
- `voyage-3-lite` (512 dim, ~3x cheaper, lower quality)
- `voyage-3-large` (1024 dim, ~3x more expensive, marginal quality gain on general English)
- `voyage-context-3` (1024 dim, contextual embeddings — each chunk's embedding incorporates surrounding-document context)
- OpenAI `text-embedding-3-small` (1536 dim)
- Local sentence-transformers model

**Why this choice:**
- `voyage-3` is the defensible mid-tier default — quality/cost balance with no over- or under-shooting.
- Voyage is Anthropic's recommended embedding partner; clean README narrative ("used Anthropic's stack end-to-end").
- 1024 dimensions is a clean column type for pgvector with no storage bloat.
- Cosine because Voyage's docs explicitly recommend it (their embeddings aren't unit-normalized).

**Migration notes — voyage-3 → voyage-context-3 later (cheap):**
- Same dimension (1024) → **no schema migration**.
- Same distance metric → no retrieval code change.
- Embedder module changes (~30 lines): the API takes document-grouped chunk lists, not an arbitrary list of strings.
- Ingestion pipeline must batch chunks **by document** (which is the natural flow anyway — one doc per upload).
- Re-embed all existing chunks (cents at this scale).
- Total effort: roughly half a day of focused work.
- **Design implication for now:** keep the embedder module behind a single function signature; don't interleave chunks from different documents in batched embedding calls.

---

## 3. Chunking strategy

**Decision:** Recursive token-based splitting (paragraph → sentence → word fallback) for both `/text` and `/document` endpoints. **600 tokens per chunk, 15% overlap (~90 tokens).** Token counting via Voyage's tokenizer, not characters.

**Alternatives considered:**
- Fixed character/token splits (rejected — cuts mid-word/sentence)
- Sentence-grouping with `nltk`/`pysbd` (variable chunk sizes; long sentences blow the budget)
- Layout/structure-aware via `docling` (per-section/per-page metadata for PDFs — strong story but +half-day to the build)
- Semantic chunking (extra embedding calls during ingestion; modest gains over recursive in recent benchmarks)
- Hierarchical / parent-child (small chunks for retrieval, larger parents for LLM context — better quality, more complexity)

**Why these specific numbers:**
- Recursive splitting respects natural boundaries when possible while keeping chunk sizes predictable; LangChain-style splitter is well-understood, ~30 lines or one library call.
- 600 tokens because Voyage embeddings are trained for general text in this range; memos/reports rarely have meaningful units shorter than ~150 tokens (paragraph) or longer than ~700 (a few paragraphs). 600 is the middle, keeping citations precise enough to spot-check.
- 15% overlap because boundary-straddling sentences would otherwise be split across chunks and missed by retrieval; 0% loses content, 25%+ inflates results with near-duplicates.

**Migration notes — recursive flat → parent-child (1 focused day if architecture is clean):**
- Schema: add `parent_id` column on `chunks` referencing another row (or a separate `parent_chunks` table).
- Chunker module: emit two-level output (parents + children with linkage).
- Retrieval: search hits children, JOIN to parents, DISTINCT to avoid duplicate parent text in LLM context.
- Citations: cite the *child* (precise quote), pass the *parent* (rich context) to the LLM. Prompt + response shape gain one field.
- Re-ingest all documents (cents at this scale).
- **Design implication for now (already baked into the plan):**
  1. Chunker is its own module with signature `chunk(text, metadata) -> list[Chunk]`. Endpoints don't know how splitting works.
  2. Retrieval is its own module with signature `retrieve(query, k, filters) -> list[RetrievedChunk]`. Endpoints don't write SQL.
  3. `Chunk` is a Pydantic model. Adding `parent_id: UUID | None = None` later is one line.

**Migration notes — recursive → structure-aware PDF via `docling` (~half-day):**
- Replace pymupdf extraction with docling for `POST /document` only.
- Docling returns Markdown with hierarchy preserved → richer per-chunk metadata (section, page).
- Chunker signature stays the same.
- README will mention this as the natural next move.

---

## 4. Metadata schema

**Decision:** Hybrid — typed columns for well-known fields + JSONB for the long tail.

**Required (typed) on upload:** `title`, `source_type`.
**Optional (typed):** `author`, `published_date`.
**Catch-all:** `metadata jsonb` with a GIN index for arbitrary extras.

**Alternatives considered:**
- Pure JSONB (one `metadata` column, anything goes — flexible but type-blind; weak data-modeling story)
- Pure typed columns (rigid; bad fit for a generic knowledge base where customers define what they send)
- Denormalizing filter fields onto the `chunks` table for query speed (premature optimization at this scale)

**Why this choice:**
- Pure JSONB fails on date filtering — there's no validation that `published_date` is actually a date, and range queries silently break.
- Pure typed columns are a migration every time a customer wants a new field.
- Hybrid gives validated SQL filters where it matters and flexibility everywhere else. Standard answer for a flexible-but-principled KB schema.
- Filtering happens via JOIN to documents at query time — single source of truth, no denormalization, Postgres handles it fine at this scale.

**Schema sketch:**

```sql
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    source_type     TEXT NOT NULL,            -- 'memo' | 'report' | 'article' | 'text'
    author          TEXT,
    published_date  DATE,
    metadata        JSONB NOT NULL DEFAULT '{}',
    raw_text        TEXT NOT NULL,
    content_hash    TEXT NOT NULL UNIQUE,     -- SHA-256 for dedupe
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_documents_source_type    ON documents (source_type);
CREATE INDEX idx_documents_published_date ON documents (published_date);
CREATE INDEX idx_documents_metadata       ON documents USING GIN (metadata);

CREATE TABLE chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INT NOT NULL,
    text         TEXT NOT NULL,
    token_count  INT NOT NULL,
    embedding    vector(1024) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (document_id, ordinal)
);
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_tsv       ON chunks USING GIN (tsv);
```

**Filter API on `/search`:**
- Typed query params: `source_type`, `author`, `published_after`, `published_before`
- JSONB equality: `meta.<key>=<value>` (e.g., `meta.department=research`)
- No DSL, no `$and`/`$or`/`$gt` operators — the brief is about RAG, not query languages.

---

## 5. PDF extraction

**Decision:** `pymupdf` (fitz).

**Alternatives considered:**
- `pypdf` (BSD-style, pure Python, lower text quality)
- `pdfplumber` (table-aware, slower, prose quality similar to pypdf)
- `docling` (structure-aware, ML models, hundreds of MB — overkill for the plain-text-only path we picked)
- `unstructured` (older structure-aware option; superseded by docling)

**Why this choice:**
- Best plain-text extraction quality of the lightweight options, fastest by a margin.
- We're not using structure (deferred to README's "next steps") so a heavier structure-aware library would be paying for capability we don't use.

**Trade-off — AGPL license:**
PyMuPDF is dual-licensed: AGPL by default, with a commercial license available from Artifex. Strict reading of AGPL would require source disclosure for a commercial network-deployed RAG service. **README will explicitly call this out** along with the path to swap for pypdf — extraction is one module, ~30 lines, same-day swap.

**Migration notes — pymupdf → docling (for structure-aware chunking, ~half-day):**
- Replace extraction module body.
- Chunker signature stays the same.
- Re-ingest PDFs to capture section/page metadata.
- Adds richer chunk-level metadata (section title, page number) → stronger citations.

---

## 6. Vector index

**Decision:** HNSW with default parameters: `m=16, ef_construction=64`. Cosine ops (`vector_cosine_ops`).

**Alternatives considered:**
- No index (sequential scan — fast at take-home scale, weaker story)
- IVFFlat (cluster-based, lower recall, requires retuning `lists` as data grows; legacy in 2026)

**Why this choice:**
- HNSW is the production default in pgvector 0.5+. Best recall/speed tradeoff.
- IVFFlat has no advantage for new projects.
- "No index works at this scale" is true but tells a worse story than "I picked the right index and used the documented defaults."

**One subtle thing worth knowing:** HNSW gives approximate nearest neighbor (recall typically >95%, not 100%). If a niche query depends on a single rare chunk, that miss rate matters. Mitigation if needed: `SET LOCAL hnsw.ef_search = 100` per query — trades a bit of latency for recall.

---

## 7. Hybrid search fusion

**Decision:** Reciprocal Rank Fusion (RRF), constant `k=60`.

**Alternatives considered:**
- Linear weighted combination `α · vector + (1−α) · bm25` (requires score normalization and tuned α)
- Convex combination with min-max or z-score normalization (same problem, more steps)
- CombSUM / CombMNZ (older, superseded by RRF)
- Skipping hybrid entirely (vector-only)

**Why this choice:**
- Parameter-free — no tuning data needed, no α to defend.
- No score normalization needed (BM25 is unbounded, cosine is `[-1, 1]`, mixing them naively is broken).
- Industry standard in 2026 (Elastic, Vespa, Weaviate, Qdrant all default to RRF).
- One SQL query with two CTEs and a JOIN — ~20 lines, no Python-side merging.

**Implementation sketch:**

```sql
WITH vec AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
  FROM chunks ORDER BY embedding <=> $1 LIMIT 50
),
bm25 AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, q) DESC) AS rank
  FROM chunks, plainto_tsquery('english', $2) q
  WHERE tsv @@ q LIMIT 50
)
SELECT c.id, c.text, c.document_id,
       COALESCE(1.0/(60+vec.rank), 0) + COALESCE(1.0/(60+bm25.rank), 0) AS rrf_score
FROM chunks c
LEFT JOIN vec  ON c.id = vec.id
LEFT JOIN bm25 ON c.id = bm25.id
WHERE vec.id IS NOT NULL OR bm25.id IS NOT NULL
ORDER BY rrf_score DESC
LIMIT 10;
```

**Trade-off — language config:** `to_tsvector('english', ...)` does English stemming + stopword removal. For multilingual or Swedish corpora, `'english'` is wrong. README will note: switch to per-language configs or `'simple'` (no stemming) as a language-agnostic fallback.

---

## 8. Citation approach (the load-bearing one)

**Decision:** Numbered inline citations (`[1]`, `[2]`) with the LLM's prompt; the response includes **all** retrieved chunks with full text, score, and document metadata so the reviewer can verify any claim in seconds.

**Alternatives considered:**
- Structured JSON output with per-claim `supported_by` arrays (more machine-checkable, more complex prompt + parsing, same underlying hallucination risk)
- Span-level citation (model returns the exact substring supporting each claim — brittle, models hallucinate spans)
- Post-hoc verification pass (second LLM call to verify each claim — 2x cost and latency, overkill)

**Why this choice:**
- The brief grades "is it actually possible to verify what the LLM says?" — a numbered citation that maps to a chunk in the response makes verification a 10-second action per claim. That *is* the goal.
- Returning all retrieved chunks (not just cited ones) lets the reviewer see what the model had access to and judge if it cited the right ones — much stronger than "trust me, here's what I cited."

**Anti-hallucination guardrails (all four):**

1. **System prompt forbids external knowledge.** *"Answer using only the provided sources. If they don't contain enough information to answer, say so. Do not use prior knowledge."* The single most important line.
2. **Refusal allowed and encouraged.** *"If the sources do not contain enough information, say 'I don't have enough information in the provided sources to answer this.'"* Without this, the model stretches to manufacture answers from weak chunks.
3. **Score threshold gate.** If the top retrieved chunk's similarity is below 0.5 (cosine), don't call the LLM at all — return `{"answer": "No relevant content found in the knowledge base.", "sources": []}`.
4. **Pass chunk metadata, not just text.** Each chunk in the prompt is preceded by `Title: ... (source_type, published_date)` so the model can contextualize and source-confuse less.

**Top-K retrieval for chat:** 8 chunks above threshold 0.5. Sweet spot — enough context for nuanced questions without dilution from low-relevance chunks.

**Response shape:**

```json
{
  "answer": "Per the Q3 memo [1], revenue grew 12% to $4.2M, driven by enterprise contracts [2].",
  "sources": [
    {
      "citation": 1,
      "chunk_id": "uuid",
      "document_id": "uuid",
      "document_title": "Q3 Revenue Memo",
      "source_type": "memo",
      "published_date": "2025-10-15",
      "score": 0.87,
      "text": "Revenue grew 12% in Q3 to $4.2M..."
    }
  ]
}
```

---

## 9. Chat LLM

**Decision:** Claude Sonnet 4.6.

**Alternatives considered:**
- Claude Haiku 4.5 (~4x cheaper, sufficient for grounded extractive RAG, slightly more risk on multi-source synthesis)
- Claude Opus 4.7 (overkill for grounded synthesis; hard to defend on cost)

**Why this choice:**
- Sonnet is the production default for RAG — best balance of synthesis quality, refusal behavior, and citation discipline.
- Cost difference vs Haiku is irrelevant at demo scale (~2¢ vs ~0.7¢ per call).
- Haiku is the sharper "I thought about cost" answer, but only if defended with benchmarks. Without benchmark data, Sonnet is the unimpeachable choice.

**Migration notes:** Model name lives in one config constant / env var. Swapping is one line.

---

## 10. /chat shape — single-turn

**Decision:** Single-turn / stateless. Request: `{question}`. Response: `{answer, sources}`.

**Alternatives considered:**
- Multi-turn with client-managed message history (adds query-rewriting complexity)
- Multi-turn with server-managed sessions (adds session storage, state management; overkill)

**Why this choice:**
- The brief literally says *"POST /chat takes a question."* Single-turn matches the spec.
- Multi-turn opens a real RAG design problem — query rewriting for follow-up retrieval (the latest user turn alone has no semantic content to retrieve on; the full conversation pollutes embeddings; the right move is an LLM call to rewrite the latest turn into a standalone query before embedding). Doing this badly is worse than not doing it.

**README will document the multi-turn upgrade path explicitly:**
> Multi-turn /chat is a natural next step. The non-trivial part isn't the API shape — it's query rewriting for retrieval. The intended approach: client passes the conversation as `messages`, the server uses the LLM to rewrite the latest turn into a standalone search query (informed by history), then retrieves on the rewritten query, then answers with the original conversation in the LLM's context. Defers cleanly into a follow-up PR.

---

## 11. API & error shapes

**Decision:** Plain success bodies, FastAPI default error shape (`{"detail": "..."}`).

**Alternatives considered:**
- Wrapped envelope (`{data: ..., error: ...}` or similar)

**Why plain:**
- It's what FastAPI does by default — wrapping is extra code that adds noise to curl examples.
- The brief explicitly says they're not evaluating perfect error handling, and adding a custom error framework would be the unnecessary abstraction the brief warns against.

**Status codes used explicitly:** 422 (Pydantic validation, free), 400 (bad file/input), 401 (missing/bad API key), 503 (Voyage/Anthropic upstream failure). No retries, no circuit breakers, no custom exception hierarchy.

---

## 12. Upload dedupe

**Decision:** SHA-256 hash of raw content stored on `documents.content_hash` (UNIQUE). On collision, return the existing `document_id` instead of creating a duplicate.

**Alternatives considered:**
- No dedupe (simpler, but reviewer-uploads-twice produces noisy duplicate search results)

**Why hash dedupe:**
- ~10 lines of code.
- Real "thought about this" signal for the reviewer.
- Saves embedding cost on accidental re-uploads.
- Doesn't preclude versioning later — the hash is alongside, not as, the primary key.

---

## 13. Tests

**Decision:** Three pytest smoke tests. ~80 lines total.

1. **Chunker sanity** — feed a sample paragraph, assert chunks have expected size/overlap properties.
2. **Upload → search smoke** — POST /text, then GET /search, assert at least one result and the top result is from the uploaded doc.
3. **Citation-source consistency** — POST /chat, assert every `[N]` in the answer maps to a `citation: N` in `sources`.

**Why this and not more:**
- The brief explicitly says they're not evaluating production maturity (CI/CD, full test coverage).
- But code quality is a grading criterion, and zero tests is a weaker signal than a few well-chosen ones.
- Three tests demonstrate the discipline without pretending to be a full suite.

---

## 14. API key auth (stretch goal)

**Decision:** Single API key, FastAPI dependency injection.

**Why:**
- Brief lists it as stretch and says even a simple API key is enough.
- ~5 lines: a dependency that reads `Authorization: Bearer <key>` and compares against env var `API_KEY`.
- Cheap signal of production-awareness. Deserves the half-PR it costs.

---

## 15. Deliberate cuts (and why)

Each of these will become a GitHub Issue and an entry in the README's "next steps" section. The point is to make the cuts *legible* — not hidden by omission.

| Cut | Why deferred | Effort to add later |
|---|---|---|
| **Multi-turn /chat with query rewriting** | Real complexity is in the query rewriter, not the API. Brief asks for single-turn. | ~1 day |
| **Structure-aware PDF chunking via docling** | Chose plain-text recursive chunking as the baseline. docling adds ML model deps. | ~half-day |
| **Parent-child / hierarchical chunking** | Real retrieval-quality win, but complicates retrieval and chat. Architecture is designed to make it cheap. | ~1 day |
| **voyage-context-3 contextual embeddings** | Same dimension as voyage-3 → cheap migration. Better story to ship voyage-3 first. | ~half-day |
| **Streaming responses on /chat** | Brief lists it as stretch. Demo is via curl; streaming complicates citation parsing for the reviewer. | ~few hours |
| **Re-ranking with a cross-encoder** | Real quality improvement but adds a dependency and another model API. RRF gets us most of the way there. | ~half-day |
| **Knowledge graph (entities + relations)** | Genuinely interesting, big rabbit hole. Not the foundation the brief asks for. | ~2-3 days |
| **Production logging/metrics/CI** | Brief explicitly excludes these. | n/a |

---

## 16. Build sequence

PRs are sized to be individually reviewable. Each merges to `main` with a merge commit (not squash) so individual commits survive.

1. **PR 1** — Scaffold: `.gitignore`, `pyproject.toml` (uv, Python 3.14), `docker-compose.yml`, `app/main.py` with `/health`, `.env.example`, README skeleton.
2. **PR 2** — Schema: `init.sql` with documents/chunks tables, pgvector extension, HNSW + GIN indexes.
3. **PR 3** — `POST /text` end-to-end: chunker module, Voyage embedder module, persistence.
4. **PR 4** — `POST /document`: pymupdf extraction + content-hash dedupe.
5. **PR 5** — `GET /search`: vector similarity with metadata filters via JOIN.
6. **PR 6** — `POST /chat`: retrieval + Sonnet 4.6 + numbered citations + four guardrails.
7. **PR 7** — Hybrid search: tsvector column + RRF query.
8. **PR 8** — API key auth dependency.
9. **PR 9** — Smoke tests (pytest).
10. **PR 10** — README polish, sample documents demonstrating metadata filtering, curl/Postman collection, opening "deliberate cuts" issues.

If time gets tight, drop PR 7 and PR 8 — they become Issues with explanations, which is also signal.
