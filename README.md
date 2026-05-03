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

# 6a. Ingest plain text
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

# 6b. Ingest a PDF (multipart/form-data — title required, rest optional;
#     the `metadata` slot is a JSON-encoded string per DECISIONS.md §17 / Step 4 Q1)
curl -X POST http://localhost:8000/document \
  -F "title=Ahody hiring brief" \
  -F "author=Ahody" \
  -F 'metadata={"type": "take-home"}' \
  -F "file=@Ahody hiring - work sample.pdf"
# → {"document_id": "...", "n_chunks": 2}

# 7. Search (top-k by hybrid retrieval + Voyage rerank; default k=10, max 50)
curl 'http://localhost:8000/search?q=revenue%20growth'

# 7b. Search with metadata filters (AND across keys, single-value-per-key)
curl 'http://localhost:8000/search?q=plans&author=Eng%20Leadership&meta.department=engineering&published_after=2026-01-01'
# → {"results": [{"chunk_id": "...", "ordinal": 0, "document_id": "...",
#                 "document_title": "...", "author": "...", "published_date": "...",
#                 "metadata": {...}, "score": 0.57, "text": "..."}]}
# `score` (in both /search and /chat responses) is Voyage rerank-2.5's
# `relevance_score` (§7.2). It's not calibrated across queries, so don't
# compare scores from different requests or threshold on a fixed value —
# rank order within a single response is the meaningful signal.

# 8. Chat — RAG with structured Anthropic citations
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How much did revenue grow in Q3 2025?"}'
# → {
#     "answer": "Revenue grew 12% in Q3 2025, driven by enterprise contracts.",
#     "answer_blocks": [
#       {
#         "text": "Revenue grew 12% in Q3 2025, driven by enterprise contracts.",
#         "citations": [
#           {
#             "chunk_id": "...",
#             "document_title": "Q3 Revenue Memo",
#             "published_date": "2025-10-15",
#             "cited_text": "Revenue grew 12% in Q3 2025 to $4.2M, driven by enterprise contracts."
#           }
#         ]
#       }
#     ],
#     "sources": [
#       {"chunk_id": "...", "score": 0.87, "cited": true,
#        "cited_text": ["Revenue grew 12% in Q3 2025 to $4.2M, driven by enterprise contracts."],
#        "text": "...", "document_title": "Q3 Revenue Memo", ...},
#       {"chunk_id": "...", "score": 0.31, "cited": false, "cited_text": [], ...}
#     ],
#     "stop_reason": "end_turn"
#   }
```

> **`meta.*` filter limitations.** Query-string values are always strings, so `meta.year=2025` builds the JSONB containment predicate `metadata @> '{"year": "2025"}'` and won't match a document ingested with the integer `{"year": 2025}` (JSONB containment requires exact type match). To filter on integer/boolean metadata today, ingest those values as strings. Nested keys (`meta.a.b=value`) and empty keys (`meta.=value`) are accepted as flat keys, not unpacked — `metadata @> '{"a.b": "value"}'`. The `meta.*` parameter shape is also not surfaced in `/openapi.json` (Swagger UI), since FastAPI can't infer dynamic key prefixes; this README is the canonical reference.

Requires Docker, `uv`, and Python 3.14 (uv will manage Python automatically if you don't have it).

> **Dedupe behavior (both endpoints).** Both `/text` and `/document` dedupe on the SHA-256 of the stored text (after stripping whitespace), so re-POSTing the same body to the same endpoint returns the existing `document_id` and creates no new rows. **Any new `title`, `author`, `published_date`, or `metadata` in the second request is ignored** — the stored row keeps its original values. This is intentional idempotent behavior (`DECISIONS.md` §12). If you need to update metadata on an existing document, that's out of scope for these endpoints.

> **Upload limits (`POST /document`).** Multipart body cap of 25 MB at the door (returns `413` early, before any extraction work) and a 3,000,000-character cap on the extracted text (returns `422`, same as `/text`'s text cap). The 25 MB body cap also applies to every other endpoint as a safety net but only meaningfully fires here. Full rationale + sizing math in [`DECISIONS.md §17`](./DECISIONS.md#17-upload-size-limits).

---

## Architecture & tradeoffs

The full record of decisions, alternatives considered, and migration paths is in [`DECISIONS.md`](./DECISIONS.md). High-level summary:

| Area | Choice | Why |
|---|---|---|
| **Stack** | Python 3.14 + FastAPI + Pydantic v2 + `uv`. Postgres 16 + pgvector via Docker Compose. | RAG tooling is Python-first. pgvector handles 100k+ chunks comfortably and gives one DB for documents, chunks, embeddings, metadata, *and* lexical ranking — no two-store sync problem. ([§1](./DECISIONS.md#1-stack)) |
| **Embeddings** | Voyage `voyage-4`, 1024-dim, cosine via pgvector's `<=>`. | Voyage is Anthropic's recommended embedding partner. `voyage-4` is the labelled general-purpose default — using the recommended default is a stronger story than picking the biggest tier without benchmarks. ([§2](./DECISIONS.md#2-embedding-model)) |
| **Chunking** | Recursive token-based: paragraph → sentence → word → token-offset fallback. 600 tokens / 15% overlap. | Respects natural boundaries when possible, predictable sizes always. The token-offset fallback keeps it correct in CJK / no-space scripts where Latin separators don't apply. ([§3](./DECISIONS.md#3-chunking-strategy)) |
| **Schema** | Hybrid: typed columns for `title` / `author` / `published_date` + JSONB `metadata` with GIN. | Validated SQL filters where it matters, flexibility everywhere else. One row per document, one row per chunk; `documents.content_hash UNIQUE` for dedupe. ([§4](./DECISIONS.md#4-metadata-schema)) |
| **Vector index** | HNSW, defaults `m=16, ef_construction=64`, cosine ops. | Production default in pgvector ≥0.5; best recall/speed tradeoff. IVFFlat has no advantage for new projects in 2026. ([§6](./DECISIONS.md#6-vector-index)) |
| **Hybrid search + rerank** | RRF (`k=60`, 50 candidates per lane, `FULL OUTER JOIN`, `ts_rank_cd`, `'simple'` tsv config) → Voyage `rerank-2.5` → top-K. | RRF is parameter-free and no score normalization is needed; SQL shape mirrors pgvector's and Supabase's reference examples. Reranker is the natural drop-in given Voyage is Anthropic's recommended embedding partner — same SDK, same error hierarchy. ([§7](./DECISIONS.md#7-hybrid-search-fusion) / [§7.2](./DECISIONS.md#72-reranker--voyage-rerank-25)) |
| **Citations** | Numbered inline (`[1]`, `[2]`); response includes every retrieved chunk with text + score + metadata. | The brief grades whether claims are *verifiable in seconds*. Returning all retrieved chunks (not just cited ones) lets the reviewer see what the model had access to. (Step 6 — [§8](./DECISIONS.md#8-citation-approach-the-load-bearing-one)) |
| **Chat LLM** | Claude Sonnet 4.6, single-turn. | Production default for grounded RAG. Single-turn matches the brief literally; multi-turn opens a query-rewriting subproblem deferred to a follow-up. ([§9](./DECISIONS.md#9-chat-llm) / [§10](./DECISIONS.md#10-chat-shape--single-turn)) |

The biggest *deliberate* cuts (multi-turn chat, structure-aware PDF chunking, parent-child chunking, contextual embeddings, cross-encoder re-ranking, knowledge graphs) are listed in [`DECISIONS.md §15`](./DECISIONS.md#15-deliberate-cuts-and-why) with effort estimates for adding each later.

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
| POST   | `/document` | live       |
| GET    | `/search`   | live       |
| POST   | `/chat`     | live       |

> Curl examples and a Postman collection land alongside the endpoints.

---

## AI collaboration notes

This implementation was built collaboratively with Claude Code (model: Claude Opus 4.7). The honest record below documents the moments the collaboration actually mattered — where Claude was wrong and I corrected it, where I pushed back on framing, and where my own tooling caught real bugs Claude introduced. Generic "Claude wrote the code, I reviewed it" claims aren't useful signal; specific recoveries are.

### How the design phase ran (before any code was written)

The whole project started with an extended planning Q&A session — no code, just a long structured conversation that produced [`DECISIONS.md`](./DECISIONS.md) as its artifact. For every decision area (stack, embedding model, chunking strategy, metadata schema, PDF extraction, vector index, hybrid-search fusion, citation approach, chat LLM, chat shape, error shapes, dedupe, tests, auth, deliberate cuts, build sequence) we worked through the same loop:

1. **Frame the decision.** What is actually being chosen here, and what constraints does the take-home brief impose vs. leave open?
2. **Enumerate alternatives.** Not just two, usually three to six — including the option I'd have reflexively picked, the option I'd have dismissed, and at least one option neither of us would have surfaced first. Claude pushed for breadth here; I pushed for honest comparison rather than rubber-stamping.
3. **Compare them on the dimensions that actually matter for *this* project** — not generic best-practice scoring. E.g., for the embedding model: dimensions, per-request token cap, per-input context length, multilingual quality, vendor lock-in, story-cohesion (Voyage as Anthropic's recommended embedding partner), and Matryoshka headroom for future storage-vs-recall tuning.
4. **Lock in a choice and write down the migration path** — what would it cost (in code, schema, and re-ingestion) to switch later? This is the most consistently useful artifact: every locked-in choice in `DECISIONS.md` has a "Migration notes" subsection so the next person doesn't have to re-derive the analysis.

### Where I disagreed and Claude was wrong

- **LangChain framing.** Claude initially justified the custom chunker by claiming *"LangChain's `RecursiveCharacterTextSplitter` can't do token-based splitting."* I pushed back — this was wrong. LangChain's splitter accepts a `length_function` and *can* be configured token-based. The honest tradeoff is dependency footprint vs. ~60 lines of custom code, not capability. Claude rewrote `DECISIONS.md §3` to reflect the real reasoning instead of a false constraint. ([commit `8d0791d`](https://github.com/nainajnahO/rag-knowledge-base/commit/8d0791d) / [§3](./DECISIONS.md#3-chunking-strategy))

- **"Latin separators are fine."** Claude's first chunker landed with separators `["\n\n", "\n", ". ", " "]` and called it done. I noticed it would only work on Latin-script languages — CJK, Thai, and any contiguous block would survive recursion intact and silently truncate at Voyage's 32K-token cap. Claude initially defended the design; I pressed; Claude verified, conceded, and added the `_token_slice` token-offset fallback. Empirically validated for English / Japanese / Arabic / Hebrew / Thai / mixed scripts / emoji-heavy text. ([PR #10](https://github.com/nainajnahO/rag-knowledge-base/pull/10) / [§3](./DECISIONS.md#3-chunking-strategy))

- **`@functools.cache` thread-safety claim.** In PR #9's body, Claude wrote *"`@cache` is thread-safe."* I asked it to verify against CPython source. Reading `_lru_cache_wrapper` directly: the `maxsize=None` path uses *no lock at all* — only the bounded path uses `RLock`. Both the old `global`-sentinel pattern and `@cache` share the same benign race where two simultaneously-first callers each create a client. Claude updated the PR body to remove the false claim and reframe it as "cleaner code, not a thread-safety improvement." ([PR #9](https://github.com/nainajnahO/rag-knowledge-base/pull/9))

- **"This PR doesn't hold a DB connection during the slow upstream call."** Claude's first PR #7 body claimed this as a benefit. The connection *is* held throughout the request (via `Depends(get_conn)`). I caught it; Claude removed the misleading sentence. Pool exhaustion under load is a real concern that the PR doesn't actually solve. ([PR #7](https://github.com/nainajnahO/rag-knowledge-base/pull/7))

### Where I stepped in to course-correct

- **Documentation drift after each PR.** I scheduled a project-health audit (cross-checking `DECISIONS.md` against actual code) before starting Step 4. It surfaced three minor doc/code drift items, including §11 understating the route's actual error-status taxonomy. Claude wouldn't have caught these without the explicit audit prompt — it had no incentive to re-read its own earlier writing critically. ([PR #12](https://github.com/nainajnahO/rag-knowledge-base/pull/12))

- **PR title and branch conventions.** I instructed Claude not to prefix PR titles or branches with "PR N:" — the GitHub PR number is already visible, and the prefix becomes wrong the moment any quality follow-up interleaves with feature PRs (which it did). This is now a persistent memory entry for future sessions.

- **Verifying staged set before every commit.** PyCharm has occasionally auto-staged unrelated changes. I instructed Claude to run `git status` + `git diff --cached` before *every* commit. This is also a persistent memory entry.

### Skills I authored and ran to catch what Claude missed

The following review skills are **slash-commands I (the user) wrote myself** and triggered manually at specific points in the build. They are not autonomous — Claude doesn't run them on its own initiative. They exist precisely because I needed independent checks on Claude's output:

- **`/pr-review`** — multi-pass deepening review. On `POST /text` (PR #7), pass 2 caught a latent bug: `conn.rollback()` called inside `with conn.transaction()` raises `ProgrammingError` (verified by reading psycopg source — `_rollback_gen` checks `_num_transactions > 0`). The fix was to move the try/except outside the with-block and let the context manager handle rollback. Without my running this skill, the bug would have shipped.

- **`/native-check`** — looks for hand-rolled code that duplicates language/standard-library/framework features. On the route, it caught the manual `global _client` sentinel pattern and recommended `@functools.cache` ([PR #9](https://github.com/nainajnahO/rag-knowledge-base/pull/9)). Also caught the older `Depends()`-as-default-value pattern and recommended migrating to `Annotated[]` style ([PR #11](https://github.com/nainajnahO/rag-knowledge-base/pull/11)).

- **`/dry-check`** — checks for shared types and data-driven rendering. Run on the current codebase: clean. Logged here for honesty.

- **`/architecture-check`** — audits architectural overkill. Run before Step 4 to check whether the in-progress design is right-sized for take-home scale.

These skills are themselves the most concrete signal of how I work with Claude: I don't trust a single pass of its review; I run independent checks I authored, and I act on what they find.

---

## What I left out and why

> See [`DECISIONS.md`](./DECISIONS.md) §15 — table of deliberate cuts with effort estimates for adding each later. Each cut is also a GitHub Issue.

---

## Known limitations

Honest list of what's true *right now*. See [`DECISIONS.md §15`](./DECISIONS.md#15-deliberate-cuts-and-why) for the full table of deliberate cuts with effort estimates for each.

- **DB connection is held during upstream LLM/embedding calls.** Each request borrows a pooled connection via `Depends(get_conn)` for its full lifetime. Affects every endpoint that calls an upstream: `/text` and `/document` each make 1 Voyage embed call, `/search` makes 2 (Voyage embed + Voyage rerank), and `/chat` makes 3 sequential calls (Voyage embed + Voyage rerank + Anthropic). Pool size is 1–10. Under any meaningful concurrency this would exhaust the pool, and Step 7's rerank stage stretched the worst-case time-per-held-connection longer than what [PR #7](https://github.com/nainajnahO/rag-knowledge-base/pull/7) originally flagged. Not addressed at take-home scale. The fix (release-before-upstream + reacquire-after) is a focused half-day.
- **HNSW is approximate.** Recall is typically >95%, not 100%. A query depending on a single rare chunk could miss it. Mitigation if ever needed: `SET LOCAL hnsw.ef_search = 100` per query — trades ~1ms latency for higher recall. ([§6](./DECISIONS.md#6-vector-index))
- **Lexical lane uses `'simple'` config.** Language-neutral — keeps non-English content from being mis-stemmed by an English-specific config, but skips per-language morphology lift. Tracked in [issue #3](https://github.com/nainajnahO/rag-knowledge-base/issues/3); rationale in [§7](./DECISIONS.md#7-hybrid-search-fusion) (sub-section 7.1).
- **PyMuPDF is AGPL.** PDF extraction (`POST /document`) uses `pymupdf` for plain-text quality. Strict reading of AGPL would require source disclosure for a commercial network-deployed RAG service. Migration path to `pypdf` is one module, ~30 lines, same-day swap. ([§5](./DECISIONS.md#5-pdf-extraction))
- **Single-turn `/chat`.** Multi-turn requires query rewriting (the latest user turn alone has no semantic content to retrieve on; the full conversation pollutes embeddings). Doing this badly is worse than not doing it. The brief explicitly asks for single-turn. ([§10](./DECISIONS.md#10-chat-shape--single-turn))
- **Dedupe is silent.** If you `POST /text` with content that matches an existing document by SHA-256 (after stripping whitespace), the existing `document_id` is returned and any new `title` / `author` / `published_date` / `metadata` in the request is ignored. There is no separate "update metadata" endpoint. ([§12](./DECISIONS.md#12-upload-dedupe))
