# RAG Knowledge Base

A retrieval-augmented-generation backend: ingest text and PDFs, search them with hybrid retrieval + reranking, and chat with verifiable citations.

Design rationale, alternatives considered, and migration paths are in [`DECISIONS.md`](./DECISIONS.md).

## Contents

- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Architecture](#architecture)
- [Data model](#data-model)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [AI collaboration notes](#ai-collaboration-notes)

## Quick start

Requires Docker, [`uv`](https://docs.astral.sh/uv/), and Python 3.14 (uv installs it for you).

```bash
# Start Postgres + pgvector (schema applies on fresh init)
docker compose up -d

# Install Python deps
uv sync

# Create .env at the project root (see .env.example)
#   DATABASE_URL=postgresql://ahody:ahody@localhost:5432/ahody
#   VOYAGE_API_KEY=<your key>      # https://www.voyageai.com
#   ANTHROPIC_API_KEY=<your key>   # https://console.anthropic.com

# Run the API
uv run uvicorn app.main:app --reload

# Sanity-check
curl http://localhost:8000/health   # → {"status":"ok"}
```

To reset the database during development (drops all data, re-applies the schema):

```bash
docker compose down -v && docker compose up -d
```

## API reference

| Method | Path        | Purpose                                                              |
|--------|-------------|----------------------------------------------------------------------|
| GET    | `/health`   | Liveness check                                                       |
| POST   | `/text`     | Ingest plain text (extracts knowledge graph)                         |
| POST   | `/document` | Ingest a PDF (multipart upload; extracts knowledge graph)            |
| GET    | `/search`   | Hybrid search + Voyage rerank, with optional entity-aware filtering  |
| POST   | `/chat`     | RAG chat with structured Anthropic citations + auto entity-awareness |

<details>
<summary><code>POST /text</code> — ingest plain text</summary>

```bash
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

**Dedupe is silent.** The endpoint dedupes on the SHA-256 of the stored text (after stripping whitespace). Re-POSTing the same body returns the existing `document_id` and creates no new rows; any new `title` / `author` / `published_date` / `metadata` is ignored. There is no metadata-update endpoint. ([§12](./DECISIONS.md#12-upload-dedupe))

</details>

<details>
<summary><code>POST /document</code> — ingest a PDF</summary>

```bash
# multipart/form-data — title required, rest optional;
# metadata is a JSON-encoded string
curl -X POST http://localhost:8000/document \
  -F "title=Ahody hiring brief" \
  -F "author=Ahody" \
  -F 'metadata={"type": "take-home"}' \
  -F "file=@Ahody hiring - work sample.pdf"
# → {"document_id": "...", "n_chunks": 2}
```

**Limits.** 25 MB multipart body cap (returns `413` before extraction); 3,000,000-character cap on extracted text (returns `422`). Same SHA-256 dedupe as `/text`. ([§17](./DECISIONS.md#17-upload-size-limits))

</details>

<details>
<summary><code>GET /search</code> — hybrid search + rerank</summary>

```bash
# top-k by hybrid retrieval + Voyage rerank; default k=10, max 50
curl 'http://localhost:8000/search?q=revenue%20growth'

# with metadata filters (AND across keys, single value per key)
curl 'http://localhost:8000/search?q=plans&author=Eng%20Leadership&meta.department=engineering&published_after=2026-01-01'
# → {"results": [{"chunk_id": "...", "ordinal": 0, "document_id": "...",
#                 "document_title": "...", "author": "...", "published_date": "...",
#                 "metadata": {...}, "score": 0.57, "text": "..."}]}
```

**`score` is uncalibrated.** It's Voyage rerank-2.5's `relevance_score` — meaningful as rank order *within* a single response, not as a threshold or cross-query comparison. ([§7.2](./DECISIONS.md#72-reranker--voyage-rerank-25))

**`meta.*` filter caveats.** Query-string values are always strings, so `meta.year=2025` builds `metadata @> '{"year": "2025"}'` and won't match an integer-typed `{"year": 2025}` (JSONB containment requires exact type match). To filter on int/bool metadata today, ingest those values as strings. Nested keys (`meta.a.b=v`) and empty keys (`meta.=v`) are treated as flat keys, not unpacked. Not surfaced in `/openapi.json` since FastAPI can't infer dynamic key prefixes — this README is the canonical reference.

**Entity-aware filtering** (knowledge graph). Repeat `?entity=` to require chunk-level co-mention of every named entity:

```bash
# "All chunks where Acme Corporation AND Berlin are both mentioned"
curl 'http://localhost:8000/search?q=expansion&entity=Acme%20Corporation&entity=Berlin'
```

Names match aliases case-insensitively (the resolution stage stores `Acme`, `Acme Corp`, `Acme Inc.` under one canonical entity). An unknown name yields zero results — under chunk-level AND, no chunk can mention an entity that isn't in the graph. ([§18](./DECISIONS.md#18-knowledge-graph-closes-30))

</details>

<details>
<summary><code>POST /chat</code> — RAG with citations</summary>

```bash
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

`sources` includes every retrieved chunk (not just cited ones), each with a `cited` flag and any verbatim `cited_text` quotes — so a reviewer can see what the model had access to.

`/chat` automatically extracts entities from the question and uses them as a graph pre-filter — same retrieval shape `?entity=…` would produce on `/search`. The pre-filter only activates when the question's surface forms case-insensitively match an alias the resolver stored, so questions phrased with abbreviations the corpus never used (e.g. asking about "Acme" when every doc says "Acme Corporation") silently degrade to plain retrieval. If the entity-filter does activate but narrows candidates to zero (entities are in the corpus but never co-mentioned in a single chunk), the handler retries without the filter so the answer pipeline gets a fair shot. ([§18.9](./DECISIONS.md#189-chat-auto-extracts-entities-and-falls-back-if-the-filter-empties))

</details>

## Architecture

| Area | Choice | Why |
|---|---|---|
| **Stack** | Python 3.14 + FastAPI + Pydantic v2 + `uv`. Postgres 16 + pgvector via Docker Compose. | RAG tooling is Python-first; pgvector handles 100k+ chunks and gives one DB for documents, chunks, embeddings, metadata, *and* lexical ranking. ([§1](./DECISIONS.md#1-stack)) |
| **Embeddings** | Voyage `voyage-4`, 1024-dim, cosine via pgvector's `<=>`. | Voyage is Anthropic's recommended embedding partner; `voyage-4` is the labelled general-purpose default. ([§2](./DECISIONS.md#2-embedding-model)) |
| **Chunking** | `semantic-text-splitter`, sized with Voyage's HuggingFace tokenizer. 600 tokens / 15% overlap. | Walks Unicode graphemes and packs to a token capacity, so chunks stay correctly sized in any script (CJK, Arabic, contiguous blocks). ([§3](./DECISIONS.md#3-chunking-strategy)) |
| **Schema** | Hybrid: typed columns for `title` / `author` / `published_date` + JSONB `metadata` with GIN. | Validated SQL filters where it matters, flexibility everywhere else. ([§4](./DECISIONS.md#4-metadata-schema)) |
| **Vector index** | HNSW, defaults `m=16, ef_construction=64`, cosine ops. | Production default in pgvector ≥0.5; best recall/speed tradeoff. ([§6](./DECISIONS.md#6-vector-index)) |
| **Hybrid search + rerank** | RRF (`k=60`, 50 candidates per lane) → Voyage `rerank-2.5` → top-K. | RRF is parameter-free; no score normalization needed. Voyage reranker is a natural drop-in. ([§7](./DECISIONS.md#7-hybrid-search-fusion) / [§7.2](./DECISIONS.md#72-reranker--voyage-rerank-25)) |
| **Citations** | Anthropic structured citations (`search_result_location`). Each claim carries a verbatim `cited_text` quote tied to a retrieved chunk. | Reviewer can verify claims in seconds; no `[N]` parsing. ([§8](./DECISIONS.md#8-citation-approach-the-load-bearing-one)) |
| **Chat LLM** | Claude Sonnet 4.6, single-turn. | Production default for grounded RAG; single-turn matches the brief. Multi-turn opens a query-rewriting subproblem. ([§9](./DECISIONS.md#9-chat-llm) / [§10](./DECISIONS.md#10-chat-shape--single-turn)) |
| **Knowledge graph** | Per-chunk Haiku extraction (PERSON / ORGANIZATION / LOCATION / EVENT) + per-document-per-type Sonnet resolution. Postgres relations tables; chunk-level AND co-mention on `?entity=`. | Industry-standard per-chunk extraction (GraphRAG / LightRAG / LlamaIndex / Neo4j); native Anthropic Structured Outputs via `messages.parse`; one DB instead of bolting on Neo4j. ([§18](./DECISIONS.md#18-knowledge-graph-closes-30)) |

Deliberate cuts (multi-turn chat, structure-aware PDF chunking, parent-child chunking, contextual embeddings, streaming `/chat`, API key auth) are listed in [`DECISIONS.md §15`](./DECISIONS.md#15-deliberate-cuts-and-why) with effort estimates, each filed as a GitHub Issue ([#3](https://github.com/nainajnahO/rag-knowledge-base/issues/3), [#25](https://github.com/nainajnahO/rag-knowledge-base/issues/25)–[#29](https://github.com/nainajnahO/rag-knowledge-base/issues/29)). The previously-deferred knowledge graph ([#30](https://github.com/nainajnahO/rag-knowledge-base/issues/30)) shipped — see [§18](./DECISIONS.md#18-knowledge-graph-closes-30).

## Data model

Two tables: `documents` (one row per uploaded source — title, author, published date, content hash for dedupe, plus a JSONB `metadata` column) and `chunks` (one row per retrievable text chunk, each with its 1024-dim `voyage-4` embedding and a foreign key back to its document). Full schema in [`sql/schema.sql`](./sql/schema.sql), applied automatically by Postgres on first container init. Rationale in `DECISIONS.md` [§4](./DECISIONS.md#4-metadata-schema) and [§7](./DECISIONS.md#7-hybrid-search-fusion) (lexical column).

## Tests

Four pytest smoke tests covering the load-bearing seams:

1. **Chunker sanity** (`tests/test_chunker.py`) — short text yields one chunk; long text yields contiguous-ordinal chunks under `MAX_TOKENS` with observable overlap; empty text yields none.
2. **Upload → search smoke** (`tests/test_search_smoke.py`) — `POST /text` then `GET /search` returns the just-uploaded document at the top.
3. **Citation consistency** (`tests/test_chat_citations.py`) — every citation in `answer_blocks[*].citations` resolves to a `chunk_id` in `sources`, and each `cited_text` is a verbatim substring of the source chunk.
4. **Entity-filtered search** (`tests/test_graph_smoke.py`) — two docs share an organization but mention different cities; single-entity filter surfaces both, AND filter on org+city excludes the doc that doesn't co-mention; an unknown entity name yields empty results.

```bash
uv run pytest -q   # ~55 seconds with the graph tests; needs both API keys + Postgres up
```

Tests hit the real dev Postgres and the real Voyage / Anthropic upstreams; tests that need an upstream key skip cleanly when the key is unset. Per-test cleanup via a `db_cleanup` fixture; payloads are uuid-salted so each run exercises the chunk→embed→persist path rather than short-circuiting through dedupe. The "three tests, not more" framing is in [`DECISIONS.md §13`](./DECISIONS.md#13-tests).

## Known limitations

- **DB connection held during upstream calls.** `Depends(get_conn)` borrows a pooled connection for the full request lifetime, including upstream LLM/embedding calls (`/chat` now makes up to 4 sequential ones with the graph layer). Pool is 1–10, so this would exhaust under meaningful concurrency. Half-day fix: release-before-upstream + reacquire-after. ([PR #7](https://github.com/nainajnahO/rag-knowledge-base/pull/7))
- **HNSW is approximate.** Recall typically >95%, not 100%. Mitigation: `SET LOCAL hnsw.ef_search = 100` per query — trades ~1ms latency for higher recall. ([§6](./DECISIONS.md#6-vector-index))
- **Lexical lane uses `'simple'` config.** Language-neutral — keeps non-English content from being mis-stemmed, but skips per-language morphology lift. ([issue #3](https://github.com/nainajnahO/rag-knowledge-base/issues/3) / [§7.1](./DECISIONS.md#7-hybrid-search-fusion))
- **Single-turn `/chat`.** Multi-turn requires query rewriting; doing it badly is worse than not doing it. Brief asks for single-turn. ([§10](./DECISIONS.md#10-chat-shape--single-turn))
- **Dedupe is silent.** `POST /text` / `/document` ignore new metadata when content matches an existing document by SHA-256. ([§12](./DECISIONS.md#12-upload-dedupe))
- **Sync graph extraction at ingest** adds ~1-2s of latency per `POST /text` / `POST /document` (per-chunk Haiku calls fan out to a thread pool, then ≤4 Sonnet resolution calls run serially). Acceptable for the demo scale; async + Batch API is the production path. ([§18.4](./DECISIONS.md#184-why-sync-at-ingest-not-asyncbatch-api))
- **Resolution canonicals capped at 200 per type.** Beyond that, older canonicals can be re-split by the resolver. v2 path: embedding-based shortlisting. ([§18.11](./DECISIONS.md#1811-known-limitations-and-their-mitigations--migration-paths))

## AI collaboration notes

Built with Claude Code (Claude Opus 4.7). The notes below document where the collaboration actually mattered — where Claude was wrong and I corrected it, where I pushed back, and where my own tooling caught real bugs. Generic "Claude wrote it, I reviewed it" claims aren't useful signal; specific recoveries are.

### Design phase (before any code)

The project started with an extended planning Q&A — no code, just a structured conversation that produced [`DECISIONS.md`](./DECISIONS.md). For every decision area we ran the same loop: frame the decision, enumerate 3–6 alternatives (including the obvious pick, the dismissed one, and at least one neither of us would have surfaced first), compare them on the dimensions that matter for *this* project (not generic best-practice scoring), then lock in a choice and write the migration path. Every locked-in choice has a "Migration notes" subsection so the next person doesn't have to re-derive the analysis.

### Where I steered

- **`DECISIONS.md` as a living steering artifact.** The whole document is the steering mechanism — not a one-shot spec. It was *planned* before any code, *iteratively built* across the design Q&A, and *edited* every time a choice changed . Pointing Claude at "go read §7 and stay consistent" is dramatically more reliable than re-explaining the project context every session — the file *is* the context.
- **Anthropic structured citations.** Claude's initial proposal for the citation layer was the conventional `[N]` footnote-marker pattern — model emits bracketed numbers in prose, we post-parse them and map back to retrieved chunks. I steered toward Anthropic's first-class structured citations API (`search_result_location` blocks with verbatim `cited_text` quotes) instead. This removes the parsing layer entirely, makes claims verifiable as a substring check, and is the load-bearing piece of the whole system per the brief — getting it wrong would have undercut the entire "verifiable in seconds" goal. ([§8](./DECISIONS.md#8-citation-approach-the-load-bearing-one))
- **GitHub Issues as pre-formulated work units.** Deliberate cuts ([#3](https://github.com/nainajnahO/rag-knowledge-base/issues/3), [#25](https://github.com/nainajnahO/rag-knowledge-base/issues/25)–[#30](https://github.com/nainajnahO/rag-knowledge-base/issues/30)), bugs, and follow-ups are filed as Issues with the investigation already written down. Steering Claude with "fix #28" pulls in a fully scoped problem statement — constraints, alternatives, effort estimate — instead of relying on me to re-articulate context in chat. Issues isolate concerns and survive across sessions; chat prompts don't.
- **Doc-drift audit.** I scheduled a project-health audit (cross-checking `DECISIONS.md` against actual code) before Step 4. It surfaced three drift items, including §11 understating the route's error-status taxonomy. Claude wouldn't have caught these without the explicit prompt — no incentive to re-read its own writing critically. ([PR #12](https://github.com/nainajnahO/rag-knowledge-base/pull/12))
- **Verifying staged set before every commit.** PyCharm has occasionally auto-staged unrelated changes, so `git status` + `git diff --cached` runs before *every* commit.

### Where Claude was wrong and I corrected it

- **LangChain framing.** Claude justified the custom chunker by claiming `RecursiveCharacterTextSplitter` *can't* do token-based splitting. Wrong — it accepts a `length_function`. The honest tradeoff was visibility-of-the-algorithm vs. using a library. `DECISIONS.md §3` was rewritten; the chunker has since been swapped to `semantic-text-splitter` ([PR #32](https://github.com/nainajnahO/rag-knowledge-base/pull/32)).
- **"Latin separators are fine."** The first chunker landed with `["\n\n", "\n", ". ", " "]` — broken for CJK / Thai / contiguous blocks (would silently truncate at Voyage's 32K cap). Claude defended the design; I pressed; the `_token_slice` token-offset fallback was added and validated across English / Japanese / Arabic / Hebrew / Thai / mixed scripts / emoji. ([PR #10](https://github.com/nainajnahO/rag-knowledge-base/pull/10))
- **`@functools.cache` thread-safety claim.** Claude wrote *"`@cache` is thread-safe"* in a PR body. Reading CPython's `_lru_cache_wrapper` directly: the `maxsize=None` path uses no lock at all. PR body was corrected. ([PR #9](https://github.com/nainajnahO/rag-knowledge-base/pull/9))
- **"Doesn't hold a DB connection during upstream calls."** Claude's first PR #7 body claimed this benefit. The connection *is* held throughout via `Depends(get_conn)`. Misleading sentence removed. ([PR #7](https://github.com/nainajnahO/rag-knowledge-base/pull/7))


### Skills I authored and ran

The following are slash-commands **I wrote** and triggered manually — not autonomous. They exist precisely because I needed independent checks on Claude's output:

- **`/pr-review`** — multi-pass deepening review. On `POST /text` (PR #7), pass 2 caught a latent bug: `conn.rollback()` called inside `with conn.transaction()` raises `ProgrammingError` (verified against psycopg source — `_rollback_gen` checks `_num_transactions > 0`). Without this skill, the bug would have shipped.
- **`/native-check`** — looks for hand-rolled code that duplicates language/framework features. Caught the manual `global _client` sentinel and recommended `@functools.cache` ([PR #9](https://github.com/nainajnahO/rag-knowledge-base/pull/9)); also caught `Depends()`-as-default and recommended migrating to `Annotated[]` ([PR #11](https://github.com/nainajnahO/rag-knowledge-base/pull/11)).
- **`/dry-check`** — checks for shared types and data-driven rendering. No findings; logged for honesty.
- **`/architecture-check`** — audits architectural overkill. Run before Step 4 to confirm the design was right-sized for take-home scale.

These skills are the most concrete signal of how I work with Claude: I don't trust a single pass of its review; I run independent checks I authored, and I act on what they find.
