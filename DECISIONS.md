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
- pgvector handles 100k+ chunks comfortably and gives us *one* database for documents, chunks, embeddings, metadata, *and* lexical ranking (via tsvector). A dedicated vector DB would force two stores in sync for no benefit at this scale.
- Python 3.14 because "default to the latest stable" is the rule. All deps in our list (`fastapi`, `pydantic`, `psycopg`, `voyageai`, `anthropic`, `pymupdf`) ship 3.14 wheels.
- Docker Compose because it's reproducible by the reviewer with one command and avoids hosted-service signup friction. Free-tier hosted Postgres (Supabase/Neon) was rejected: pausing free tiers, shared credentials, and using ~5% of a managed BaaS would be exactly the unnecessary abstraction the brief warns against.

**Migration notes:** None — this is a load-bearing decision. Don't change the language or vector store later.

---

## 2. Embedding model

**Decision:** Voyage `voyage-4`, 1024 dimensions, cosine distance.

**Alternatives considered:**
- `voyage-4-large` (same 1024-dim default with Matryoshka 256/512/2048; "best general-purpose & multilingual" tier — rejected as overshooting; "biggest model" needs corpus-specific benchmarks to defend, which a take-home doesn't have. Also has a tighter per-request token cap of 120K vs voyage-4's 320K, so any future migration would also need the embedder to re-tune its sub-batching limits.)
- `voyage-4-lite` (same dim options; "optimized for latency and cost" tier — rejected as undershooting on a quality-graded RAG eval where cost differences are pennies)
- `voyage-4-nano` (open-weight, smallest of the 4-series — interesting but no meaningful advantage over `voyage-4` here)
- `voyage-context-3` (1024-dim, contextual embeddings — each chunk's embedding incorporates surrounding-document context; deferred per migration notes below)
- `voyage-3` / `voyage-3.5` / `voyage-3-large` (previous generation per Voyage's own docs — Voyage explicitly recommends migrating to the 4-series; no reason to ship stale)
- OpenAI `text-embedding-3-small` (1536 dim — cross-vendor option; rejected to keep the "Anthropic stack end-to-end" narrative)
- Local sentence-transformers model (no inference infra; rejected)

**Why this choice:**
- `voyage-4` is Voyage's labelled general-purpose default for retrieval — using the recommended default is a stronger story than picking the biggest tier and defending it.
- Voyage is Anthropic's recommended embedding partner; clean README narrative ("used Anthropic's stack end-to-end").
- 1024 dimensions is a clean column type for pgvector with no storage bloat. Matryoshka variants (256/512/2048) are available without a model swap if storage or speed ever becomes a real constraint.
- Cosine via pgvector's `<=>` operator — the conventional choice for retrieval and the one Voyage's documentation examples use. voyage-4 embeddings are L2-normalized, so cosine, dot product, and Euclidean rank identically anyway.
- 32K-token per-input context length is far above our 600-token chunks — no risk of truncation.

**API usage notes (load-bearing for the embedder module):**
- **Always pass `input_type`.** Voyage prepends a different internal prompt depending on whether the text is being embedded as a *document* (for indexing) or as a *query* (for retrieval). Skipping `input_type` works but produces weaker retrieval. The embedder module's signature is `embed_chunks(chunks, input_type="document") -> list[list[float]]`; ingestion calls it with the default, search/chat (Step 5/6) call it with `input_type="query"`. Embeddings produced with vs without `input_type` are still mutually compatible — this is purely a quality lever.
- **Per-request caps for voyage-4:** ≤1,000 inputs AND ≤320,000 tokens per `/v1/embeddings` call. The embedder module owns sub-batching and enforces both ceilings simultaneously. With 600-token chunks, the token cap binds first at ~533 chunks per call; almost no real document hits either limit in one batch.
- **`output_dtype="float"`** (the default) — full-precision 32-bit embeddings. Quantized variants (int8 / binary) save storage but lose recall; not worth the complexity at this scale.

**Migration notes — voyage-4 → voyage-context-3 later (cheap):**
- Same default dimension (1024) → **no schema migration**.
- Same distance metric → no retrieval code change.
- Embedder module changes (~30 lines): different SDK method (`vo.contextualized_embed(...)`), different REST endpoint (`/v1/contextualizedembeddings`), and the API takes document-grouped chunk lists (`List[List[str]]`) rather than a flat list of strings.
- Tighter per-request caps: ≤1,000 inputs / ≤120K tokens / ≤16K chunks total — embedder's sub-batching needs to track all three.
- Ingestion pipeline must batch chunks **by document** (which is the natural flow anyway — one doc per upload).
- Re-embed all existing chunks (cents at this scale).
- Total effort: roughly half a day of focused work.
- **Design implication for now:** keep the embedder module behind a single function signature; don't interleave chunks from different documents in batched embedding calls.

---

## 3. Chunking strategy

**Decision:** Recursive token-based splitting (paragraph → sentence → word → token-offset fallback) for both `/text` and `/document` endpoints. **600 tokens per chunk, 15% overlap (~90 tokens).** Token counting via Voyage's tokenizer, not characters.

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

**Why custom code instead of `langchain-text-splitters`:** LangChain's `RecursiveCharacterTextSplitter` accepts a `length_function`, so it can be configured to be token-based. Honest tradeoff: ~60 lines of custom code vs. a small dependency. We chose custom to keep the chunking strategy fully visible in the codebase, avoid LangChain's package-restructuring history, and minimize dependency footprint. A swap is mechanical if multilingual or hierarchical chunking ever justifies the dependency.

**Multilingual fallback (token-offset slicing).** The semantic separators (`\n\n`, `\n`, `. `, ` `) are Latin-script-biased — CJK languages without inter-word spaces, or any contiguous block, would otherwise survive recursion intact and either truncate at Voyage's 32K-token input cap or produce one oversize chunk per document. The deepest fallback (`_token_slice` in `app/chunking.py`) uses Voyage's tokenizer to read each token's `(char_start, char_end)` offset and slices the text into `OVERLAP_TOKENS`-sized pieces by character index. The packer then produces correctly-sized windows with overlap from those small pieces, exactly as it does for separator-derived pieces. Result: chunks are ≤ MAX_TOKENS in any language, overlap is preserved at chunk boundaries, the embedding never gets a >32K-token input. Not as semantically aligned as Latin-text chunking (boundaries can fall mid-word in any language; mid-sentence in CJK), but functionally correct everywhere.

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

**Required (typed) on upload:** `title`.
**Optional (typed):** `author`, `published_date`.
**Catch-all:** `metadata jsonb` with a GIN index for arbitrary extras (including any categorical tagging the user wants to attach).

**A note on `source_type`:** earlier drafts of this design included a typed `source_type` column with a fixed taxonomy (`'memo' | 'report' | 'article' | 'text'`) inspired by the brief's mention of "memos, reports, articles." The brief uses that phrase only as background context describing what users might upload — it does *not* require a fixed categorization, and the metadata schema is explicitly "what you define." Baking the taxonomy into the schema would be inventing constraints the brief doesn't impose. Categorical tagging instead lives in the JSONB `metadata` column (e.g., `{"type": "memo", "department": "finance"}`) and is filterable via `@>` containment, indexed by GIN.

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
    author          TEXT,
    published_date  DATE,
    metadata        JSONB NOT NULL DEFAULT '{}',
    raw_text        TEXT NOT NULL,
    content_hash    TEXT NOT NULL UNIQUE,     -- SHA-256 for dedupe
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_documents_published_date ON documents (published_date);
CREATE INDEX idx_documents_metadata       ON documents USING GIN (metadata);

CREATE TABLE chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INT NOT NULL,
    text         TEXT NOT NULL,
    token_count  INT NOT NULL,
    embedding    vector(1024) NOT NULL,
    UNIQUE (document_id, ordinal)
);
CREATE INDEX idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

The `tsv` column and its GIN index land later, in the hybrid-search step (§16, Step 7). They're scaffolding for the optional lexical lane and don't belong in the must-have schema.

**Filter API on `/search`:**
- Typed query params: `author`, `published_after`, `published_before`
- JSONB containment: `metadata @> '{"type": "memo"}'` (the API surface accepts simple `meta.<key>=<value>` query params and translates them to JSONB containment server-side)
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

### 6.1 Iterative scan for filtered queries

`/search` and `/chat` retrieve with metadata filters joined to `documents`. Naive HNSW + post-filter can underfill `LIMIT k` when filters reject most candidates: HNSW returns its initial batch (`hnsw.ef_search`, default 40), the filter rejects some, and you get back fewer than `k`. To handle this, `app/retrieval.py` sets `SET LOCAL hnsw.iterative_scan = 'strict_order'` (pgvector ≥0.8.0) inside the retrieval transaction. The index keeps walking the graph past its initial batch until enough rows pass `WHERE` to fill `k`, while preserving similarity ordering. Bounded by `hnsw.max_scan_tuples` (default 20,000) so very selective filters don't trigger a full scan.

`strict_order` over `relaxed_order` because results are returned to the client in displayed score order — out-of-order results would need an outer sort to fix, gaining nothing at this scale.

**Migration notes:** none. The GUC is per-transaction; bumping `max_scan_tuples` if very selective filters routinely underfill is a one-line SET inside `retrieve()`.

**Validation gap (honest note):** at the smoke-test corpus size (~5 chunks), the planner picks Seq Scan + Hash Join, not HNSW — `EXPLAIN` confirmed. Iterative scan only engages when the planner has chosen an HNSW Index Scan node, which won't happen until the corpus is large enough that the HNSW path beats brute-force on cost. The mechanism is therefore production-default insurance, not validated empirically at the only scale this code has been tested at.

**Why iterative scan and not over-fetch CTE (revisited).** During PR #18 review we re-examined whether to swap `SET LOCAL hnsw.iterative_scan` for an over-fetch CTE pattern (inner CTE pulls top-200 by distance, outer JOIN+filter+LIMIT k). Verified the alternative empirically with `EXPLAIN ANALYZE` on the live DB: at the small-corpus / no-HNSW-engaged case, our current shape is **strictly better** — 0.16ms vs 0.28ms execution, 38 vs 67 buffer hits, and it has no underfill failure mode (the JOIN+filter eliminates rows before the sort, where over-fetch CTE truncates to 200 candidates and only then filters). The case for over-fetch CTE was never the no-HNSW path; it was decoupling from the planner's HNSW decision when HNSW *would* be useful but the planner refuses. pgvector 0.8.0's improved cost model is supposed to fix that planner reluctance directly — going back to over-fetch CTE in 2026 is "doing things the pre-0.8.0 way." Decision: keep `SET LOCAL hnsw.iterative_scan`, accept the validation gap, trust pgvector's design over re-implementing it in SQL. Issue #19 closed with these findings.

**Score range:** the SELECT computes `1 - (embedding <=> :q)` where `<=>` is cosine distance in `[0, 2]`, so `score ∈ [-1, 1]` mathematically. voyage-4 embeddings are L2-normalized, and natural-language pairs almost always score ≥ 0; negative scores indicate the query and chunk are pointing in opposite directions in vector space (rare in practice).

---

## 7. Hybrid search fusion

**Decision:** Reciprocal Rank Fusion (RRF), constant `k=60`.

**Terminology note:** This section uses "lexical ranking" or "lexical lane" rather than "BM25" for the keyword-search component. Postgres's full-text search ships with `ts_rank` and `ts_rank_cd`, which are term-frequency-based ranking functions — *not* the canonical BM25 formula (which adds IDF and saturation parameters `k1`, `b`). The role is identical (lexical complement to the dense lane) and RRF fusion is rank-based, so the specific score formula matters less than the rank ordering it produces. Real BM25 in Postgres requires the `pg_search` extension (ParadeDB), deemed unnecessary at take-home scale. The brief uses "BM25" as a category label; we read it that way.

**Alternatives considered:**
- Linear weighted combination `α · vector + (1−α) · lex` (requires score normalization and tuned α; `lex` here standing in for whatever lexical score we use)
- Convex combination with min-max or z-score normalization (same problem, more steps)
- CombSUM / CombMNZ (older, superseded by RRF)
- Skipping hybrid entirely (vector-only)

**Why this choice:**
- Parameter-free — no tuning data needed, no α to defend.
- No score normalization needed (lexical scores from `ts_rank` are query-dependent in magnitude, cosine is `[-1, 1]`; mixing them naively is broken).
- Industry standard in 2026 (Elastic, Vespa, Weaviate, Qdrant all default to RRF).
- One SQL query with two CTEs and a JOIN — ~20 lines, no Python-side merging.

In the SQL sketch below, `$1` is the query embedding, produced by the same embedder module used during ingestion but called with `input_type="query"` per §2's API usage notes; `$2` is the raw query string.

**Implementation sketch** *(the `tsv` column referenced below lands later, in the hybrid-search step (§16, Step 7); the must-have schema omits it)*:

```sql
WITH vec AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
  FROM chunks ORDER BY embedding <=> $1 LIMIT 50
),
lex AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, q) DESC) AS rank
  FROM chunks, plainto_tsquery('simple', $2) q
  WHERE tsv @@ q LIMIT 50
)
SELECT c.id, c.text, c.document_id,
       COALESCE(1.0/(60+vec.rank), 0) + COALESCE(1.0/(60+lex.rank), 0) AS rrf_score
FROM chunks c
LEFT JOIN vec ON c.id = vec.id
LEFT JOIN lex ON c.id = lex.id
WHERE vec.id IS NOT NULL OR lex.id IS NOT NULL
ORDER BY rrf_score DESC
LIMIT 10;
```

### 7.1 Lexical lane language config — `'simple'`

The `tsvector` column needs a Postgres text-search config that governs tokenization, stemming, and stop-word filtering. The brief doesn't specify a corpus language, so we picked the language-neutral option.

**Alternatives considered:**
- **`'english'`** — Porter stemmer + English stop words. Best lexical recall on English prose. Actively bad on non-English text: mis-stems Norwegian, French, etc., turning the lexical lane into noise on those docs.
- **A specific non-English config** (e.g., `'norwegian'`) — same shape, different bet on which single language wins. Wrong if the corpus turns out to be in a different language.
- **Per-document language tracking** — store `language` per document, copy onto each chunk, generate `tsv` with the per-row config (`to_tsvector(language, text)`). Properly multilingual lexical ranking, but requires language detection at upload and query time, plus a denormalized `language` column on chunks (Postgres GENERATED columns can only reference same-row data). See migration notes below.

**Why `'simple'`:**
- The lexical lane's main value-add in our hybrid is exact-term matching (proper nouns, codes, acronyms, technical jargon). `'simple'` delivers that for any language.
- The morphology and stop-word lift of a language-specific config (`"policies"` matching `"policy"`) is partially absorbed by voyage-4 in the dense lane — semantically equivalent inflections produce nearby vectors.
- The brief is silent on language. `'english'` would be a bet that *penalizes* non-English content rather than just leaving it un-optimized.
- Cross-language retrieval is unaffected by this choice — voyage-4 (multilingual) carries that case through the dense lane regardless of what we put in the lexical lane.

**Migration notes — `'simple'` → per-document language tracking (~half-day):**
- Add `language regconfig NOT NULL` on `documents` (default `'simple'`) — detected at upload via a small library (`langdetect`, `fasttext-langid`) or supplied by the client.
- Add `language regconfig NOT NULL` on `chunks` — denormalized from the parent doc on insert (Postgres GENERATED columns can't reference other tables).
- Drop the existing `tsv` column and `idx_chunks_tsv` GIN index.
- Re-add `tsv` as `GENERATED ALWAYS AS (to_tsvector(language, text)) STORED`.
- Recreate the GIN index.
- At query time: detect (or accept via query param) the query's language, parse with `plainto_tsquery(detected_lang, :q)`.
- Tracked in [issue #3](https://github.com/nainajnahO/rag-knowledge-base/issues/3).

---

## 8. Citation approach (the load-bearing one)

**Decision:** Anthropic's first-class **Search Result content blocks** with structured citations (`cited_text` returned in API response). Each retrieved chunk becomes a `search_result` block with `citations.enabled = true`; Claude's response is a list of text blocks where cited blocks carry `citations: list[search_result_location]` with verbatim `cited_text`. We pass through Anthropic's native shape rather than remapping into a custom `[N]` numbered-citation scheme.

**Alternatives considered:**
- **Prompt-based numbered citations (`[1]`, `[2]`)** — what we originally locked. Teach the model `[N]` syntax in the system prompt, parse `[N]` markers from the answer text, map back to chunks. Works, but Anthropic's own evals (their public docs) report this approach hallucinates citations significantly more than their structured-citations feature, and `[N]` is parseable-but-not-guaranteed (the model can fabricate a `[5]` when only 4 chunks exist).
- **Structured JSON output with per-claim `supported_by` arrays** — incompatible with Citations per Anthropic's docs (a 400 from the API). Skipped on those grounds; would also re-introduce the prompt-engineering verifiability burden.
- **Span-level citation via prompt** — model returns the exact substring supporting each claim. Brittle; models hallucinate spans. Anthropic's structured citations are precisely this idea, done correctly at the API level.
- **Post-hoc verification pass** — second LLM call to validate each claim. 2× cost, 2× latency, doesn't add over what Anthropic's structured citations already guarantee.

**Why this choice:**
- The brief grades **"is it actually possible to verify what the LLM says?"** and **"does the system hallucinate?"** Anthropic's structured-citations API is the strongest available answer to both:
  - `cited_text` is **guaranteed** to be a verbatim substring of the provided chunk (per Anthropic docs: *"citations are guaranteed to contain valid pointers to the provided documents"*).
  - In Anthropic's published evals, the structured-citations feature *"was found to be significantly more likely to cite the most relevant quotes from documents as compared to purely prompt-based approaches."*
- Returning all retrieved chunks (not just cited ones) lets the reviewer see what the model had access to and judge if it cited the right ones — preserved from the original §8 intent.
- The system prompt becomes shorter and easier to audit because the API handles citation formatting; we only have to enforce three guardrails (sources-only, refusal allowed, no speculation).
- Native Anthropic shapes are passed through to the response (`answer_blocks`) so a client can render claim-with-quote inline if it wants. A flat `answer: str` is also returned for simple display.

**Anti-hallucination guardrails (still all four; mechanism shifts):**

1. **System prompt forbids external knowledge.** *"You answer questions using only the provided search results... Do not use prior knowledge. Do not speculate."* The single most important line.
2. **Refusal allowed and encouraged.** *"If the search results do not contain enough information to answer the question, say: 'I don't have enough information in the provided sources to answer this.'"* In-prompt instruction.
3. **Score threshold gate (cosine ≥ 0.5).** Lives in the route layer, not the prompt: if `chunks[0].score < 0.5` (or 0 chunks retrieved), don't call Anthropic — return the refusal response with `sources: []`. Saves a ~$0.02 call when retrieval is hopeless.
4. **Chunk metadata passed automatically.** Each `search_result` block has a `title` field set to `"<doc_title> (<published_date>)"` (date when present). The model sees this natively without any prompt-engineering — no need to inject `Title: ... (date)` headers into chunk text.

**Top-K retrieval for chat:** 8 chunks above threshold 0.5. Sweet spot — enough context for nuanced questions without dilution from low-relevance chunks.

**Per-chunk packaging.** Each retrieved chunk → one `search_result` content block:

```python
{
    "type": "search_result",
    "source": str(chunk.chunk_id),                 # UUID — granularity matches retrieval
    "title": f"{doc_title} ({published_date})"     # date appended when present
            if published_date else doc_title,
    "content": [{"type": "text", "text": chunk.text}],   # one block per chunk
    "citations": {"enabled": True},
}
```

`source = chunk_id` (not `document_id`) because the verifiability story is "this exact quote came from this exact chunk." Citations come back with `source` carrying the same `chunk_id`, so we can do an exact dict lookup `chunk_by_id[citation.source]` to map back.

One block per chunk for this PR. Splitting chunks into sentence-level blocks would give finer citation granularity (`cited_text` is the joined content of `content[start_block_index:end_block_index]`, so finer blocks = shorter cited quotes). Deferred — `cited_text = full chunk` is already verifiable by `grep`-ing our `chunks.text`. Sentence-splitting is a local change in `app/llm.py`'s `build_search_result_block` if/when we want it.

**Request structure (Anthropic's "Method 2: top-level search results"):** user message contains the 8 `search_result` blocks first, then a `text` block with the question. No tool-use loop (Method 1 is for dynamic tool-driven RAG; we have pre-fetched content).

**Response shape:**

```json
{
  "answer": "Revenue grew 12% in Q3 2025, driven by enterprise contracts.",
  "answer_blocks": [
    {
      "text": "Revenue grew 12% in Q3 2025, driven by enterprise contracts.",
      "citations": [
        {
          "chunk_id": "uuid",
          "document_title": "Q3 Revenue Memo",
          "published_date": "2025-10-15",
          "cited_text": "Revenue grew 12% in Q3 2025 to $4.2M, driven by enterprise contracts."
        }
      ]
    }
  ],
  "sources": [
    {
      "chunk_id": "uuid",
      "ordinal": 0,
      "document_id": "uuid",
      "document_title": "Q3 Revenue Memo",
      "author": "Finance Team",
      "published_date": "2025-10-15",
      "metadata": {"type": "memo", "department": "finance"},
      "score": 0.87,
      "text": "Revenue grew 12% in Q3 2025 to $4.2M, driven by enterprise contracts. The finance team expects continued growth into Q4...",
      "cited": true,
      "cited_text": ["Revenue grew 12% in Q3 2025 to $4.2M, driven by enterprise contracts."]
    }
  ],
  "stop_reason": "end_turn"
}
```

`answer` is the concatenation of all `answer_blocks[i].text` for clients that just want a string. `answer_blocks` is Anthropic's native shape (preserving the per-block citation linkage). `sources` is all 8 retrieved chunks (not just cited ones, per the original §8 intent — let the reviewer see what the model had access to). `cited: bool` distinguishes which chunks the model actually grounded claims on; `cited_text: list[str]` carries the verbatim quotes (multiple if the model cited the same chunk in multiple `answer_blocks`). `stop_reason` is passed through from Anthropic — `"end_turn"` is the healthy case; clients should treat `"max_tokens"` as a truncation signal and `"refusal"` as a model-level decline.

**Refusal response shape** (top score < 0.5 OR 0 chunks retrieved):

```json
{
  "answer": "I don't have enough information in the provided sources to answer this.",
  "answer_blocks": [
    {
      "text": "I don't have enough information in the provided sources to answer this.",
      "citations": []
    }
  ],
  "sources": [],
  "stop_reason": null
}
```

`sources: []` because we never called Anthropic — the LLM didn't "have access to" anything. `stop_reason: null` for the same reason — there was no model call to report a stop_reason for. If the reviewer wants to see what retrieval found, they can hit `/search?q=<question>` directly.

**Migration notes — one block per chunk → sentence-splitting (~hour, local):**
- In `app/llm.py.build_search_result_block`, split `chunk.text` into sentences (`pysbd` or simple `re.split`), emit one `{"type": "text", "text": s}` per sentence in the `content` array.
- No schema migration, no other code changes. `cited_text` becomes shorter and per-claim-precise.
- Tracked as a follow-up issue if quality demands surface.

**Migration notes — Anthropic structured citations → multi-vendor portability (~half-day):**
- This locks `/chat` into Anthropic's API shape. If we ever swap LLM providers, we'd need to reimplement citations. §9 already locked Sonnet 4.6 and treats LLM swap as "one config line" — so this isn't really new lock-in.
- If/when we want to abstract: introduce a `Citation` adapter layer that wraps either Anthropic's structured citations or a prompt-based fallback for other providers. Adds one indirection layer.

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

**Constants:** `app/llm.py` exposes `CHAT_MODEL = "claude-sonnet-4-6"` (alias, not a dated snapshot — auto-tracks the latest minor revision; swap to `"claude-sonnet-4-6-YYYYMMDD"` for production reproducibility) and `CHAT_MAX_TOKENS = 2048`.

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

**Status codes used explicitly:**

- **422** — Pydantic validation (free). Includes the 3M-character extracted-text cap on both `/text` and `/document` (see §17).
- **413** — multipart file body exceeds the 25 MB cap on `POST /document` (see §17). Rejected at the door before extraction.
- **400** — bad file/input (e.g., empty text, malformed PDF, missing `%PDF-` magic bytes) AND upstream rejection of the input (Voyage `InvalidRequestError`).
- **401** — missing/bad API key (Step 8 — not yet wired).
- **429** — upstream rate limited (Voyage `RateLimitError`).
- **500** — server misconfig (e.g., missing/invalid `VOYAGE_API_KEY` surfacing as Voyage `AuthenticationError`). Distinguished from 503 so operators don't mistake a config bug for a transient upstream issue.
- **503** — upstream transient failure (generic `VoyageError`; Anthropic upstream errors in Step 6 will follow the same hierarchy).

**Vendor error mapping locations.** Voyage SDK errors → `map_voyage_errors()` in `app/embeddings.py`; used by `embed_with_error_mapping` (chunks, in `app/ingest.py`) and `embed_query_with_error_mapping` (queries, in `/search` and `/chat`'s retrieval call). Anthropic SDK errors → `map_anthropic_errors()` in `app/llm.py`; used by `generate_answer` for `/chat`. Both context managers map to the same status-code hierarchy so operators see one taxonomy regardless of which upstream failed:
- `AuthenticationError` / `PermissionDeniedError` → 500 (server misconfig — operators don't mistake a missing/wrong API key for a transient blip and wait it out).
- `BadRequestError` / `InvalidRequestError` → 400 (upstream rejected our payload).
- `RateLimitError` → 429.
- Catch-all `APIError` (Anthropic) / `VoyageError` (Voyage) → 503 (timeout, ServiceUnavailable, generic upstream).

No retries, no circuit breakers, no custom exception hierarchy.

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
| **voyage-context-3 contextual embeddings** | Same dimension as voyage-4 → cheap migration. Better story to ship the standard voyage-4 model first; document-grouped batching adds complexity worth deferring. | ~half-day |
| **Streaming responses on /chat** | Brief lists it as stretch. Demo is via curl; streaming complicates citation parsing for the reviewer. | ~few hours |
| **Re-ranking with a cross-encoder** (Voyage `rerank-2.5`) | Real quality improvement but adds a dependency and another model API. RRF gets us most of the way there. Voyage's `rerank-2.5` (32K context, multilingual, instruction-following) is the natural drop-in given the rest of the stack. | ~half-day |
| **Knowledge graph (entities + relations)** | Genuinely interesting, big rabbit hole. Not the foundation the brief asks for. | ~2-3 days |
| **Per-document language tracking for multilingual lexical ranking** ([#3](https://github.com/nainajnahO/rag-knowledge-base/issues/3)) | Hybrid uses `'simple'` tsv config — covers any language adequately but skips per-language morphology and stop-word lift. Real per-language lexical morphology needs per-doc language detection. See §7.1. | ~half-day |
| **Production logging/metrics/CI** | Brief explicitly excludes these. | n/a |

---

## 16. Build sequence

PRs are sized to be individually reviewable. Each merges to `main` with a merge commit (not squash) so individual commits survive.

> **Numbering.** The steps below are **build-sequence ordinals**, not GitHub PR numbers. Quality follow-ups (e.g., chunker fallbacks, dependency-injection refactors) get their own GitHub PRs interleaved with feature PRs, so GitHub `#N` and "Step N" in this list are not the same N. Use `gh pr list --state merged` to see the actual PR-by-PR landing order.

1. **Step 1** — Scaffold: `.gitignore`, `pyproject.toml` (uv, Python 3.14), `docker-compose.yml`, `app/main.py` with `/health`, `.env.example`, README skeleton.
2. **Step 2** — Schema: `sql/schema.sql` with `documents` and `chunks` tables, pgvector extension, HNSW vector index, and GIN index on JSONB metadata. Auto-applied on first container init via `/docker-entrypoint-initdb.d/`. Lexical-ranking column (`tsv` + GIN) deferred to Step 7.
3. **Step 3** — `POST /text` end-to-end: chunker module, Voyage embedder module, persistence.
4. **Step 4** — `POST /document`: pymupdf extraction + content-hash dedupe.
5. **Step 5** — `GET /search`: vector similarity with metadata filters via JOIN. **Done.** k=10/max=50, no score threshold, AND-across-keys metadata filters, HNSW iterative scan with `strict_order` (§6.1), shared `RetrievedChunk` model with `/chat`. See PR for the full set of locked design choices.
6. **Step 6** — `POST /chat`: retrieval + Sonnet 4.6 + structured citations + four guardrails. **Done.** Path B chosen during planning research after surfacing Anthropic's first-class Search Result content blocks (§8 rewrite). Response carries `answer` (display), `answer_blocks` (native Anthropic shape), and `sources` (all retrieved chunks with `cited` flag and verbatim `cited_text`). See PR for the full set of locked design choices.
7. **Step 7** — Hybrid search: tsvector column + RRF query.
8. **Step 8** — API key auth dependency.
9. **Step 9** — Smoke tests (pytest).
10. **Step 10** — README polish, sample documents demonstrating metadata filtering, curl/Postman collection, opening "deliberate cuts" issues.

If time gets tight, drop Step 7 and Step 8 — they become Issues with explanations, which is also signal.

---

## 17. Upload size limits

**Decision:** Two-layer cap on uploaded content:

- **25 MB file-size cap** on the multipart body of `POST /document`. Returns **413 Payload Too Large**, rejected at the door before the body bytes are read into memory.
- **3,000,000 character cap on extracted text** on both `POST /text` and `POST /document`. Returns **422 Unprocessable Entity** via Pydantic validation on `/text`, and via a page-by-page running-total check inside `app/extraction.py` (raises `TextTooLargeError`, mapped to 422 by the route) on `/document`.

**Alternatives considered:**

- **No caps at all.** Simpler but no defense against either RAM blow-up (huge file read into memory before extraction) or runaway embedding cost (huge text → thousands of chunks → many Voyage calls).
- **File-size cap only.** Catches huge uploads cheaply at the door, but a text-heavy 25 MB PDF can extract to ~20 MB of plain text → ~10,000 chunks at 600 tokens/chunk → real Voyage cost + minutes of latency per upload.
- **Text-length cap only.** Catches runaway content cost, but a 500 MB PDF still gets fully read into RAM *before* extraction tells us to reject it. The text-length cap fires too late to defend memory.
- **Tighter file cap (~5 MB) sized to make the text cap unnecessary.** Math: 3M chars of text-heavy PDF ≈ 5 MB file. Aggressive — would reject many legitimate academic papers (image-heavy academic PDFs are routinely 5–10 MB).

**Why two layers:**

- **File-size cap (25 MB)** protects RAM at the door. PyMuPDF reads the whole file into memory before extraction; without a door check, a single malicious upload could exhaust the worker. 25 MB covers any normal document (memos ≤ 100 KB, reports 1–2 MB, image-heavy academic papers 5–10 MB). The per-request memory ceiling is **input + extracted-text buffer**: 25 MB for the file bytes plus up to ~2× `MAX_TEXT_CHARS` for the in-flight `parts` list and joined string (≈ 24 MB worst case in Python), bounded because `extract_text` aborts page-by-page once the running total crosses `MAX_TEXT_CHARS` (otherwise compressed text streams could expand far beyond the 25 MB input). Multiply by uvicorn's effective concurrency (Starlette's threadpool size, default 40) to get the worker-level ceiling. (DB pool size doesn't gate this; body parsing happens before the conn dep blocks on a pool slot.)
- **Text-length cap (3M chars)** protects against runaway chunking + embedding cost *after* extraction. At ~4 chars/token, 3M chars ≈ 750K tokens ≈ ~1,400 chunks at 600 tokens/chunk with 15% overlap. Without this cap, a text-heavy 25 MB PDF could produce ~10,000 chunks per upload — real money in Voyage embeddings + minutes of latency.
- **Symmetry between `/text` and `/document`.** Both endpoints feed the same chunking + embedding + persistence pipeline, so they share the same content limit. A user can't bypass `/text`'s 3M-char cap by repackaging the same content as a PDF.

**Why these specific numbers:**

- **25 MB file cap.** Typical memo: 100 KB. Typical report: 1–2 MB. Image-heavy academic paper: 5–10 MB. Slide decks with embedded raster images: 10–20 MB. 25 MB covers all of these comfortably while still bounding worst-case memory at a manageable ceiling. Round number, easy to remember, easy to widen later if a real document hits the cap.
- **3M character cap.** ~750K tokens of text → ~1,400 chunks. The largest legitimate single-upload artifact a reviewer would plausibly send is a textbook chapter at ~50K characters; 3M gives 60× headroom while still bounding the worst case at ~$0.15 in Voyage cost per upload (vs. ~$1+ unbounded).

**Implementation:**

- **File-size cap.** Enforced via FastAPI middleware that reads the `Content-Length` header before the multipart body is parsed. Reject early with 413 — never reads the body bytes. (For requests without `Content-Length`, fall back to a streaming-byte counter that aborts at 25 MB.) Middleware is **global** (applies to every endpoint), not path-scoped to `/document`. Free for the other endpoints — `/text` is already bounded by the 3M-char cap (≤ ~12 MB UTF-8 worst case) and `/chat` request bodies are tiny — and avoids a per-path branch in the middleware.
- **Text-length cap.** `/text` enforces it via the reusable type alias `MaxText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]` in `app/limits.py` (`IngestTextRequest.text: MaxText` replaces today's inline `Field(max_length=3_000_000)`). `/document` enforces the same constant page-by-page inside `app/extraction.py`: `extract_text` walks pages, tracks a running character total (including the 2-char `"\n\n"` joins), and raises `TextTooLargeError` the moment it crosses `MAX_TEXT_CHARS`. The route catches that and returns 422 — same status code, same constant, no shared validator. Two enforcement sites for two genuinely different shapes (Pydantic-validated request body vs. a streaming extraction loop), one shared constant. Empty extraction is caught **after** the text-length cap (it can't fire if extraction was capped first) and raised as **400** (mirroring `/text`'s post-strip check at `app/routes/text.py:19-20`), so §11's 400 entry remains authoritative for empty-text errors.
- **Error shape.** Both errors use FastAPI's default `{"detail": "..."}` per §11 — no wrapping, no custom hierarchy.
- **Where the constants live.** `MAX_UPLOAD_BYTES = 25 * 1024 * 1024`, `MAX_TEXT_CHARS = 3_000_000`, and the `MaxText` type alias all live as module-level definitions in `app/limits.py`. Middleware imports `MAX_UPLOAD_BYTES`; `IngestTextRequest` imports `MaxText`; `app/extraction.py` imports `MAX_TEXT_CHARS` directly for its page-by-page running-total check. Not surfaced through `Settings` / env vars — these are tied to embedding economics and worker memory, not per-deployment knobs.

**Migration notes:** None — these are operational constants. If a real customer uploads a document that hits either cap, the cap moves (one edit in `app/limits.py`); the *structure* (file cap + text cap, two layers) is the load-bearing decision.
