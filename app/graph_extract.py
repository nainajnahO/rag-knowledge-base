"""Per-chunk entity + relation extraction via Haiku structured outputs.

Calls `client.beta.messages.parse(...)` with `ExtractedGraph` as
`output_format`. The system block carries `cache_control={"type": "ephemeral"}`
so the fixed extraction prompt is a cache hit across the per-chunk loop;
only the chunk text is a cache miss per call.

DECISIONS.md §KG #4: per-chunk granularity is the GraphRAG / LightRAG /
LlamaIndex / Neo4j industry standard. The Anthropic cookbook's per-document
shape was designed for short Wikipedia summaries (~1.5K tokens); per-chunk
fits multi-page PDFs without long-context attention drift.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import anthropic
from fastapi import HTTPException

from app.llm import get_client
from app.models import Chunk, ExtractedGraph
from app.settings import settings

# Verbatim from Anthropic's "Knowledge graph construction with Claude"
# cookbook (platform.claude.com/cookbook/capabilities-knowledge-graph-guide),
# with ARTIFACT removed (DECISIONS.md §KG #9 — noisy on memo/report content
# due to self-references and weak within-type coherence).
EXTRACTION_SYSTEM = (
    "Extract a knowledge graph from the document below.\n\n"
    "Guidelines:\n"
    "- Extract only entities that are central to what this document is about "
    "— skip incidental mentions.\n"
    "- For each entity, write a one-sentence description grounded in this "
    "document. These descriptions are used later to disambiguate entities "
    "with similar names.\n"
    "- Predicates should be short verb phrases (\"commanded\", \"launched "
    "from\", \"part of\").\n"
    "- Every relation must connect two entities you extracted.\n"
    "- Entity types: PERSON, ORGANIZATION, LOCATION, EVENT only."
)

EXTRACTION_USER_TMPL = "<document>\n{text}\n</document>"


@contextmanager
def map_graph_extract_errors() -> Iterator[None]:
    """Translate Anthropic SDK errors to HTTPException — same taxonomy as map_anthropic_errors."""
    try:
        yield
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
        raise HTTPException(status_code=500, detail=f"graph extraction auth failure: {exc}") from exc
    except anthropic.BadRequestError as exc:
        raise HTTPException(status_code=400, detail=f"graph extraction rejected input: {exc}") from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"graph extraction rate limited: {exc}") from exc
    except anthropic.APIError as exc:
        raise HTTPException(status_code=503, detail=f"graph extraction upstream failure: {exc}") from exc


def extract_graph_from_chunk(chunk: Chunk) -> ExtractedGraph:
    """Extract entities + relations from one chunk via Haiku structured outputs.

    The system block is prompt-cached so per-chunk loops over a single
    document only pay full price for the chunk text. `parsed_output` is
    Optional[T] — None when the response had no text block — so we fall
    back to an empty graph rather than raise (the cookbook's recommended
    defensive shape).
    """
    with map_graph_extract_errors():
        response = get_client().beta.messages.parse(
            model=settings.extraction_model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": EXTRACTION_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_USER_TMPL.format(text=chunk.text),
                }
            ],
            output_format=ExtractedGraph,
        )
    return response.parsed_output or ExtractedGraph(entities=[], relations=[])


def extract_graphs_from_chunks(chunks: list[Chunk]) -> list[ExtractedGraph]:
    """Extract one graph per chunk, in order. Empty input returns empty output."""
    return [extract_graph_from_chunk(c) for c in chunks]


def extract_entities_from_question(question: str) -> list[str]:
    """Extract entity names from a chat question for graph pre-filtering.

    Reuses the Haiku extraction shape (same prompt, same parsed output type)
    and returns just the raw entity name strings — UUIDs come later via
    `app.graph.resolve_entity_names`. This is best-effort: any failure
    falls through to an empty list so the caller can degrade to plain
    retrieval rather than refuse the request (DECISIONS.md §KG #14).
    """
    try:
        with map_graph_extract_errors():
            response = get_client().beta.messages.parse(
                model=settings.extraction_model,
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": EXTRACTION_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": EXTRACTION_USER_TMPL.format(text=question),
                    }
                ],
                output_format=ExtractedGraph,
            )
    except Exception:
        # Graph entity extraction is non-critical for /chat. Any failure
        # — auth, rate limit, network — falls through to plain retrieval.
        return []

    graph = response.parsed_output
    return [e.name for e in graph.entities] if graph else []
