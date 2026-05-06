"""GET /search — hybrid retrieval + rerank (DECISIONS.md §7 / §7.2)."""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field, field_validator

from app.db import ConnDep
from app.graph import resolve_entity_names
from app.models import SearchResponse
from app.rerank import rerank
from app.retrieval import RRF_CANDIDATES_PER_LANE, Filters, retrieve

router = APIRouter()


class SearchParams(BaseModel):
    """Bound from query string by FastAPI's Annotated[Model, Query()]."""

    # max_length=4096 short-circuits absurd queries before they reach Voyage
    # (the 25 MB body cap doesn't apply to GET URLs, and a meaningful
    # retrieval query is rarely longer than a paragraph).
    q: str = Field(min_length=1, max_length=4096, description="Search query")
    k: int = Field(default=10, ge=1, le=50)
    author: str | None = None
    published_after: date | None = None
    published_before: date | None = None

    @field_validator("q")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


def parse_meta_filters(request: Request) -> dict[str, str]:
    """Pull `meta.<key>=<value>` query params into a flat dict (single-value-per-key).

    Multiple values for the same key collapse to the last; documented as a
    single-value-per-key API. Numeric/boolean coercion is intentionally not
    attempted — query params are strings, JSONB values may not be.
    """
    return {k[5:]: v for k, v in request.query_params.items() if k.startswith("meta.")}


# A UUID guaranteed not to match any chunk_entity_mentions row. Appended
# to the resolved list when a user-passed entity name fails to resolve, so
# the EXISTS + HAVING COUNT(DISTINCT) = n_entities gate in retrieval.py
# yields empty results — matching the DECISIONS.md §18.7 contract that
# "an unknown name yields no chunks." gen_random_uuid() never produces
# all-zeros, so this is collision-free.
_UNRESOLVABLE_ENTITY_SENTINEL = "00000000-0000-0000-0000-000000000000"


def parse_entity_filter(
    conn: ConnDep,
    entity: Annotated[
        list[str],
        Query(description="Entity names — repeat for multiple (chunk-level AND)."),
    ] = [],
) -> list[str]:
    """Resolve repeated ?entity=<name> query params to entity UUIDs (str form).

    Case-insensitive alias lookup via app.graph.resolve_entity_names. When
    every passed name resolves, returns the deduped UUID list and retrieval
    runs the standard chunk-level AND filter. When *any* passed name fails
    to match an alias, the filter is unsatisfiable (no chunk can mention an
    entity that doesn't exist) so this function appends an
    `_UNRESOLVABLE_ENTITY_SENTINEL` UUID — the EXISTS+HAVING gate then
    yields empty results.

    /chat takes a different path (DECISIONS.md §18.9): unresolved names
    silently drop and the route falls back to plain hybrid retrieval rather
    than emptying. /chat calls `resolve_entity_names` directly, not this.
    """
    if not entity:
        return []

    resolved = resolve_entity_names(conn, entity)

    # Per-name probe to detect any name that failed to resolve. Comparing
    # len(resolved) to len(entity) isn't reliable: aliases of the same
    # canonical entity collapse to one UUID, so `[Acme, Acme Corp]` (both
    # aliases of one entity) returns one UUID even though both names
    # resolved successfully.
    if any(not resolve_entity_names(conn, [name]) for name in entity):
        resolved = [*resolved, _UNRESOLVABLE_ENTITY_SENTINEL]

    return resolved


@router.get("/search", response_model=SearchResponse)
def search(
    conn: ConnDep,
    params: Annotated[SearchParams, Query()],
    metadata: Annotated[dict[str, str], Depends(parse_meta_filters)],
    entity_ids: Annotated[list[str], Depends(parse_entity_filter)],
) -> SearchResponse:
    filters = Filters(
        author=params.author,
        published_after=params.published_after,
        published_before=params.published_before,
        metadata=metadata,
        entity_ids=entity_ids,
    )
    candidates = retrieve(conn, params.q, k=RRF_CANDIDATES_PER_LANE, filters=filters)
    results = rerank(params.q, candidates, top_k=params.k)
    return SearchResponse(results=results)
