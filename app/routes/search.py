"""GET /search — hybrid retrieval + rerank (DECISIONS.md §7 / §7.2)."""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field, field_validator

from app.db import ConnDep
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


@router.get("/search", response_model=SearchResponse)
def search(
    conn: ConnDep,
    params: Annotated[SearchParams, Query()],
    metadata: Annotated[dict[str, str], Depends(parse_meta_filters)],
) -> SearchResponse:
    filters = Filters(
        author=params.author,
        published_after=params.published_after,
        published_before=params.published_before,
        metadata=metadata,
    )
    candidates = retrieve(conn, params.q, k=RRF_CANDIDATES_PER_LANE, filters=filters)
    results = rerank(params.q, candidates, top_k=params.k)
    return SearchResponse(results=results)
