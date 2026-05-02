"""GET /search — dense vector retrieval with metadata filters (DECISIONS.md §6).

Step 7 (hybrid search) will replace the call into `retrieval.retrieve` with a
hybrid variant; the route shape stays the same.
"""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field, field_validator

from app.db import ConnDep
from app.models import SearchResponse
from app.retrieval import Filters, retrieve

router = APIRouter()


class SearchParams(BaseModel):
    """Bound from query string by FastAPI's Annotated[Model, Query()]."""

    q: str = Field(min_length=1, description="Search query")
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
    results = retrieve(conn, params.q, params.k, filters)
    return SearchResponse(results=results)
