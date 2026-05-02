from datetime import date
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.limits import MaxText


class Chunk(BaseModel):
    text: str
    ordinal: int
    token_count: int


class IngestTextRequest(BaseModel):
    title: str = Field(min_length=1)
    text: MaxText
    author: str | None = None
    published_date: date | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestDocumentRequest(BaseModel):
    """Form-encoded request body for POST /document.

    Fields ride the multipart envelope as individual form slots (per
    DECISIONS.md, Q1 of Step 4 planning). `metadata` is a JSON-encoded
    string because multipart form fields can't carry nested objects;
    the route parses it via json.loads after Pydantic validation.
    The 3M-char text cap applies to the *extracted* text — re-validated
    in the route via TypeAdapter(MaxText), not declared on this model.
    """

    title: str = Field(min_length=1)
    author: str | None = None
    published_date: date | None = None
    metadata: str = "{}"


class IngestResponse(BaseModel):
    document_id: UUID
    n_chunks: int
