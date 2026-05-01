from datetime import date
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    text: str
    ordinal: int
    token_count: int


class IngestTextRequest(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=3_000_000)
    author: str | None = None
    published_date: date | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestTextResponse(BaseModel):
    document_id: UUID
    n_chunks: int
