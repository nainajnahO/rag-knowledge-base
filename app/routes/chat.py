"""POST /chat — RAG with structured Anthropic citations (DECISIONS.md §8 Path B)."""

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from app.db import ConnDep
from app.llm import CHAT_SCORE_THRESHOLD, generate_answer
from app.models import (
    AnswerBlock,
    ChatResponse,
    ChatSource,
    CitationRef,
    RetrievedChunk,
)
from app.retrieval import Filters, retrieve

router = APIRouter()

# DECISIONS.md §8 — top-K retrieval for chat.
CHAT_TOP_K = 8

# Refusal text mirrors the SYSTEM_PROMPT instruction so the
# threshold-gate path produces the same string the model would emit
# under guardrail #2.
REFUSAL_TEXT = "I don't have enough information in the provided sources to answer this."


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4096)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, conn: ConnDep) -> ChatResponse:
    chunks = retrieve(conn, req.question, k=CHAT_TOP_K, filters=Filters())

    if not chunks or chunks[0].score < CHAT_SCORE_THRESHOLD:
        return _refusal_response()

    message = generate_answer(req.question, chunks)
    return _build_response(message, chunks)


def _refusal_response() -> ChatResponse:
    return ChatResponse(
        answer=REFUSAL_TEXT,
        answer_blocks=[AnswerBlock(text=REFUSAL_TEXT, citations=[])],
        sources=[],
        stop_reason=None,
    )


def _build_response(
    message, retrieved: list[RetrievedChunk]
) -> ChatResponse:
    """Map Anthropic Message → our ChatResponse.

    Preserves the native answer_blocks shape; flattens to `answer` for
    simple display; annotates each retrieved chunk with cited/cited_text.
    """
    chunk_by_id = {str(c.chunk_id): c for c in retrieved}
    answer_blocks: list[AnswerBlock] = []
    citations_by_chunk: dict[str, list[str]] = {}

    for block in message.content:
        if block.type != "text":
            continue
        refs: list[CitationRef] = []
        for cite in block.citations or []:
            if cite.type != "search_result_location":
                continue
            # Defensive: Anthropic's contract says `source` echoes a value we
            # passed in, but if it ever doesn't, drop the citation rather than
            # 500 the request.
            chunk = chunk_by_id.get(cite.source)
            if chunk is None:
                continue
            refs.append(
                CitationRef(
                    chunk_id=chunk.chunk_id,
                    document_title=chunk.document_title,
                    published_date=chunk.published_date,
                    cited_text=cite.cited_text,
                )
            )
            citations_by_chunk.setdefault(cite.source, []).append(cite.cited_text)
        answer_blocks.append(AnswerBlock(text=block.text, citations=refs))

    answer = "".join(b.text for b in answer_blocks)
    sources = [
        ChatSource(
            **chunk.model_dump(),
            cited=str(chunk.chunk_id) in citations_by_chunk,
            cited_text=citations_by_chunk.get(str(chunk.chunk_id), []),
        )
        for chunk in retrieved
    ]
    return ChatResponse(
        answer=answer,
        answer_blocks=answer_blocks,
        sources=sources,
        stop_reason=message.stop_reason,
    )
