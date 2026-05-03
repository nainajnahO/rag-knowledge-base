from semantic_text_splitter import TextSplitter

from app.embeddings import per_text_token_counts
from app.models import Chunk

# Per DECISIONS.md §3: token-based splitting via `semantic-text-splitter`,
# 600 tokens / 15% overlap (~90 tokens). Token counts route through Voyage's
# tokenizer (the same one used at embed time), so chunk sizes match exactly
# what the embedding API will see.
MAX_TOKENS = 600
OVERLAP_TOKENS = 90


def _token_count(text: str) -> int:
    return per_text_token_counts([text])[0]


_splitter = TextSplitter.from_callback(
    _token_count, capacity=MAX_TOKENS, overlap=OVERLAP_TOKENS
)


def chunk_text(text: str) -> list[Chunk]:
    if not text:
        return []

    pieces = _splitter.chunks(text)
    if not pieces:
        return []

    counts = per_text_token_counts(pieces)
    return [
        Chunk(text=p, ordinal=i, token_count=c)
        for i, (p, c) in enumerate(zip(pieces, counts))
    ]
