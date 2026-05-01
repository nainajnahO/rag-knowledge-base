from functools import cache

import voyageai
from tokenizers import Encoding

from app.models import Chunk
from app.settings import settings

# Voyage-4 per-request caps (docs.voyageai.com/docs/embeddings).
MAX_INPUTS_PER_REQUEST = 1000
MAX_TOKENS_PER_REQUEST = 320_000


@cache
def _get_client() -> voyageai.Client:
    return voyageai.Client(api_key=settings.voyage_api_key or None)


def embed_chunks(chunks: list[Chunk], input_type: str = "document") -> list[list[float]]:
    """Embed chunks for one document.

    Sub-batches internally to stay under voyage-4's 1000-input / 320K-token
    per-request caps. Returns embeddings in input order. Voyage errors are
    not caught here; they propagate to the route as 503.
    """
    if not chunks:
        return []

    embeddings: list[list[float]] = []
    batch_texts: list[str] = []
    batch_tokens = 0

    def flush() -> None:
        nonlocal batch_texts, batch_tokens
        if not batch_texts:
            return
        result = _get_client().embed(batch_texts, model=settings.embedding_model, input_type=input_type)
        embeddings.extend(result.embeddings)
        batch_texts = []
        batch_tokens = 0

    for chunk in chunks:
        # Flush before adding if this chunk would exceed either cap.
        would_exceed_inputs = len(batch_texts) + 1 > MAX_INPUTS_PER_REQUEST
        would_exceed_tokens = batch_tokens + chunk.token_count > MAX_TOKENS_PER_REQUEST
        if batch_texts and (would_exceed_inputs or would_exceed_tokens):
            flush()
        batch_texts.append(chunk.text)
        batch_tokens += chunk.token_count

    flush()
    return embeddings


def count_tokens(texts: list[str]) -> int:
    return _get_client().count_tokens(texts, model=settings.embedding_model)


def per_text_token_counts(texts: list[str]) -> list[int]:
    if not texts:
        return []
    return [len(e.tokens) for e in tokenize(texts)]


def tokenize(texts: list[str]) -> list[Encoding]:
    """Return Voyage's HuggingFace `Encoding` objects for each text.

    Each encoding exposes `.tokens`, `.ids`, and `.offsets` (a list of
    (char_start, char_end) tuples per token). The chunker uses `.offsets`
    to slice text by token boundary when no semantic separator applies.
    """
    return _get_client().tokenize(texts, model=settings.embedding_model)
