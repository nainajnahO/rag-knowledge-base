from app.embeddings import per_text_token_counts, tokenize
from app.models import Chunk

# Per DECISIONS.md §3: recursive token-based splitting, 600 tokens / 15% overlap.
# Separators are Latin-script-biased; for any piece they fail to split (e.g.
# CJK text without spaces, or any contiguous block), the deepest-fallback
# `_token_slice` slices by token offset from Voyage's tokenizer so chunks are
# still correctly sized in any language.
SEPARATORS = ["\n\n", "\n", ". ", " "]
MAX_TOKENS = 600
OVERLAP_TOKENS = 90


def chunk_text(text: str) -> list[Chunk]:
    if not text:
        return []

    pieces = _split_recursive([text], sep_idx=0)
    if not pieces:
        return []

    piece_counts = per_text_token_counts(pieces)
    windows = _pack_with_overlap(pieces, piece_counts)

    # Re-tokenize final joined chunks for exact persisted token_count.
    # The token-offset fallback guarantees no piece exceeds MAX_TOKENS, so
    # the previous Voyage-truncation cap on token_count is no longer needed.
    final_counts = per_text_token_counts(windows)
    return [
        Chunk(text=w, ordinal=i, token_count=c)
        for i, (w, c) in enumerate(zip(windows, final_counts))
    ]


def _split_recursive(texts: list[str], sep_idx: int) -> list[str]:
    if not texts:
        return []

    counts = per_text_token_counts(texts)
    result: list[str] = []

    for text, count in zip(texts, counts):
        # Recurse to OVERLAP_TOKENS, not MAX_TOKENS, so the packer has
        # fine-grained pieces to carry forward as overlap. Pieces larger
        # than OVERLAP_TOKENS would otherwise produce zero-overlap chunk
        # boundaries on long paragraphs.
        if count <= OVERLAP_TOKENS:
            result.append(text)
        elif sep_idx < len(SEPARATORS):
            sub_pieces = _split_keeping_sep(text, SEPARATORS[sep_idx])
            sub_pieces = [p for p in sub_pieces if p.strip()]
            result.extend(_split_recursive(sub_pieces, sep_idx + 1))
        else:
            # No semantic separator applies (e.g. CJK text without spaces, or
            # any contiguous block). Slice on token offset into small pieces so
            # the packer handles them uniformly — produces correct sizing AND
            # overlap in any language.
            result.extend(_token_slice(text, OVERLAP_TOKENS))

    return result


def _token_slice(text: str, piece_tokens: int) -> list[str]:
    """Slice `text` into pieces of at most `piece_tokens` by token-character offset.

    Uses Voyage's tokenizer to find each token's character span, then walks the
    spans in `piece_tokens`-sized strides, taking `text[char_start:char_end]`
    for each stride. Pieces are non-overlapping char slices, so the packer
    later layers MAX_TOKENS-windowing and OVERLAP_TOKENS-overlap on top
    uniformly with separator-derived pieces.

    Language-agnostic: Voyage's BPE tokenizer is unicode-aware and produces
    character-indexed offsets covering all input. Verified empirically for
    English, Japanese, Arabic, Hebrew, Thai, mixed scripts, and emoji-heavy
    text — every case yielded 100% offset coverage and exact reconstruction
    via "".join.
    """
    encoding = tokenize([text])[0]
    offsets = encoding.offsets
    if len(offsets) <= piece_tokens:
        return [text]

    pieces: list[str] = []
    for start in range(0, len(offsets), piece_tokens):
        end = min(start + piece_tokens, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        pieces.append(text[char_start:char_end])
    return pieces


def _split_keeping_sep(text: str, sep: str) -> list[str]:
    parts = text.split(sep)
    return [p + sep for p in parts[:-1]] + [parts[-1]]


def _pack_with_overlap(pieces: list[str], piece_counts: list[int]) -> list[str]:
    windows: list[str] = []
    cur_pieces: list[str] = []
    cur_counts: list[int] = []

    for piece, count in zip(pieces, piece_counts):
        if cur_pieces and sum(cur_counts) + count > MAX_TOKENS:
            windows.append("".join(cur_pieces))

            tail_pieces: list[str] = []
            tail_counts: list[int] = []
            running = 0
            for p, c in zip(reversed(cur_pieces), reversed(cur_counts)):
                if running + c > OVERLAP_TOKENS:
                    break
                tail_pieces.insert(0, p)
                tail_counts.insert(0, c)
                running += c
            cur_pieces = tail_pieces
            cur_counts = tail_counts

        cur_pieces.append(piece)
        cur_counts.append(count)

    if cur_pieces:
        windows.append("".join(cur_pieces))

    return windows
