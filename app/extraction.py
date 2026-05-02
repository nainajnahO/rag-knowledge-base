"""PDF text extraction (DECISIONS.md §5).

One function: `extract_text(pdf_bytes) -> str`. Walks every page, joins them
with paragraph breaks (`\\n\\n`) so the chunker's top-level separator stays
meaningful. Pure: takes bytes, returns text. The route owns HTTP-status
mapping and the magic-byte pre-check.

Migration story: swapping pymupdf → pypdf is a same-day replacement of this
file's body — the function signature stays.
"""

import pymupdf


def extract_text(pdf_bytes: bytes) -> str:
    """Extract concatenated plain text from every page of a PDF.

    Raises `pymupdf.FileDataError` if the PDF is corrupt or malformed; the
    route catches this and maps to HTTP 400. The caller is responsible for
    the `%PDF-` magic-byte check before invoking this function.
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n\n".join(page.get_text() for page in doc)
