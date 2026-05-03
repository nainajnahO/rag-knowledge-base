"""Test 3 of §13 — citation/source consistency.

Updated from §13's original `[N]`-style description to match §8's structured
citations: every citation in answer_blocks[*].citations must resolve to a
chunk_id present in `sources`, and each cited_text must be a verbatim
substring of that source's text.
"""

from fastapi.testclient import TestClient

from .conftest import requires_anthropic, requires_voyage


@requires_voyage
@requires_anthropic
def test_chat_citations_resolve_to_sources_with_verbatim_text(
    client: TestClient,
    db_cleanup: None,
) -> None:
    payload = {
        "title": "Acme widget revenue memo",
        "text": (
            "In Q3 2025, Acme Corporation reported total widget revenue of "
            "$4.2 million, a 12% increase year-over-year. The growth was "
            "driven by enterprise customers in the manufacturing sector. "
            "Acme expects continued growth into Q4 2025."
        ),
        "author": "Acme Finance",
    }
    upload = client.post("/text", json=payload)
    assert upload.status_code == 200, upload.text

    response = client.post("/chat", json={"question": "What was Acme's Q3 2025 widget revenue?"})
    assert response.status_code == 200, response.text
    body = response.json()

    sources_by_id = {s["chunk_id"]: s for s in body["sources"]}
    assert sources_by_id, "expected at least one retrieved source"

    cited_chunk_ids = {
        cite["chunk_id"]
        for block in body["answer_blocks"]
        for cite in block["citations"]
    }

    if not cited_chunk_ids:
        # Refusal or uncited synthesis is allowed by the API contract; skip
        # the substring check rather than fail. The `sources_by_id` assert
        # above already exercises the retrieval path.
        return

    for block in body["answer_blocks"]:
        for cite in block["citations"]:
            assert cite["chunk_id"] in sources_by_id, (
                f"citation references chunk_id {cite['chunk_id']} not in sources"
            )
            source = sources_by_id[cite["chunk_id"]]
            assert cite["cited_text"] in source["text"], (
                "cited_text is not a verbatim substring of the source chunk"
            )
