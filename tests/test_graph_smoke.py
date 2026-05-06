"""Test 4 — knowledge-graph extraction + entity-filtered search smoke.

Verifies the full ingest → extract → resolve → persist → entity-filtered
retrieve loop:
  1. Two documents that share an organization but mention different cities
     are ingested. Extraction + resolution run synchronously per ingest.
  2. ?entity=<org> on /search surfaces chunks from both docs (single filter).
  3. ?entity=<org>&entity=<unique-city> AND-filters to chunks from only the
     document that mentions both. The other doc must be absent.
  4. ?entity=<unknown-name> returns empty (under chunk-level AND, an
     unresolvable name yields no chunks — DECISIONS.md §KG #14 surface).

Note: entities are corpus-global (not document-scoped, see DECISIONS.md
§KG limitations) so cross-test pollution is possible; assertions check
*absence* of the wrong document rather than position-1 of the right one,
which is robust to whatever else is in the entities table.
"""

from uuid import uuid4

from fastapi.testclient import TestClient

from .conftest import requires_anthropic, requires_voyage


@requires_voyage
@requires_anthropic
def test_entity_extraction_and_filtered_search(
    client: TestClient,
    db_cleanup: None,
) -> None:
    salt = uuid4().hex[:8]
    org = f"Zorbnak-{salt}"
    city_a = f"Belgrade-{salt}"
    city_b = f"Antwerp-{salt}"

    # Doc A — mentions the org and city_a, not city_b
    upload_a = client.post(
        "/text",
        json={
            "title": f"{org} {city_a} expansion notes",
            "text": (
                f"{org} opened its new {city_a} headquarters in Q1 2026. "
                f"The {city_a} office will lead European operations and employ 200 engineers."
            ),
        },
    )
    assert upload_a.status_code == 200, upload_a.text
    doc_a_id = upload_a.json()["document_id"]

    # Doc B — mentions the org and city_b, not city_a
    upload_b = client.post(
        "/text",
        json={
            "title": f"{org} {city_b} partnership notes",
            "text": (
                f"{org} signed a partnership with Tanaka Industries in {city_b} "
                f"to coordinate Asia-Pacific supply chains."
            ),
        },
    )
    assert upload_b.status_code == 200, upload_b.text
    doc_b_id = upload_b.json()["document_id"]

    # Single-entity filter: both docs surface (each mentions the org).
    r_single = client.get("/search", params={"q": org, "entity": org})
    assert r_single.status_code == 200, r_single.text
    single_doc_ids = {res["document_id"] for res in r_single.json()["results"]}
    assert doc_a_id in single_doc_ids, "?entity=<org> should surface doc A"
    assert doc_b_id in single_doc_ids, "?entity=<org> should surface doc B"

    # Two-entity AND filter: only doc A's chunk co-mentions <org> + city_a.
    r_and = client.get(
        "/search",
        params={"q": f"{org} {city_a}", "entity": [org, city_a]},
    )
    assert r_and.status_code == 200, r_and.text
    and_doc_ids = {res["document_id"] for res in r_and.json()["results"]}
    assert doc_b_id not in and_doc_ids, (
        f"doc B (no {city_a} mention) leaked into ?entity={org}&entity={city_a} results"
    )


@requires_voyage
@requires_anthropic
def test_entity_filter_unknown_name_returns_empty(
    client: TestClient,
    db_cleanup: None,
) -> None:
    """An ?entity= with no matching alias yields zero results under chunk-level AND."""
    r = client.get(
        "/search",
        params={"q": "anything", "entity": f"NonexistentEntity-{uuid4().hex}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["results"] == [], "expected empty results for unknown entity name"
