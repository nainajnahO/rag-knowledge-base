"""Hand-crafted corpus + golden questions for the graph-vs-baseline eval.

Five short text documents share a small entity universe (one organization
expanding into Berlin and Tokyo, one partner organization with its own
location and founder, and a person who appears in two unrelated contexts).
The universe is constructed so several questions are answerable correctly
only when chunk-level co-mention narrows the candidate pool — which is
the brief's literal "X together with Y" example shape.

All document titles use the EVAL/ prefix so the runner can clean up its
own data without touching other ingested content. Each question carries
explicit pass criteria: case-insensitive substrings the answer must
contain, and document titles that must appear in the response sources.
"""

from typing import TypedDict


class Document(TypedDict):
    title: str
    text: str


class Question(TypedDict):
    question: str
    expected_substrings: list[str]
    expected_document_titles: list[str]


CORPUS: list[Document] = [
    {
        "title": "EVAL/Quintara Berlin office",
        "text": (
            "Quintara Industries opened its Berlin headquarters on March 15, 2026. "
            "CEO Maria Garcia announced that the Berlin office would lead European "
            "operations and employ 200 engineers."
        ),
    },
    {
        "title": "EVAL/Quintara Tokyo partnership",
        "text": (
            "Quintara Industries signed a partnership with Tanaka Holdings in Tokyo "
            "to coordinate Asia-Pacific supply chains. The agreement spans five years "
            "and covers twelve markets."
        ),
    },
    {
        "title": "EVAL/Garcia TechSummit keynote",
        "text": (
            "Maria Garcia delivered the opening keynote at TechSummit 2026 in San "
            "Francisco on April 8, 2026. Her talk focused on autonomous logistics "
            "platforms and predictive routing."
        ),
    },
    {
        "title": "EVAL/Quintara Q1 earnings",
        "text": (
            "Quintara Industries reported Q1 2026 revenue of $4.2 billion, a 23% "
            "year-over-year increase. The company attributed growth to new "
            "infrastructure contracts."
        ),
    },
    {
        "title": "EVAL/Tanaka Holdings background",
        "text": (
            "Tanaka Holdings, founded in 1962 by Hiroshi Tanaka, operates 47 "
            "manufacturing facilities across Japan. The company is headquartered "
            "in Tokyo."
        ),
    },
]

QUESTIONS: list[Question] = [
    # Single-entity sanity — both modes should pass.
    {
        "question": "What did Quintara Industries report in Q1 2026?",
        "expected_substrings": ["4.2 billion"],
        "expected_document_titles": ["EVAL/Quintara Q1 earnings"],
    },
    # Co-mention narrowing: Quintara appears in 3 docs (Berlin, Tokyo, earnings);
    # Berlin appears only in doc 1. Graph-on AND filter narrows to that one.
    {
        "question": "What did Quintara Industries announce in Berlin?",
        "expected_substrings": ["headquarters"],
        "expected_document_titles": ["EVAL/Quintara Berlin office"],
    },
    # Garcia + Berlin AND filter — Garcia is in doc 1 (Berlin) and doc 3
    # (TechSummit/SF); only doc 1 mentions both.
    {
        "question": "What did Maria Garcia announce in Berlin?",
        "expected_substrings": ["200 engineers"],
        "expected_document_titles": ["EVAL/Quintara Berlin office"],
    },
    # Garcia + TechSummit AND filter — only doc 3.
    {
        "question": "What did Maria Garcia present at TechSummit 2026?",
        "expected_substrings": ["autonomous logistics"],
        "expected_document_titles": ["EVAL/Garcia TechSummit keynote"],
    },
    # Single-entity sanity — both modes should pass.
    {
        "question": "Where is Tanaka Holdings headquartered?",
        "expected_substrings": ["Tokyo"],
        "expected_document_titles": ["EVAL/Tanaka Holdings background"],
    },
    # Quintara + Tanaka AND filter — Tanaka is in docs 2 and 5; only doc 2
    # mentions both, and only doc 2 has the partnership detail.
    {
        "question": (
            "What is the size of the partnership between Quintara Industries "
            "and Tanaka Holdings?"
        ),
        "expected_substrings": ["five years"],
        "expected_document_titles": ["EVAL/Quintara Tokyo partnership"],
    },
    # Single-entity sanity — both modes should pass.
    {
        "question": "Who founded Tanaka Holdings?",
        "expected_substrings": ["Hiroshi Tanaka"],
        "expected_document_titles": ["EVAL/Tanaka Holdings background"],
    },
]
