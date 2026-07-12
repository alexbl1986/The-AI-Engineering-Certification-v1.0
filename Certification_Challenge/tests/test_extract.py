"""Structural extraction over the real committed desk reviews.

These run offline against the two bundled PDFs and assert structural
properties (not brittle exact strings): correct page counts, doc-type/date
detection, and — critically — that the RTL-repair pass leaves no residual
artifact in the extracted spans.
"""

import re
from pathlib import Path

import pytest

from app.rag.extract import (
    Span,
    detect_doc_type,
    detect_review_date,
    extract_spans,
    page_count,
)

REVIEWS = Path(__file__).resolve().parent.parent / "data" / "reviews"
DAILY = REVIEWS / "סקירת דסק יומית מידעפנים 08_07_26.pdf"
WEEKLY = REVIEWS / "סקירת_דסק_שבועית_060726_מידע_פנים.pdf"

# The artifact is sentence punctuation flipped onto the FRONT of a Hebrew word
# (",זיכרון"); a lone "." span or an LTR fragment (".5%") is not an artifact.
_LEADING_PUNCT_ON_HEBREW = re.compile(r"^[.,:;!?][א-ת]")
_MAQAF = re.compile(r"-[א-ת][A-Za-z]")


@pytest.mark.parametrize("path, pages", [(DAILY, 6), (WEEKLY, 7)])
def test_extracts_all_pages(path, pages):
    assert page_count(str(path)) == pages
    spans = extract_spans(str(path))
    assert spans, "expected non-empty spans"
    assert all(isinstance(s, Span) for s in spans)
    assert {s.page for s in spans} == set(range(1, pages + 1))


@pytest.mark.parametrize("path", [DAILY, WEEKLY])
def test_no_residual_rtl_artifacts(path):
    spans = extract_spans(str(path))
    assert not [s.text for s in spans if _LEADING_PUNCT_ON_HEBREW.search(s.text)]
    assert not [s.text for s in spans if _MAQAF.search(s.text)]


def test_detects_doc_type_and_date():
    assert detect_doc_type(extract_spans(str(DAILY))) == "daily"
    assert detect_doc_type(extract_spans(str(WEEKLY))) == "weekly"
    assert detect_review_date(extract_spans(str(DAILY))) == "2026-07-08"
    assert detect_review_date(extract_spans(str(WEEKLY))) == "2026-07-06"


def test_detects_bold_headings():
    # The document title is a large bold span; make sure weight is captured.
    spans = extract_spans(str(DAILY))
    body = max(s.size for s in spans)
    assert any(s.bold for s in spans if s.size == body)
