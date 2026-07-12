"""Adaptive chunker: real-PDF invariants + fast unit tests for the helpers."""

from pathlib import Path

import pytest

from app.rag.chunk import (
    _MIN_CHUNK_CHARS,
    _extract_tickers,
    _split_into_sections,
    _split_long,
    chunk_document,
    parent_documents,
)
from app.rag.extract import Line

REVIEWS = Path(__file__).resolve().parent.parent / "data" / "reviews"
DAILY = REVIEWS / "סקירת דסק יומית מידעפנים 08_07_26.pdf"
WEEKLY = REVIEWS / "סקירת_דסק_שבועית_060726_מידע_פנים.pdf"

MAX_CHARS = 1200
BOILERPLATE = ["מסמך לימודי בלבד", "לשימוש פנימי של הדסק"]


@pytest.mark.parametrize(
    "path, doc_type, review_date, pages",
    [
        (DAILY, "daily", "2026-07-08", 6),
        (WEEKLY, "weekly", "2026-07-06", 7),
    ],
)
def test_chunk_invariants(path, doc_type, review_date, pages):
    chunks = chunk_document(str(path))
    assert chunks

    for c in chunks:
        assert c.text.strip()
        # Guardrails: within budget (a merged fragment may add up to the floor).
        assert _MIN_CHUNK_CHARS <= len(c.text) <= MAX_CHARS + _MIN_CHUNK_CHARS
        assert c.doc_type == doc_type
        assert c.review_date == review_date
        assert c.source == path.name
        assert c.chunk_id and c.parent_id  # stable ids for RRF / parent recovery
        assert c.pages and all(1 <= p <= pages for p in c.pages)
        assert all(phrase not in c.text for phrase in BOILERPLATE)  # no furniture
        if c.section:  # heading is injected into the searchable text
            assert c.section in c.text

    # Most content lands under a detected section heading.
    with_section = sum(1 for c in chunks if c.section)
    assert with_section >= 0.8 * len(chunks)


@pytest.mark.parametrize("path", [DAILY, WEEKLY])
def test_parents_cover_children_and_carry_heading(path):
    parents = parent_documents(str(path))
    children = chunk_document(str(path))
    parent_ids = {p.parent_id for p in parents}

    # Every child links back to a real parent section.
    assert all(c.parent_id in parent_ids for c in children)
    # Each parent's heading is in its own searchable text.
    assert all(p.section in p.text for p in parents if p.section)


def test_ticker_whitelist_filters_candidates():
    whitelist = {"MU", "VIX", "AI"}
    chunks = chunk_document(str(WEEKLY), ticker_whitelist=whitelist)
    found = {t for c in chunks for t in c.tickers}
    assert found <= whitelist
    assert "MU" in found  # Micron is discussed in the weekly


def _line(text, size, page=1):
    return Line(text=text, size=size, bold=size >= 12, page=page)


def test_split_into_sections_groups_under_headings():
    lines = [
        _line("שורת הדסק", 13),
        _line("גוף ראשון", 8),
        _line("גוף שני", 8),
        _line("Conviction Map", 13),
        _line("גוף שלישי", 8),
    ]
    sections = _split_into_sections(lines, threshold=10.0)
    assert [s.heading for s in sections] == ["שורת הדסק", "Conviction Map"]
    assert [len(s.lines) for s in sections] == [2, 1]


def test_consecutive_heading_lines_merge_into_one_section():
    # A wrapped heading arrives as several stacked heading-sized lines; they
    # must form one heading, not empty sections whose text is dropped.
    lines = [
        _line("TSMC PIC/CPO", 10),
        _line("- צוואר בקבוק חדש", 10),
        _line("אופטית", 10),
        _line("גוף הפסקה הראשונה", 8),
        _line("המשך גוף", 8),
    ]
    sections = _split_into_sections(lines, threshold=10.0)
    assert len(sections) == 1
    assert sections[0].heading == "TSMC PIC/CPO - צוואר בקבוק חדש אופטית"
    assert len(sections[0].lines) == 2


def test_split_long_breaks_at_whitespace_within_budget():
    text = " ".join(["word"] * 500)  # ~2500 chars
    pieces = _split_long(text, 1200)
    assert len(pieces) > 1
    assert all(len(p) <= 1200 for p in pieces)
    assert " ".join(pieces).split() == text.split()  # no content lost


def test_extract_tickers_respects_whitelist():
    text = "MU and VIX up, AAOI flat, notagood ONE"
    assert _extract_tickers(text, {"MU", "VIX"}) == ("MU", "VIX")
    # No whitelist -> all upper-case runs are candidates.
    assert "AAOI" in _extract_tickers(text, None)
