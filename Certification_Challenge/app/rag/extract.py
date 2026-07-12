"""PyMuPDF structural extraction for the Hebrew desk reviews (ADR-0004).

We keep PyMuPDF's structural map — per-span font size + weight — because the
adaptive chunker needs it to tell headings from body. Every span's text is run
through the deterministic RTL-repair pass on the way out, so downstream code
never sees the extraction artifacts.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import fitz  # pymupdf

from app.rag.rtl import repair_rtl

# pymupdf span "flags": bit 2**4 marks a synthetic/real bold face.
_BOLD_FLAG = 1 << 4

_DATE = re.compile(r"\b(\d{2})/(\d{2})/(\d{2})\b")  # DD/MM/YY in the review header


@dataclass(frozen=True)
class Span:
    """A single extracted text run with the layout signal the chunker uses."""

    text: str
    size: float
    font: str
    bold: bool
    page: int  # 1-indexed


@dataclass(frozen=True)
class Line:
    """A visual line: spans joined, with the size the chunker classifies on."""

    text: str
    size: float  # representative = largest span on the line (headings are uniform)
    bold: bool  # any bold span on the line
    page: int  # 1-indexed


_WS = re.compile(r"\s+")


def extract_spans(path: str) -> list[Span]:
    """Return every text span in reading order, RTL-repaired, across all pages."""
    spans: list[Span] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for s in line.get("spans", []):
                        font = s.get("font", "")
                        bold = bool(s.get("flags", 0) & _BOLD_FLAG) or "Bold" in font
                        spans.append(
                            Span(
                                text=repair_rtl(s["text"]),
                                size=round(s["size"], 1),
                                font=font,
                                bold=bold,
                                page=page_index,
                            )
                        )
    return spans


def extract_lines(path: str) -> list[Line]:
    """Return visual lines in reading order, spans joined and RTL-repaired.

    A line's representative ``size`` is the largest span size on it, so a
    heading line reads as a heading even if it carries a trailing small span.
    Blank lines are dropped.
    """
    lines: list[Line] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    raw_spans = line.get("spans", [])
                    if not raw_spans:
                        continue
                    text = _WS.sub(" ", "".join(repair_rtl(s["text"]) for s in raw_spans)).strip()
                    if not text:
                        continue
                    size = round(max(s["size"] for s in raw_spans), 1)
                    bold = any(
                        (s.get("flags", 0) & _BOLD_FLAG) or "Bold" in s.get("font", "")
                        for s in raw_spans
                    )
                    lines.append(Line(text=text, size=size, bold=bold, page=page_index))
    return lines


def modal_body_size(spans: list[Span]) -> float:
    """Char-weighted most-common font size — the document's body text size."""
    weights: Counter[float] = Counter()
    for s in spans:
        weights[s.size] += len(s.text.strip())
    return weights.most_common(1)[0][0]


def page_count(path: str) -> int:
    with fitz.open(path) as doc:
        return doc.page_count


def detect_doc_type(spans: list[Span]) -> str:
    """Classify the review as ``"daily"`` or ``"weekly"`` from its title text."""
    text = " ".join(s.text for s in spans)
    if "יומית" in text:
        return "daily"
    if "שבועית" in text:
        return "weekly"
    return "unknown"


def detect_review_date(spans: list[Span]) -> str | None:
    """Return the review date as an ISO ``YYYY-MM-DD`` string, if present.

    Dates in the header are DD/MM/YY (two-digit year in the 2000s).
    """
    for s in spans:
        m = _DATE.search(s.text)
        if m:
            dd, mm, yy = m.groups()
            return f"20{yy}-{mm}-{dd}"
    return None
