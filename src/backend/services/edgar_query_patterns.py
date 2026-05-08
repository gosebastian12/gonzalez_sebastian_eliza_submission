"""Regex patterns and metadata-filter extraction for EDGAR filing queries."""

from __future__ import annotations

import re
from typing import Final

FORM_RE: Final = re.compile(r"\b10[- ]?([kq])\b", re.IGNORECASE)
QUARTER_RE: Final = re.compile(r"\b(20\d{2}Q[1-4])\b", re.IGNORECASE)
DATE_RE: Final = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
TICKER_RE: Final = re.compile(r"\b[A-Z]{1,5}\b")
YEAR_RE: Final = re.compile(r"\b(19|20)\d{2}\b")
FY_YEAR_RE: Final = re.compile(r"\bfy\s*[- ]?\s*((?:19|20)\d{2})\b", re.IGNORECASE)
FISCAL_YEAR_RE: Final = re.compile(
    r"\bfiscal(?:\s+year)?\s+(?:of\s+)?((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
FILENAME_ISO_DATE_RE: Final = re.compile(r"_(\d{4}-\d{2}-\d{2})_")


def extract_metadata_filters(query: str) -> dict[str, str]:
    """Build Chroma ``where`` filters from explicit cues in the user question."""
    filters: dict[str, str] = {}

    form = FORM_RE.search(query)
    if form:
        filters["form_type"] = f"10-{form.group(1).upper()}"

    quarter = QUARTER_RE.search(query)
    if quarter:
        filters["quarter"] = quarter.group(1).upper()

    date = DATE_RE.search(query)
    if date:
        filters["filing_date"] = date.group(1)

    ticker_candidates = [t for t in TICKER_RE.findall(query) if t not in {"RAG", "LLM", "SEC"}]
    if len(ticker_candidates) == 1:
        filters["ticker"] = ticker_candidates[0]

    return filters
