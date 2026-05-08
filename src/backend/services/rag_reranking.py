"""Lexical/metadata re-ranking and recency tie-breaking for retrieved chunks."""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import Document

from edgar_query_patterns import (
    DATE_RE,
    FILENAME_ISO_DATE_RE,
    FISCAL_YEAR_RE,
    FY_YEAR_RE,
    QUARTER_RE,
    YEAR_RE,
)

METADATA_WORD_STOPWORDS = frozenset(
    """
    the a an and or but if to of for in on at by from with without into about over under
    than then so as is are was were be been being it its this that these those what which
    who whom how when where why can could should would will shall may might must tell give
    show list summarize summary describe explain compare latest recent filings filing sec
    edgar report annual quarterly company stock shares revenue income earnings cash debt risk
    factors financial financials business operations management discussion analysis please
    """.split()
)


def query_pins_specific_period(question: str) -> bool:
    """True when the user narrows time (year, quarter, fiscal year, or ISO date)."""
    if DATE_RE.search(question):
        return True
    if QUARTER_RE.search(question):
        return True
    if FY_YEAR_RE.search(question):
        return True
    if FISCAL_YEAR_RE.search(question):
        return True
    if YEAR_RE.search(question):
        return True
    return False


def query_content_keywords(question: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9.-]{2,}", question.lower())
    return [t for t in tokens if t not in METADATA_WORD_STOPWORDS]


def _quarter_end_epoch(year: int, quarter: int) -> float:
    month = quarter * 3
    last_day = monthrange(year, month)[1]
    dt = datetime(year, month, last_day, tzinfo=timezone.utc)
    return dt.timestamp()


def report_recency_epoch(metadata: dict[str, Any]) -> float:
    """Best-effort filing/report time for ordering (newer = larger)."""

    def _parse_iso_date(value: str) -> float | None:
        value = value.strip()[:10]
        try:
            dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None

    fd = str(metadata.get("filing_date") or "").strip()
    if fd:
        parsed = _parse_iso_date(fd)
        if parsed is not None:
            return parsed

    rp = str(metadata.get("report_period") or "").strip()
    if rp:
        parsed = _parse_iso_date(rp)
        if parsed is not None:
            return parsed

    quarter = str(metadata.get("quarter") or "").strip().upper().replace(" ", "")
    qm = re.match(r"^(20\d{2})Q([1-4])$", quarter)
    if qm:
        return _quarter_end_epoch(int(qm.group(1)), int(qm.group(2)))

    file_name = str(metadata.get("file_name") or "")
    dates = FILENAME_ISO_DATE_RE.findall(file_name)
    epochs: list[float] = []
    for entry in dates:
        parsed = _parse_iso_date(entry)
        if parsed is not None:
            epochs.append(parsed)
    if epochs:
        return max(epochs)

    return 0.0


def keyword_metadata_alignment_score(doc: Document, question: str) -> float:
    """Lexical alignment of chunk metadata (+ light body hits) with the user question."""
    meta = doc.metadata
    keywords = query_content_keywords(question)
    hay_parts = [
        str(meta.get(key, ""))
        for key in (
            "company",
            "ticker",
            "form_type",
            "quarter",
            "report_period",
            "filing_date",
            "section_title",
            "file_name",
            "chunk_type",
        )
    ]
    haystack = " ".join(hay_parts).lower()
    score = 0.0
    for kw in keywords:
        if kw in haystack:
            score += 1.25

    q_lower = question.lower()
    ticker = str(meta.get("ticker") or "").strip().upper()
    if ticker and re.search(rf"\b{re.escape(ticker.lower())}\b", q_lower):
        score += 4.5

    company = str(meta.get("company") or "").strip().lower()
    if len(company) >= 4:
        norm = re.sub(r"[^\w\s]+", "", company)
        for part in norm.split():
            if len(part) >= 4 and part in q_lower:
                score += 2.0
                break

    form_val = str(meta.get("form_type") or "").lower().replace(" ", "")
    if form_val:
        if "10-k" in form_val or "10k" in form_val:
            if "10-k" in q_lower or re.search(r"\b10\s*k\b", q_lower) or "annual" in q_lower:
                score += 2.25
        if "10-q" in form_val or "10q" in form_val:
            if "10-q" in q_lower or re.search(r"\b10\s*q\b", q_lower) or "quarterly" in q_lower:
                score += 2.25

    section = str(meta.get("section_title") or "").lower()
    for kw in keywords:
        if len(kw) >= 5 and kw in section:
            score += 0.85

    body = doc.page_content.lower()[:12000]
    for kw in keywords:
        hits = body.count(kw)
        if hits:
            score += min(hits, 4) * 0.35

    return score


def fusion_metadata_keyword_overlap(doc: Document, query: str) -> int:
    """Light overlap score used inside reciprocal-rank fusion."""
    haystack = " ".join(
        str(doc.metadata.get(k, ""))
        for k in ("company", "ticker", "form_type", "quarter", "report_period", "filing_date")
    ).lower()
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9-]+", query)]
    return sum(1 for token in tokens if len(token) >= 3 and token in haystack)


def rerank_lexical_then_recency(
    docs: list[Document],
    query: str,
    *,
    disable_recency_boost: bool,
) -> list[Document]:
    """Rank by keyword/metadata match; break ties with newer filings unless period is pinned."""
    if not docs:
        return []
    pinned = query_pins_specific_period(query)
    ordered = list(docs)
    if not pinned and not disable_recency_boost:
        ordered.sort(key=lambda d: report_recency_epoch(d.metadata), reverse=True)
    ordered.sort(
        key=lambda d: keyword_metadata_alignment_score(d, query),
        reverse=True,
    )
    return ordered
