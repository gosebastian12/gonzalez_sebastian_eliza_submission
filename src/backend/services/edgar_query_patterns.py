"""Regex patterns and metadata-filter extraction for EDGAR filing queries."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Final

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

# Longest phrases first (matched against query.lower()).
COMPANY_NAME_TO_TICKER: Final[tuple[tuple[str, str], ...]] = tuple(
    sorted(
        [
            ("jpmorgan chase", "JPM"),
            ("jp morgan chase", "JPM"),
            ("jp morgan", "JPM"),
            ("jpmorgan", "JPM"),
            ("apple inc.", "AAPL"),
            ("apple inc", "AAPL"),
            ("apple", "AAPL"),
            ("tesla inc.", "TSLA"),
            ("tesla inc", "TSLA"),
            ("tesla", "TSLA"),
            ("microsoft corporation", "MSFT"),
            ("microsoft", "MSFT"),
            ("amazon.com", "AMZN"),
            ("amazon", "AMZN"),
            ("alphabet", "GOOGL"),
            ("google", "GOOGL"),
            ("nvidia corporation", "NVDA"),
            ("nvidia", "NVDA"),
            ("meta platforms", "META"),
            ("facebook", "META"),
            ("meta", "META"),
            ("berkshire hathaway", "BRK.B"),
            ("walt disney", "DIS"),
            ("disney", "DIS"),
            ("mcdonald", "MCD"),
            ("mcdonalds", "MCD"),
            ("coca-cola", "KO"),
            ("coca cola", "KO"),
            ("exxon", "XOM"),
            ("chevron", "CVX"),
            ("goldman sachs", "GS"),
            ("bank of america", "BAC"),
            ("wells fargo", "WFC"),
            ("citigroup", "C"),
            ("visa", "V"),
            ("mastercard", "MA"),
        ],
        key=lambda x: -len(x[0]),
    )
)

# Words that match ``TICKER_RE`` but are not stock symbols.
FALSE_TICKER_WORDS: Final[frozenset[str]] = frozenset(
    {
        "SEC",
        "AND",
        "THE",
        "FOR",
        "ARE",
        "NOT",
        "ALL",
        "ANY",
        "USD",
        "GAAP",
        "NYSE",
        "NASDAQ",
        "EDIT",
        "ITEM",
        "PART",
        "FORM",
        "NOTE",
        "NOTES",
        "RISK",
        "EPS",
        "CEO",
        "CFO",
        "IPO",
        "ETF",
        # Avoid matching the ``K`` / ``Q`` in ``10-K`` / ``10-Q``.
        "K",
        "Q",
        # Common ALL-CAPS words mis-read as tickers when users shout-case prompts.
        "HOW",
        "HAS",
        "HAD",
        "HIS",
        "HER",
        "OUR",
        "ITS",
        "LAST",
        "OVER",
        "YEAR",
        "YEARS",
        "TWO",
        "WAY",
        "DAY",
        "MAY",
        "NOW",
        "NEW",
        "OFF",
        "OUT",
        "OWN",
        "PER",
        "SEE",
        "SAY",
        "SHE",
        "END",
        "TOP",
        "NET",
        "LOW",
        "ROW",
        "GET",
        "GOT",
        "LET",
        "FAR",
        "FEW",
        "BACK",
        "TIME",
        "VERY",
        "WHEN",
        "WHAT",
        "WHO",
        "WHY",
        "INTO",
        "FROM",
        "THEN",
        "THAN",
        "THAT",
        "THIS",
        "WITH",
        "WILL",
        "JUST",
        "ONLY",
        "ALSO",
        "WELL",
        "MUCH",
        "SOME",
        "BEEN",
        "HAVE",
        "DOES",
        "WERE",
    }
)


def form_type_metadata_clause(form_val: str) -> dict[str, Any]:
    """Map normalized ``10-K`` / ``10-Q`` filters to values stored during ingestion."""
    raw = form_val.strip()
    norm = raw.upper().replace(" ", "")
    if norm == "10-K":
        variants = ("10-K", "10K", "10-K (Annual Report)")
    elif norm == "10-Q":
        variants = ("10-Q", "10Q", "10-Q (Quarterly Report)")
    else:
        return {"form_type": raw}
    return {"$or": [{"form_type": v} for v in variants]}


def normalize_query_text(raw: str) -> str:
    """Unicode-normalize and straighten apostrophes so company-name aliases match reliably."""
    s = unicodedata.normalize("NFKC", raw)
    return s.replace("\u2019", "'").replace("\u2018", "'")


def _alias_matches_query(name: str, ql: str) -> bool:
    """Avoid substring false positives (e.g. ``apple`` inside ``pineapple``)."""
    if not name.strip():
        return False
    if " " in name.strip():
        return name in ql
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", ql))


def extract_tickers_for_retrieval(query: str) -> list[str]:
    """Resolve explicit tickers plus common company names → ticker for multi-entity RAG."""
    ql = normalize_query_text(query).lower()
    found: list[str] = []
    seen: set[str] = set()

    for name, sym in COMPANY_NAME_TO_TICKER:
        if _alias_matches_query(name, ql) and sym not in seen:
            found.append(sym)
            seen.add(sym)

    for raw in TICKER_RE.findall(query):
        u = raw.upper()
        if u in FALSE_TICKER_WORDS or len(raw) < 2:
            continue
        if u not in seen:
            found.append(u)
            seen.add(u)

    return found


def company_name_ticker_boost(question_lower: str, ticker_upper: str) -> float:
    """Extra lexical score when the question names a company that maps to chunk metadata ticker."""
    if not ticker_upper.strip():
        return 0.0
    for name, sym in COMPANY_NAME_TO_TICKER:
        if sym == ticker_upper.strip().upper() and _alias_matches_query(name, question_lower):
            return 6.0
    return 0.0


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

    ticker_candidates = [
        t
        for t in TICKER_RE.findall(query)
        if t not in {"RAG", "LLM", "SEC"} and len(t) >= 2
    ]
    ticker_candidates = [t for t in ticker_candidates if t.upper() not in FALSE_TICKER_WORDS]
    if len(ticker_candidates) == 1:
        filters["ticker"] = ticker_candidates[0]

    return filters


def build_chroma_where_clause(query: str) -> dict[str, Any] | None:
    """Metadata-first narrowing for hybrid RAG: Chroma ``where`` applied before vector search.

    Combines explicit query cues (form, quarter, filing date) with ticker scope from
    ``extract_tickers_for_retrieval`` (company names + explicit symbols). Multiple tickers
    become ``{"$or": [{"ticker": ...}, ...]}`` merged with ``$and`` alongside other filters.
    """
    explicit = extract_metadata_filters(query)
    resolved = extract_tickers_for_retrieval(query)

    parts: list[dict[str, Any]] = []
    for key, val in explicit.items():
        if key == "ticker":
            continue
        if key == "form_type":
            parts.append(form_type_metadata_clause(val))
            continue
        parts.append({key: val})

    ticker_list: list[str] = []
    seen: set[str] = set()
    for t in resolved:
        u = t.strip().upper()
        if u and u not in seen:
            ticker_list.append(u)
            seen.add(u)
    if "ticker" in explicit:
        u = explicit["ticker"].strip().upper()
        if u and u not in seen:
            ticker_list.append(u)
            seen.add(u)

    if ticker_list:
        if len(ticker_list) == 1:
            parts.append({"ticker": ticker_list[0]})
        else:
            parts.append({"$or": [{"ticker": sym} for sym in ticker_list]})

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def build_chroma_where_clause_for_ticker(query: str, ticker: str) -> dict[str, Any] | None:
    """Metadata filter scoped to a single ticker (for multi-company hybrid retrieval).

    Reuses form/quarter/date constraints from the question but fixes ``ticker`` so semantic
    search runs per issuer instead of one global top-``k`` over ``$or`` tickers (which one
    company can dominate).
    """
    sym = ticker.strip().upper()
    if not sym:
        return None

    explicit = extract_metadata_filters(query)
    parts: list[dict[str, Any]] = []
    for key, val in explicit.items():
        if key == "ticker":
            continue
        if key == "form_type":
            parts.append(form_type_metadata_clause(val))
            continue
        parts.append({key: val})
    parts.append({"ticker": sym})

    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}
