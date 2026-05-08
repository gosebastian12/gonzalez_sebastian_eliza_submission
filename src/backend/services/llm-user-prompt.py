from __future__ import annotations

import argparse
import json
import os
import re
import time
from calendar import monthrange
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings


FORM_RE = re.compile(r"\b10[- ]?([kq])\b", re.IGNORECASE)
QUARTER_RE = re.compile(r"\b(20\d{2}Q[1-4])\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
# Calendar year or fiscal-year style mentions pin retrieval to a period (skip recency boost).
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
FY_YEAR_RE = re.compile(r"\bfy\s*[- ]?\s*((?:19|20)\d{2})\b", re.IGNORECASE)
FISCAL_YEAR_RE = re.compile(
    r"\bfiscal(?:\s+year)?\s+(?:of\s+)?((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
FILENAME_ISO_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_")
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

# Leave blank per requirement.
SYSTEM_PROMPT = """
You are a helpful assistant that can answer user business and investment focused questions by
analyzing and summarizing the contextual text provided in the second half of this prompt. This
context is text from recent quarterly (10-Q) *and* annual (10-K) SEC EDGAR filings.
You need to answer the question based on the provided context. Do not hallcuniate or ignore the
given context whatsoever. You must prioritize the context over your own knowledge or prior experience.

That context may include tabular financial data whose structure is typically:
    | Column 1 | | Column 2 | | Column 3 |
    Sub-header/Additional-Column-Name | | | ... |
    | Entry 1   | Entry 2    | Entry 3   |
    | Entry 4   | Entry 5    | Entry 6   |
    ...
    <Aggregate-Header> | <Aggregate-Entry> | <Aggregate-Entry> | ... |
Note that symbols such as "$" or "%" may be used to indicate currency or percentages.
You are expected to extract the relevant data, conduct a analysis w/it, and use your results to
answer the user's question(s).

Utilize a professional, business-appropriate tone. Provide objective results that are helpful, insightful,
and based in the reality described below.

========================================
Textual context to emphasis::

"""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    stripped = raw.strip()
    return stripped if stripped else default


@dataclass
class RAGConfig:
    """Tune via env vars (all optional).

    Context budget (llama3.2:1b supports a large window in Ollama; defaults stay modest for RAM/latency).
    ``RAG_NUM_CTX`` — passed to Ollama as ``num_ctx`` (KV cache size).
    ``RAG_NUM_PREDICT`` — max output tokens; must leave room inside ``RAG_NUM_CTX``.
    ``RAG_PROMPT_OVERHEAD_TOKENS`` — reserve tokens for system prompt, chat template, question & wrappers.
    ``RAG_MAX_CONTEXT_CHARS`` — soft cap on retrieved text length (also clamped by context math).
    ``RAG_CHARS_PER_TOKEN_EST`` — chars/token guess for clamping retrieved text (tabular EDGAR text ≈ 3–4).
    ``RAG_NUM_CTX_HARD_CAP`` — safety clamp on absurd ``RAG_NUM_CTX`` values (default 131072).
    ``RAG_DISABLE_RECENCY_BOOST`` — skip recency tie-breaking in lexical re-ranking (testing only).
    """

    collection_name: str = "edgar_reports"
    chroma_host: str = "127.0.0.1"
    chroma_port: int = 8001
    embedding_model: str = "qwen3-embedding:0.6b"
    llm_model: str = "llama3.2:1b"
    semantic_k: int = 6
    metadata_k: int = 12
    final_k: int = 4
    use_reranker: bool = False
    rerank_top_n: int = 3
    disable_recency_boost: bool = False
    # Generation / latency (Ollama). Defaults sized so prompt + retrieval fit comfortably in num_ctx.
    num_predict: int = 256
    num_ctx: int = 4096
    reasoning: bool | None = False
    keep_alive: str | None = "10m"
    ollama_http_timeout_s: float = 600.0
    # Prompt packing (retrieval injected into user message — must fit remaining num_ctx budget).
    max_context_chars: int = 9600
    prompt_overhead_tokens: int = 780
    chars_per_token_estimate: float = 3.3

    @classmethod
    def from_env(cls) -> RAGConfig:
        hard_cap = _env_int("RAG_NUM_CTX_HARD_CAP", 131072)
        raw_ctx = _env_int("RAG_NUM_CTX", cls.num_ctx)
        num_ctx = max(2048, min(raw_ctx, hard_cap))

        cfg = cls(
            collection_name=_env_str("RAG_COLLECTION", cls.collection_name),
            chroma_host=_env_str("RAG_CHROMA_HOST", cls.chroma_host),
            chroma_port=_env_int("RAG_CHROMA_PORT", cls.chroma_port),
            embedding_model=_env_str("RAG_EMBEDDING_MODEL", cls.embedding_model),
            llm_model=_env_str("RAG_LLM_MODEL", cls.llm_model),
            semantic_k=_env_int("RAG_SEMANTIC_K", cls.semantic_k),
            metadata_k=_env_int("RAG_METADATA_K", cls.metadata_k),
            final_k=_env_int("RAG_FINAL_K", cls.final_k),
            use_reranker=_env_bool("RAG_USE_RERANKER", cls.use_reranker),
            rerank_top_n=_env_int("RAG_RERANK_TOP_N", cls.rerank_top_n),
            disable_recency_boost=_env_bool("RAG_DISABLE_RECENCY_BOOST", cls.disable_recency_boost),
            num_predict=_env_int("RAG_NUM_PREDICT", cls.num_predict),
            num_ctx=num_ctx,
            reasoning=(
                False
                if os.environ.get("RAG_REASONING", "").strip().lower() in {"0", "false", "no", "off"}
                else (
                    True
                    if os.environ.get("RAG_REASONING", "").strip().lower()
                    in {"1", "true", "yes", "on"}
                    else cls.reasoning
                )
            ),
            keep_alive=os.environ.get("RAG_KEEP_ALIVE", cls.keep_alive),
            ollama_http_timeout_s=_env_float("RAG_OLLAMA_HTTP_TIMEOUT_S", cls.ollama_http_timeout_s),
            max_context_chars=_env_int("RAG_MAX_CONTEXT_CHARS", cls.max_context_chars),
            prompt_overhead_tokens=_env_int("RAG_PROMPT_OVERHEAD_TOKENS", cls.prompt_overhead_tokens),
            chars_per_token_estimate=_env_float(
                "RAG_CHARS_PER_TOKEN_EST", cls.chars_per_token_estimate
            ),
        )
        return cfg.normalized()

    def normalized(self) -> RAGConfig:
        """Ensure ``num_predict``, overhead, and a minimum retrieval budget fit in ``num_ctx``."""
        overhead = max(64, self.prompt_overhead_tokens)
        predict = max(32, self.num_predict)
        ctx = max(2048, self.num_ctx)
        min_retrieval_tokens = 512
        slack = 96

        # Leave room for retrieved text in the user message (see ``_effective_context_char_budget``).
        max_predict_room = ctx - overhead - min_retrieval_tokens - slack
        if predict > max_predict_room:
            predict = max(max_predict_room, 64)

        total_fixed = overhead + predict + min_retrieval_tokens + slack
        if total_fixed > ctx:
            overhead = max(64, ctx - predict - min_retrieval_tokens - slack)

        est = max(2.5, min(self.chars_per_token_estimate, 6.0))
        return replace(
            self,
            num_ctx=ctx,
            num_predict=predict,
            prompt_overhead_tokens=overhead,
            chars_per_token_estimate=est,
        )


class EdgarHybridRAG:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config.normalized()
        timeout_kw = {"timeout": self.config.ollama_http_timeout_s}
        self.embeddings = OllamaEmbeddings(
            model=self.config.embedding_model,
            sync_client_kwargs=timeout_kw,
        )
        self.vectorstore = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
            host=self.config.chroma_host,
            port=self.config.chroma_port,
        )
        self.llm = ChatOllama(
            model=self.config.llm_model,
            temperature=0,
            num_predict=self.config.num_predict,
            num_ctx=self.config.num_ctx,
            reasoning=self.config.reasoning,
            keep_alive=self.config.keep_alive,
            client_kwargs=timeout_kw,
        )

    def answer(self, user_prompt: str) -> dict[str, Any]:
        retrieved_docs = self.retrieve(user_prompt)
        context = self._build_context(retrieved_docs)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "User question:\n{question}\n\n"
                    "Retrieved context:\n{context}\n\n"
                    "Use only relevant retrieved context. If context is insufficient, say so clearly.",
                ),
            ]
        )
        chain = prompt | self.llm
        response = chain.invoke({"question": user_prompt, "context": context})
        return {
            "answer": response.content,
            "sources": [self._source_label(doc) for doc in retrieved_docs],
            "metadata_filters_used": self._extract_metadata_filters(user_prompt),
            "retrieved_count": len(retrieved_docs),
            "period_pinned_in_query": query_pins_specific_period(user_prompt),
        }

    def retrieve(self, query: str) -> list[Document]:
        # Reuse one query embedding for all vector lookups.
        query_vector = self.embeddings.embed_query(query)
        semantic_docs = self.vectorstore.similarity_search_by_vector(
            query_vector, k=self.config.semantic_k
        )

        # Metadata-keyword path: parse known report metadata tokens from user prompt.
        metadata_filters = self._extract_metadata_filters(query)
        metadata_docs: list[Document] = []
        if metadata_filters:
            # Use metadata-only fetch first to avoid a second expensive vector query when
            # filtering by known report metadata (ticker/form/quarter/date).
            data = self.vectorstore._collection.get(  # noqa: SLF001
                where=metadata_filters,
                limit=self.config.metadata_k,
                include=["documents", "metadatas"],
            )
            metadata_docs = self._documents_from_get_result(data)

        fused = self._fuse_rankings(semantic_docs, metadata_docs, query)
        reranked = self._rerank_lexical_then_recency(fused, query)
        if self.config.use_reranker:
            reranked = self._rerank_with_llm(query, reranked, self.config.rerank_top_n)
        return reranked[: self.config.final_k]

    def _rerank_lexical_then_recency(self, docs: list[Document], query: str) -> list[Document]:
        """Rank by keyword/metadata match; break ties with newer filings unless period is pinned."""
        if not docs:
            return []
        pinned = query_pins_specific_period(query)
        ordered = list(docs)
        if not pinned and not self.config.disable_recency_boost:
            ordered.sort(key=lambda d: report_recency_epoch(d.metadata), reverse=True)
        ordered.sort(
            key=lambda d: keyword_metadata_alignment_score(d, query),
            reverse=True,
        )
        return ordered

    @staticmethod
    def _documents_from_get_result(data: dict[str, Any]) -> list[Document]:
        docs: list[Document] = []
        texts = data.get("documents") or []
        metas = data.get("metadatas") or []
        for index, text in enumerate(texts):
            if not text:
                continue
            metadata = metas[index] if index < len(metas) and metas[index] else {}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def _extract_metadata_filters(self, query: str) -> dict[str, str]:
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

        # Keep ticker optional: only if likely explicit stock symbol mention.
        ticker_candidates = [t for t in TICKER_RE.findall(query) if t not in {"RAG", "LLM", "SEC"}]
        if len(ticker_candidates) == 1:
            filters["ticker"] = ticker_candidates[0]

        return filters

    def _doc_key(self, doc: Document) -> str:
        meta = doc.metadata
        return (
            f"{meta.get('file_name', '')}::"
            f"{meta.get('chunk_index', '')}::"
            f"{meta.get('section_title', '')}"
        )

    def _fuse_rankings(
        self,
        semantic_docs: list[Document],
        metadata_docs: list[Document],
        query: str,
    ) -> list[Document]:
        # Reciprocal rank fusion across two ranked lists + metadata keyword bonus.
        rrf_k = 60
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(semantic_docs, start=1):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_map[key] = doc

        for rank, doc in enumerate(metadata_docs, start=1):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            # Bonus for metadata overlap with user prompt terms.
            scores[key] += 0.015 * self._metadata_keyword_overlap(doc, query)
            doc_map[key] = doc

        ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
        return [doc_map[key] for key in ranked_keys]

    def _metadata_keyword_overlap(self, doc: Document, query: str) -> int:
        haystack = " ".join(
            str(doc.metadata.get(k, ""))
            for k in ("company", "ticker", "form_type", "quarter", "report_period", "filing_date")
        ).lower()
        tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9-]+", query)]
        return sum(1 for token in tokens if len(token) >= 3 and token in haystack)

    def _rerank_with_llm(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        candidate_docs = docs[:top_n]
        scored: list[tuple[float, Document]] = []
        for doc in candidate_docs:
            snippet = doc.page_content[:1500]
            scoring_prompt = (
                "Score relevance from 0 to 100 for the user question.\n"
                "Return only JSON: {\"score\": <number>}.\n\n"
                f"Question: {query}\n\nDocument:\n{snippet}"
            )
            message = self.llm.invoke(
                [SystemMessage(content=""), HumanMessage(content=scoring_prompt)]
            )
            score = self._safe_parse_score(str(message.content))
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        reranked_docs = [doc for _, doc in scored]

        # Keep unscored tail in original order.
        tail = docs[top_n:]
        return reranked_docs + tail

    @staticmethod
    def _safe_parse_score(raw: str) -> float:
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return float(payload.get("score", 0.0))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        number = re.search(r"(\d+(?:\.\d+)?)", raw)
        return float(number.group(1)) if number else 0.0

    def _effective_context_char_budget(self) -> int:
        """Max chars for retrieved context so prompt fits under ``num_ctx``."""
        cfg = self.config
        reserved = cfg.num_predict + cfg.prompt_overhead_tokens
        available_tokens = cfg.num_ctx - reserved
        available_tokens = max(available_tokens, 128)
        return max(int(available_tokens * cfg.chars_per_token_estimate), 384)

    def _build_context(self, docs: list[Document]) -> str:
        hard_cap = self._effective_context_char_budget()
        budget = min(max(self.config.max_context_chars, 256), hard_cap)
        budget = max(budget, 384)
        blocks: list[str] = []
        used = 0
        for i, doc in enumerate(docs, start=1):
            m = doc.metadata
            header = (
                f"[Chunk {i}] "
                f"ticker={m.get('ticker', '')} "
                f"form={m.get('form_type', '')} "
                f"period={m.get('report_period', '')} "
                f"file={m.get('file_name', '')} "
                f"section={m.get('section_title', '')}"
            )
            body = doc.page_content.strip()
            block = f"{header}\n{body}"
            overhead = len(block) + (2 if blocks else 0)
            if used + overhead > budget:
                remaining = budget - used - len(header) - 3
                if remaining > 200:
                    block = f"{header}\n{body[:remaining]}…"
                    blocks.append(block)
                break
            blocks.append(block)
            used += overhead
        return "\n\n".join(blocks)

    @staticmethod
    def _source_label(doc: Document) -> str:
        m = doc.metadata
        return (
            f"{m.get('file_name', 'unknown')} | "
            f"{m.get('section_title', 'unknown section')} | "
            f"chunk={m.get('chunk_index', 'n/a')}"
        )


def estimate_k_impact(
    avg_chunk_tokens: int = 360,
    query_and_format_tokens: int = 780,
    context_window_tokens: int = 4096,
) -> dict[str, Any]:
    # Conservative planning values for prompt-budgeting with large models.
    recommendations: list[dict[str, int]] = []
    for k in (4, 6, 8, 10):
        retrieval_tokens = k * avg_chunk_tokens
        remaining_for_system_and_generation = (
            context_window_tokens - query_and_format_tokens - retrieval_tokens
        )
        recommendations.append(
            {
                "k": k,
                "estimated_retrieval_tokens": retrieval_tokens,
                "estimated_remaining_tokens": max(0, remaining_for_system_and_generation),
            }
        )
    return {
        "assumptions": {
            "avg_chunk_tokens": avg_chunk_tokens,
            "query_and_format_tokens": query_and_format_tokens,
            "context_window_tokens": context_window_tokens,
        },
        "recommendations": recommendations,
        "suggested_default_k": 6,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid retrieval + prompt injection for EDGAR RAG.")
    parser.add_argument("--query", required=True, help="User prompt text.")
    parser.add_argument("--collection", default="edgar_reports")
    parser.add_argument("--chroma-host", default="127.0.0.1")
    parser.add_argument("--chroma-port", type=int, default=8001)
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--llm-model", default="llama3.2:1b")
    parser.add_argument("--semantic-k", type=int, default=6)
    parser.add_argument("--final-k", type=int, default=4)
    parser.add_argument("--use-reranker", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = time.time()
    args = parse_args()
    config = replace(
        RAGConfig.from_env(),
        collection_name=args.collection,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        semantic_k=args.semantic_k,
        final_k=args.final_k,
        use_reranker=args.use_reranker,
    ).normalized()
    rag = EdgarHybridRAG(config)
    result = rag.answer(args.query)
    print(json.dumps(result, indent=2))
    print(json.dumps({"k_analysis": estimate_k_impact()}, indent=2))
    print(json.dumps({"timing_seconds": round(time.time() - started, 3)}, indent=2))


if __name__ == "__main__":
    main()
