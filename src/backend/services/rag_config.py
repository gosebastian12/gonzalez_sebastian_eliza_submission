"""Environment-driven configuration for the EDGAR hybrid RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, or return ``default`` if unset/blank.

    Args:
        name: Environment variable name (e.g. ``RAG_NUM_CTX``).
        default: Value used when the variable is missing or empty after strip.

    Returns:
        Parsed ``int`` from the variable, or ``default``.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean-like environment variable, or return ``default`` if unset/blank.

    Args:
        name: Environment variable name.
        default: Value when missing/empty.

    Returns:
        ``default`` when unset or blank-after-strip. Otherwise ``True`` for ``1``, ``true``,
        ``yes``, ``on`` (case-insensitive), and ``False`` for any other non-empty value.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, or return ``default`` if unset/blank.

    Args:
        name: Environment variable name.
        default: Fallback when missing/empty.

    Returns:
        Parsed ``float``, or ``default``.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_str(name: str, default: str) -> str:
    """Read a string environment variable, or return ``default`` if unset.

    Args:
        name: Environment variable name.
        default: Used when the variable is missing; also when set to whitespace-only.

    Returns:
        Stripped string from the environment, or ``default``.
    """
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
    ``RAG_MIN_CHUNK_BODY_CHARS`` — prefer chunks at least this long for substantive tie-breaks.
    ``RAG_RERANK_LENGTH_LOG_WEIGHT`` — after relevance, favor longer chunks using ``log1p(char_len) * weight``.
    ``RAG_MULTI_ENTITY_PER_TICKER_SEMANTIC_K`` / ``multi_entity_per_ticker_semantic_k`` —
    Chroma ``k`` for **each** issuer when ``extract_tickers_for_retrieval`` finds **two or more**
    tickers (multi-company questions). ``EdgarHybridRAG.retrieve`` runs one
    ``similarity_search_by_vector`` per symbol with ``build_chroma_where_clause_for_ticker`` so
    hits are not drawn from a single global top-``k`` over an ``$or`` of tickers (where one
    embedding could dominate). The per-symbol ``k`` is
    ``max(multi_entity_per_ticker_semantic_k, (semantic_k + n - 1) // n)`` with ``n`` = number of
    detected tickers: your setting is a **floor** per ticker, while the second term **splits**
    ``semantic_k`` fairly across issuers (integer ceiling via ``+ n - 1``). If the merged pool
    is empty, retrieval **falls back** to one search with ``k=semantic_k`` and the usual combined
    ``where`` clause. For **zero or one** detected ticker this field is **not** used (retrieval
    uses ``semantic_k`` only). Integer; default ``5``. Env unset → dataclass default.
    """

    collection_name: str = "edgar_reports"
    chroma_host: str = "127.0.0.1"
    chroma_port: int = 8001
    embedding_model: str = "qwen3-embedding:0.6b"
    llm_model: str = "llama3.2:1b"
    semantic_k: int = 9
    multi_entity_per_ticker_semantic_k: int = 5
    final_k: int = 6
    min_chunk_body_chars: int = 80
    rerank_length_log_weight: float = 1.35
    use_reranker: bool = True
    rerank_top_n: int = 6
    disable_recency_boost: bool = False
    num_predict: int = 512
    num_ctx: int = 4096
    reasoning: bool | None = False
    keep_alive: str | None = "10m"
    ollama_http_timeout_s: float = 600.0
    max_context_chars: int = 9600
    prompt_overhead_tokens: int = 780
    chars_per_token_estimate: float = 3.3

    @classmethod
    def from_env(cls) -> RAGConfig:
        """Build a ``RAGConfig`` from process environment variables and normalize it.

        All knobs are optional at the OS level; defaults come from dataclass fields and
        ``RAGConfig``'s class docstring lists the supported ``RAG_*`` names.

        Returns:
            A new ``RAGConfig`` instance with ``.normalized()`` applied so ``num_predict``,
            ``num_ctx``, and overhead stay mutually consistent.
        """
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
            multi_entity_per_ticker_semantic_k=_env_int(
                "RAG_MULTI_ENTITY_PER_TICKER_SEMANTIC_K",
                cls.multi_entity_per_ticker_semantic_k,
            ),
            final_k=_env_int("RAG_FINAL_K", cls.final_k),
            min_chunk_body_chars=_env_int(
                "RAG_MIN_CHUNK_BODY_CHARS", cls.min_chunk_body_chars
            ),
            rerank_length_log_weight=_env_float(
                "RAG_RERANK_LENGTH_LOG_WEIGHT", cls.rerank_length_log_weight
            ),
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
        """Return a copy with ``num_predict``, overhead, and retrieval slack clamped into ``num_ctx``.

        Ensures a minimum slice of the context window remains for retrieved text, caps
        ``chars_per_token_estimate`` to a sane band, and may lower ``num_predict`` or
        ``prompt_overhead_tokens`` if their sum would exceed ``num_ctx``.

        Returns:
            New ``RAGConfig`` via ``dataclasses.replace``; fields not involved in the budget
            math are copied unchanged.
        """
        overhead = max(64, self.prompt_overhead_tokens)
        predict = max(32, self.num_predict)
        ctx = max(2048, self.num_ctx)
        min_retrieval_tokens = 512
        slack = 96

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
