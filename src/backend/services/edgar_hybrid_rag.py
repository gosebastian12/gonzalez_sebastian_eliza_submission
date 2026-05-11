"""Hybrid Chroma retrieval + Ollama generation for EDGAR filings."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from edgar_query_patterns import (
    build_chroma_where_clause,
    build_chroma_where_clause_for_ticker,
    extract_metadata_filters,
    extract_tickers_for_retrieval,
)
from rag_config import RAGConfig
from rag_context import (
    build_retrieval_context,
    effective_context_char_budget,
    source_label,
)
from rag_prompts import SYSTEM_PROMPT
from rag_reranking import query_pins_specific_period, rerank_lexical_then_recency


class EdgarHybridRAG:
    """Wire Chroma vector search, retrieval heuristics, re-ranking, and ChatOllama generation.

    ``answer`` builds a LangChain chat prompt (system + human) with retrieved filing text;
    ``retrieve`` applies metadata filters, embedding search, dedupe, and lexical/optional LLM re-rank.
    """

    def __init__(self, config: RAGConfig) -> None:
        """Create clients for embeddings, Chroma, and the chat model from ``config``.

        Args:
            config: ``RAGConfig`` instance; stored as ``self.config`` after ``.normalized()`` and
                used for collection name, Chroma host/port, Ollama models, and timeouts.
        """
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
        """Run retrieval, pack context under budget, invoke the chat model, and return API fields.

        Args:
            user_prompt: End-user question text (same string embedded for Chroma and shown in the
                human template as ``{question}``).

        Returns:
            Dict with ``answer`` (model text), ``sources`` / ``source_chunks``, metadata filter
            diagnostics, retrieval ``where`` clause echo, counts, and ``period_pinned_in_query``.
        """
        retrieved_docs, chroma_where = self.retrieve(user_prompt)
        hard_cap = effective_context_char_budget(
            num_ctx=self.config.num_ctx,
            num_predict=self.config.num_predict,
            prompt_overhead_tokens=self.config.prompt_overhead_tokens,
            chars_per_token_estimate=self.config.chars_per_token_estimate,
        )
        context = build_retrieval_context(
            retrieved_docs,
            max_context_chars=self.config.max_context_chars,
            hard_cap_chars=hard_cap,
        )

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
        n = len(retrieved_docs)
        return {
            "answer": response.content,
            "sources": [source_label(doc) for doc in retrieved_docs],
            "source_chunks": [
                {"label": source_label(doc), "content": doc.page_content}
                for doc in retrieved_docs
            ],
            "metadata_filters_used": extract_metadata_filters(user_prompt),
            "retrieval_where_clause": chroma_where,
            "retrieved_count": n,
            "sources_count": n,
            "period_pinned_in_query": query_pins_specific_period(user_prompt),
        }

    def retrieve(self, query: str) -> tuple[list[Document], dict[str, Any] | None]:
        """Metadata filter (Chroma ``where``) first, then semantic search inside that subset.

        Implements multi-ticker per-symbol pools with merge, empty-pool fallback, unfiltered
        fallback when filters yield nothing, dedupe, lexical re-rank, optional LLM prefix re-rank,
        and ``final_k`` truncation.

        Args:
            query: User question; drives ``build_chroma_where_clause``, embedding, and re-rank.

        Returns:
            ``(final_docs, chroma_where)`` where ``chroma_where`` is the metadata filter dict (or
            ``None``) passed to Chroma for the primary path.
        """
        chroma_where = build_chroma_where_clause(query)
        query_vector = self.embeddings.embed_query(query)
        tickers_multi = extract_tickers_for_retrieval(query)

        if len(tickers_multi) >= 2:
            per_ticker_k = max(
                self.config.multi_entity_per_ticker_semantic_k,
                (self.config.semantic_k + len(tickers_multi) - 1) // len(tickers_multi),
            )
            pools: list[Document] = []
            for sym in tickers_multi:
                wt = build_chroma_where_clause_for_ticker(query, sym)
                pools.extend(
                    self.vectorstore.similarity_search_by_vector(
                        query_vector,
                        k=per_ticker_k,
                        filter=wt,
                    )
                )
            semantic_docs = dedupe_documents(pools, document_identity_key)
            if not semantic_docs:
                semantic_docs = self.vectorstore.similarity_search_by_vector(
                    query_vector,
                    k=self.config.semantic_k,
                    filter=chroma_where,
                )
        else:
            semantic_docs = self.vectorstore.similarity_search_by_vector(
                query_vector,
                k=self.config.semantic_k,
                filter=chroma_where,
            )

        if not semantic_docs and chroma_where is not None:
            semantic_docs = self.vectorstore.similarity_search_by_vector(
                query_vector,
                k=self.config.semantic_k,
                filter=None,
            )
        semantic_docs = dedupe_documents(semantic_docs, document_identity_key)
        reranked = rerank_lexical_then_recency(
            semantic_docs,
            query,
            disable_recency_boost=self.config.disable_recency_boost,
            min_chunk_body_chars=self.config.min_chunk_body_chars,
            length_log_weight=self.config.rerank_length_log_weight,
        )
        if self.config.use_reranker:
            reranked = rerank_with_llm(self.llm, query, reranked, self.config.rerank_top_n)
        final_docs = reranked[: self.config.final_k]
        return final_docs, chroma_where


def dedupe_documents(docs: list[Document], key_fn: Callable[[Document], str]) -> list[Document]:
    """Return ``docs`` in first-seen order, skipping duplicates by ``key_fn(doc)``.

    Args:
        docs: Iterable of ``Document`` instances (order preserved for first occurrence).
        key_fn: Callable returning a hashable identity string per document.

    Returns:
        New list containing only the first document for each distinct key.
    """
    seen: set[str] = set()
    out: list[Document] = []
    for doc in docs:
        key = key_fn(doc)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def document_identity_key(doc: Document) -> str:
    """Stable string key for deduping chunks from the same filing section/index.

    Args:
        doc: ``Document`` with ``file_name``, ``chunk_index``, and ``section_title`` metadata.

    Returns:
        ``"{file_name}::{chunk_index}::{section_title}"`` with empty-string fallbacks for missing keys.
    """
    meta = doc.metadata
    return (
        f"{meta.get('file_name', '')}::"
        f"{meta.get('chunk_index', '')}::"
        f"{meta.get('section_title', '')}"
    )


def rerank_with_llm(llm: ChatOllama, query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Re-score the first ``top_n`` documents with the LLM and move them ahead by score.

    Each candidate receives a short snippet and a JSON ``{"score": ...}`` style response;
    ``parse_llm_score`` extracts the numeric value. Remaining documents keep lexical order after
    the rescored prefix.

    Args:
        llm: Configured ``ChatOllama`` instance (same as generation model).
        query: User question passed into the scoring prompt.
        docs: Full candidate list after lexical re-rank.
        top_n: Number of leading documents to send through LLM scoring (must be ``>= 0``).

    Returns:
        New list: LLM-sorted prefix of length ``<= top_n`` followed by the untouched tail
        ``docs[top_n:]``.
    """
    scored: list[tuple[float, Document]] = []
    for doc in docs:
        snippet = doc.page_content[:1500]
        scoring_prompt = (
            "Score relevance from 0 to 100 for the user question.\n"
            'Return only JSON: {"score": <number>}.\n\n'
            f"Question: {query}\n\nDocument:\n{snippet}"
        )
        message = llm.invoke([SystemMessage(content=""), HumanMessage(content=scoring_prompt)])
        score = parse_llm_score(str(message.content))
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:top_n:]]


def parse_llm_score(raw: str) -> float:
    """Parse a relevance score from LLM output that may be JSON or free text.

    Args:
        raw: Model response string; ideally JSON ``{"score": <number>}``.

    Returns:
        Float score from JSON when valid; otherwise first number match in ``raw``, or ``0.0``.
    """
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return float(payload.get("score", 0.0))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    number = re.search(r"(\d+(?:\.\d+)?)", raw)
    return float(number.group(1)) if number else 0.0
