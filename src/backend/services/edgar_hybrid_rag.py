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

from edgar_query_patterns import extract_metadata_filters
from rag_config import RAGConfig
from rag_context import (
    build_retrieval_context,
    effective_context_char_budget,
    source_label,
)
from rag_prompts import SYSTEM_PROMPT
from rag_reranking import (
    fusion_metadata_keyword_overlap,
    query_pins_specific_period,
    rerank_lexical_then_recency,
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
        return {
            "answer": response.content,
            "sources": [source_label(doc) for doc in retrieved_docs],
            "metadata_filters_used": extract_metadata_filters(user_prompt),
            "retrieved_count": len(retrieved_docs),
            "period_pinned_in_query": query_pins_specific_period(user_prompt),
        }

    def retrieve(self, query: str) -> list[Document]:
        query_vector = self.embeddings.embed_query(query)
        semantic_docs = self.vectorstore.similarity_search_by_vector(
            query_vector, k=self.config.semantic_k
        )

        metadata_filters = extract_metadata_filters(query)
        metadata_docs: list[Document] = []
        if metadata_filters:
            data = self.vectorstore._collection.get(  # noqa: SLF001
                where=metadata_filters,
                limit=self.config.metadata_k,
                include=["documents", "metadatas"],
            )
            metadata_docs = documents_from_chroma_get(data)

        fused = fuse_rrf_rankings(
            semantic_docs,
            metadata_docs,
            query,
            doc_key_fn=document_identity_key,
            overlap_fn=fusion_metadata_keyword_overlap,
        )
        reranked = rerank_lexical_then_recency(
            fused,
            query,
            disable_recency_boost=self.config.disable_recency_boost,
        )
        if self.config.use_reranker:
            reranked = rerank_with_llm(self.llm, query, reranked, self.config.rerank_top_n)
        return reranked[: self.config.final_k]


def documents_from_chroma_get(data: dict[str, Any]) -> list[Document]:
    docs: list[Document] = []
    texts = data.get("documents") or []
    metas = data.get("metadatas") or []
    for index, text in enumerate(texts):
        if not text:
            continue
        metadata = metas[index] if index < len(metas) and metas[index] else {}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def document_identity_key(doc: Document) -> str:
    meta = doc.metadata
    return (
        f"{meta.get('file_name', '')}::"
        f"{meta.get('chunk_index', '')}::"
        f"{meta.get('section_title', '')}"
    )


def fuse_rrf_rankings(
    semantic_docs: list[Document],
    metadata_docs: list[Document],
    query: str,
    *,
    doc_key_fn: Callable[[Document], str],
    overlap_fn: Callable[[Document, str], int],
    rrf_k: int = 60,
    overlap_weight: float = 0.015,
) -> list[Document]:
    """Reciprocal rank fusion with optional lexical overlap bonus on the metadata list."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(semantic_docs, start=1):
        key = doc_key_fn(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(metadata_docs, start=1):
        key = doc_key_fn(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        scores[key] += overlap_weight * overlap_fn(doc, query)
        doc_map[key] = doc

    ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [doc_map[key] for key in ranked_keys]


def rerank_with_llm(llm: ChatOllama, query: str, docs: list[Document], top_n: int) -> list[Document]:
    candidate_docs = docs[:top_n]
    scored: list[tuple[float, Document]] = []
    for doc in candidate_docs:
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
    reranked_docs = [doc for _, doc in scored]
    tail = docs[top_n:]
    return reranked_docs + tail


def parse_llm_score(raw: str) -> float:
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return float(payload.get("score", 0.0))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    number = re.search(r"(\d+(?:\.\d+)?)", raw)
    return float(number.group(1)) if number else 0.0
