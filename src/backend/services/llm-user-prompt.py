from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
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

# Leave blank per requirement.
SYSTEM_PROMPT = ""


@dataclass
class RAGConfig:
    collection_name: str = "edgar_reports"
    chroma_host: str = "127.0.0.1"
    chroma_port: int = 8001
    embedding_model: str = "qwen3-embedding:8b"
    llm_model: str = "gpt-oss:20b"
    semantic_k: int = 8
    final_k: int = 6
    use_reranker: bool = False
    rerank_top_n: int = 12


class EdgarHybridRAG:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.embeddings = OllamaEmbeddings(model=config.embedding_model)
        self.vectorstore = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embeddings,
            host=config.chroma_host,
            port=config.chroma_port,
        )
        self.llm = ChatOllama(model=config.llm_model, temperature=0)

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
        }

    def retrieve(self, query: str) -> list[Document]:
        semantic_docs = self.vectorstore.similarity_search(query, k=self.config.semantic_k)

        # Metadata-keyword path: parse known report metadata tokens from user prompt.
        metadata_filters = self._extract_metadata_filters(query)
        metadata_docs: list[Document] = []
        if metadata_filters:
            metadata_docs = self.vectorstore.similarity_search(
                query,
                k=self.config.semantic_k,
                filter=metadata_filters,
            )

        fused = self._fuse_rankings(semantic_docs, metadata_docs, query)
        if self.config.use_reranker:
            fused = self._rerank_with_llm(query, fused, self.config.rerank_top_n)
        return fused[: self.config.final_k]

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

    @staticmethod
    def _build_context(docs: list[Document]) -> str:
        blocks: list[str] = []
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
            blocks.append(f"{header}\n{doc.page_content.strip()}")
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
    query_and_format_tokens: int = 220,
    context_window_tokens: int = 8192,
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
        "suggested_default_k": 8,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid retrieval + prompt injection for EDGAR RAG.")
    parser.add_argument("--query", required=True, help="User prompt text.")
    parser.add_argument("--collection", default="edgar_reports")
    parser.add_argument("--chroma-host", default="127.0.0.1")
    parser.add_argument("--chroma-port", type=int, default=8001)
    parser.add_argument("--embedding-model", default="qwen3-embedding:8b")
    parser.add_argument("--llm-model", default="gpt-oss:20b")
    parser.add_argument("--semantic-k", type=int, default=8)
    parser.add_argument("--final-k", type=int, default=6)
    parser.add_argument("--use-reranker", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RAGConfig(
        collection_name=args.collection,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        semantic_k=args.semantic_k,
        final_k=args.final_k,
        use_reranker=args.use_reranker,
    )
    rag = EdgarHybridRAG(config)
    result = rag.answer(args.query)
    print(json.dumps(result, indent=2))
    print(json.dumps({"k_analysis": estimate_k_impact()}, indent=2))


if __name__ == "__main__":
    main()
