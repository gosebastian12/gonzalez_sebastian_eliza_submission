"""CLI entrypoint and stable imports for the EDGAR hybrid RAG service.

``fastapi`` / ``importlib`` load this file by path; sibling modules are imported by temporarily
adding this directory to ``sys.path``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

_SERVICES_DIR = Path(__file__).resolve().parent
if str(_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICES_DIR))

from edgar_hybrid_rag import EdgarHybridRAG  # noqa: E402
from rag_config import RAGConfig  # noqa: E402

__all__ = ["RAGConfig", "EdgarHybridRAG", "estimate_k_impact", "parse_args", "main"]


def estimate_k_impact(
    avg_chunk_tokens: int = 360,
    query_and_format_tokens: int = 780,
    context_window_tokens: int = 4096,
) -> dict[str, Any]:
    """Rough token budget table for how many retrieved chunks ``k`` might fit in a window.

    Used for CLI-side reporting only; does not change ``EdgarHybridRAG`` behavior.

    Args:
        avg_chunk_tokens: Assumed mean tokens per retrieved chunk body.
        query_and_format_tokens: Reserved tokens for system + question + wrappers.
        context_window_tokens: Total context size to model (e.g. ``num_ctx``).

    Returns:
        Dict with ``assumptions``, ``recommendations`` (per-``k`` rows), and ``suggested_default_k``.
    """
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
    """Parse CLI flags for ad-hoc ``EdgarHybridRAG.answer`` runs.

    Returns:
        ``argparse.Namespace`` with ``--query`` (required) and optional overrides for collection,
        Chroma host/port, embedding and LLM model names, ``semantic_k``, ``final_k``, and
        ``--use-reranker`` (flag). Other ``RAG_*`` settings still load from the environment via
        ``RAGConfig.from_env()`` before ``dataclasses.replace``.
    """
    parser = argparse.ArgumentParser(description="Hybrid retrieval + prompt injection for EDGAR RAG.")
    parser.add_argument("--query", required=True, help="User prompt text.")
    parser.add_argument("--collection", default="edgar_reports")
    parser.add_argument("--chroma-host", default="127.0.0.1")
    parser.add_argument("--chroma-port", type=int, default=8001)
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--llm-model", default="llama3.2:1b")
    parser.add_argument("--semantic-k", type=int, default=9)
    parser.add_argument("--final-k", type=int, default=7)
    parser.add_argument("--use-reranker", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entry: merge flags with env-backed config, run ``rag.answer``, print JSON + timing.

    Side effects: prints ``answer`` payload JSON, ``estimate_k_impact`` JSON, and elapsed seconds
    to stdout.
    """
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
