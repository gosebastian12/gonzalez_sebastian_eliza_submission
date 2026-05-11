"""Format retrieved chunks into the LLM prompt under character/token budgets."""

from __future__ import annotations

from langchain_core.documents import Document


def effective_context_char_budget(
    *,
    num_ctx: int,
    num_predict: int,
    prompt_overhead_tokens: int,
    chars_per_token_estimate: float,
) -> int:
    """Upper bound on characters for retrieved context so the prompt fits under ``num_ctx``.

    Reserves tokens for generation (``num_predict``) and non-retrieval prompt overhead, then
    converts the remaining token budget to characters using ``chars_per_token_estimate``.

    Args:
        num_ctx: Ollama / model context window size (KV cache), e.g. from ``RAGConfig.num_ctx``.
        num_predict: Max completion tokens reserved, e.g. ``RAGConfig.num_predict``.
        prompt_overhead_tokens: Reserved tokens for system template, question wrappers, etc.
        chars_per_token_estimate: Heuristic chars per token for EDGAR-like text.

    Returns:
        A conservative character cap (at least 384) for ``build_retrieval_context``'s
        ``hard_cap_chars`` argument.
    """
    reserved = num_predict + prompt_overhead_tokens
    available_tokens = max(num_ctx - reserved, 128)
    return max(int(available_tokens * chars_per_token_estimate), 384)


def build_retrieval_context(
    docs: list[Document],
    *,
    max_context_chars: int,
    hard_cap_chars: int,
    min_partial_body_chars: int = 200,
) -> str:
    """Concatenate chunk bodies with headers under a shared character budget.

    Tries to include every retrieved document: if a full chunk does not fit, appends a truncated
    body (when the remaining space exceeds ``min_partial_body_chars``) so later chunks can
    still appear.

    Args:
        docs: Ordered LangChain ``Document`` list (typically post re-rank / ``final_k`` slice).
        max_context_chars: Soft cap from configuration (e.g. ``RAGConfig.max_context_chars``).
        hard_cap_chars: Hard ceiling from ``effective_context_char_budget``; budget is
            ``min(max(max_context_chars, 256), hard_cap_chars)`` then floored to at least 384.
        min_partial_body_chars: Minimum characters of body to keep when truncating a chunk;
            if remaining space is smaller, the chunk is skipped rather than emitting a sliver.

    Returns:
        Single string with ``\\n\\n``-separated blocks, each starting with a ``[Chunk n]`` header
        line built from document metadata.
    """
    budget = min(max(max_context_chars, 256), hard_cap_chars)
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
        sep_cost = 2 if blocks else 0
        block_full = f"{header}\n{body}"
        total_full = sep_cost + len(block_full)

        if used + total_full <= budget:
            blocks.append(block_full)
            used += total_full
            continue

        remaining_for_body = budget - used - sep_cost - len(header) - 1
        if remaining_for_body > min_partial_body_chars:
            truncated = f"{header}\n{body[:remaining_for_body]}…"
            blocks.append(truncated)
            used += sep_cost + len(truncated)

    return "\n\n".join(blocks)


def source_label(doc: Document) -> str:
    """Human-readable one-line label for a chunk (UI ``sources`` list and related JSON).

    Args:
        doc: A ``Document`` whose metadata includes ``file_name``, ``section_title``, and
            ideally ``chunk_index``.

    Returns:
        Pipe-separated label string used in API payloads and the chat template.
    """
    m = doc.metadata
    return (
        f"{m.get('file_name', 'unknown')} | "
        f"{m.get('section_title', 'unknown section')} | "
        f"chunk={m.get('chunk_index', 'n/a')}"
    )
