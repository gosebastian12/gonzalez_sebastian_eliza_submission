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
    """Upper bound on characters for retrieved context so the prompt fits under ``num_ctx``."""
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
    """Concatenate chunk bodies with headers under the char budget.

    Tries to include **every** retrieved chunk: uses a truncated body when a full chunk does not
    fit, then continues (so later chunks are not dropped just because an earlier one was large).
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
    m = doc.metadata
    return (
        f"{m.get('file_name', 'unknown')} | "
        f"{m.get('section_title', 'unknown section')} | "
        f"chunk={m.get('chunk_index', 'n/a')}"
    )
