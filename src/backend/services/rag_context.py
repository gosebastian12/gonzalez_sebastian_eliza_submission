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
) -> str:
    """Concatenate chunk bodies with headers; respects ``min(max_context_chars, hard_cap_chars)``."""
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


def source_label(doc: Document) -> str:
    m = doc.metadata
    return (
        f"{m.get('file_name', 'unknown')} | "
        f"{m.get('section_title', 'unknown section')} | "
        f"chunk={m.get('chunk_index', 'n/a')}"
    )
