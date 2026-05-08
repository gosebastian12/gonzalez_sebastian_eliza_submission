from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


SEC_REPORT_START_RE = re.compile(
    r"UNITED STATES\s*SECURITIES AND EXCHANGE COMMISSION",
    re.IGNORECASE,
)
HEADING_RE = re.compile(
    r"(?im)^\s*(PART\s+[IVX]+(?:\s*[-–—:]\s*.*)?|ITEM\s+\d+[A-Z]?(?:\.\s*|[-–—:]\s*|\s+).*)$"
)
SUBHEADING_RE = re.compile(
    r"(?im)^\s*(?:[A-Z][A-Z0-9&(),\-/ ]{6,}|(?:Note|Notes)\s+\d+[A-Z]?(?:\.\s*|[-–—:]\s*|\s+).*)$"
)
TABLE_LINE_RE = re.compile(r"\|")
TABLE_FOOTNOTE_RE = re.compile(r"^\s*(?:see|note|notes|\(|\*)", re.IGNORECASE)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = REPO_ROOT.parent / "edgar_corpus"


@dataclass
class SectionBlock:
    title: str
    content: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk SEC EDGAR filings and ingest them into Chroma with LangChain."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Directory containing *_full.txt SEC report files.",
    )
    parser.add_argument(
        "--glob",
        default="*_full.txt",
        help="Glob pattern within corpus-dir for reports.",
    )
    parser.add_argument(
        "--collection",
        default="edgar_reports",
        help="Target Chroma collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        default="qwen3-embedding:0.6b",
        help="Ollama embedding model.",
    )
    parser.add_argument("--chroma-host", default="127.0.0.1", help="Chroma host.")
    parser.add_argument("--chroma-port", type=int, default=8001, help="Chroma host port.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2200,
        help="Maximum characters per final chunk.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=250,
        help="Character overlap used when splitting long chunks.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Optional cap on number of files to process (0 = all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build chunks and print stats without embedding or ingestion.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks per embedding+ingestion batch.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip chunks already present in the target collection by deterministic id.",
    )
    return parser.parse_args()


def read_report(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_header_metadata(raw_text: str, filename: str) -> dict[str, str]:
    metadata: dict[str, str] = {"file_name": filename}
    allowed = {
        "company": "company",
        "company_name": "company",
        "ticker": "ticker",
        "filing_type": "filing_type",
        "filing_date": "filing_date",
        "report_period": "report_period",
        "quarter": "quarter",
        "cik": "cik",  # central index key
        "source": "source",
    }

    for line in raw_text.splitlines()[:40]:
        if line.startswith("==="):
            break
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        key = key.strip().lower().replace(" ", "_")
        key = allowed.get(key)
        if not key:
            continue
        value = value.strip()
        if value:
            metadata[key] = value

    if not metadata.get("ticker") or not metadata.get("filing_type"):
        name_bits = filename.replace(".txt", "").split("_")
        if len(name_bits) >= 5:
            metadata.setdefault("ticker", name_bits[0])
            metadata.setdefault("filing_type", name_bits[1])
            metadata.setdefault("quarter", name_bits[2])
            metadata.setdefault("filing_date", name_bits[3])

    metadata.setdefault("company", "")
    metadata.setdefault("form_type", metadata.get("filing_type", ""))
    metadata.setdefault("report_period", metadata.get("report_period", ""))
    return metadata


def strip_preface_noise(raw_text: str) -> str:
    match = SEC_REPORT_START_RE.search(raw_text)
    if match:
        return raw_text[match.start() :].strip()
    return raw_text.strip()


def normalize_for_chunking(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # These files often concatenate tokens into very long lines.
    markers = [
        r"Table of Contents",
        r"PART\s+[IVX]+",
        r"Item\s+\d+[A-Z]?",
        r"SIGNATURES?",
        r"UNITED STATES",
    ]
    for marker in markers:
        normalized = re.sub(fr"(?i)\s*({marker})\s*", r"\n\1\n", normalized)

    # Avoid huge run-on lines by introducing line breaks before likely row separators.
    normalized = normalized.replace(" | ", " |\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_into_sections(text: str) -> list[SectionBlock]:
    lines = text.splitlines()
    sections: list[SectionBlock] = []
    current_title = "DOCUMENT_START"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if HEADING_RE.match(stripped):
            if current_lines:
                sections.append(
                    SectionBlock(title=current_title, content="\n".join(current_lines).strip())
                )
            current_title = stripped
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        sections.append(SectionBlock(title=current_title, content="\n".join(current_lines).strip()))

    return [section for section in sections if section.content]


def is_table_line(line: str) -> bool:
    if TABLE_LINE_RE.search(line):
        return True
    numeric_tokens = re.findall(r"[-$]?\d[\d,]*(?:\.\d+)?", line)
    return len(numeric_tokens) >= 3 and len(line.split()) <= 30


def split_tables_and_text(section: SectionBlock) -> list[tuple[str, str]]:
    lines = section.content.splitlines()
    chunks: list[tuple[str, str]] = []
    i = 0
    prose_buffer: list[str] = []

    while i < len(lines):
        line = lines[i]
        if is_table_line(line):
            if prose_buffer:
                chunks.append(("text", "\n".join(prose_buffer).strip()))
                prose_buffer = []

            table_lines: list[str] = []
            caption = lines[i - 1].strip() if i > 0 else ""
            if caption and caption not in table_lines and not is_table_line(caption):
                table_lines.append(caption)

            while i < len(lines) and (is_table_line(lines[i]) or not lines[i].strip()):
                table_lines.append(lines[i])
                i += 1

            # Attach likely footnotes directly following a table.
            while i < len(lines) and TABLE_FOOTNOTE_RE.match(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1

            table_text = "\n".join(table_lines).strip()
            if table_text:
                chunks.append(("table", table_text))
            continue

        prose_buffer.append(line)
        i += 1

    if prose_buffer:
        chunks.append(("text", "\n".join(prose_buffer).strip()))

    return [(chunk_type, text) for chunk_type, text in chunks if text]


def split_semantic_text(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not blocks:
        return [text.strip()] if text.strip() else []

    grouped: list[str] = []
    running: list[str] = []
    for block in blocks:
        if SUBHEADING_RE.match(block):
            if running:
                grouped.append("\n\n".join(running).strip())
                running = []
            running.append(block)
            continue
        running.append(block)

    if running:
        grouped.append("\n\n".join(running).strip())

    return [chunk for chunk in grouped if chunk]


def split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = text.rfind("\n", start, end)
            if split_at == -1 or split_at <= start + max_chars // 2:
                split_at = text.rfind(" ", start, end)
            if split_at == -1 or split_at <= start + max_chars // 2:
                split_at = end
        else:
            split_at = end

        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)
        if split_at >= len(text):
            break
        start = max(split_at - overlap_chars, start + 1)
    return chunks


def build_documents_for_report(
    report_path: Path,
    max_chars: int,
    overlap_chars: int,
) -> list[Document]:
    raw_text = read_report(report_path)
    base_metadata = extract_header_metadata(raw_text, report_path.name)
    report_body = normalize_for_chunking(strip_preface_noise(raw_text))

    sections = split_into_sections(report_body)
    documents: list[Document] = []
    chunk_index = 0

    for section_index, section in enumerate(sections):
        piece_blocks = split_tables_and_text(section)
        for block_index, (block_type, block_text) in enumerate(piece_blocks):
            semantic_units = (
                split_semantic_text(block_text) if block_type == "text" else [block_text]
            )
            for unit in semantic_units:
                for piece in split_with_overlap(unit, max_chars=max_chars, overlap_chars=overlap_chars):
                    chunk_metadata = {
                        **base_metadata,
                        "section_title": section.title,
                        "section_index": section_index,
                        "block_index": block_index,
                        "chunk_index": chunk_index,
                        "chunk_type": block_type,
                    }
                    chunk_id = _stable_chunk_id(chunk_metadata, piece)
                    chunk_metadata["chunk_id"] = chunk_id
                    documents.append(Document(page_content=piece, metadata=chunk_metadata))
                    chunk_index += 1

    return documents


def _stable_chunk_id(metadata: dict[str, str | int], content: str) -> str:
    base = (
        f"{metadata.get('file_name', '')}|"
        f"{metadata.get('chunk_index', '')}|"
        f"{metadata.get('section_title', '')}|"
        f"{content[:120]}"
    )
    digest = hashlib.sha1(base.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"{metadata.get('ticker', 'UNK')}-{metadata.get('filing_date', 'NA')}-{digest}"


def iter_report_paths(corpus_dir: Path, pattern: str, limit_files: int) -> Iterable[Path]:
    paths = sorted(corpus_dir.glob(pattern))
    if limit_files > 0:
        paths = paths[:limit_files]
    return paths


def ingest_documents(
    docs: list[Document],
    collection: str,
    embedding_model: str,
    chroma_host: str,
    chroma_port: int,
    batch_size: int,
    skip_existing: bool,
) -> None:
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    _assert_chroma_reachable(chroma_host=chroma_host, chroma_port=chroma_port)
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        host=chroma_host,
        port=chroma_port,
    )
    total = len(docs)
    started = time.time()
    print(
        f"Starting ingestion into collection '{collection}' at http://{chroma_host}:{chroma_port} "
        f"using model '{embedding_model}' (batch size: {batch_size})...",
        flush=True,
    )
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_docs = docs[start:end]
        batch_ids = [str(doc.metadata.get("chunk_id", "")) for doc in batch_docs]
        if skip_existing:
            existing = vectorstore._collection.get(ids=batch_ids, include=[])  # noqa: SLF001
            existing_ids = set(existing.get("ids") or [])
            if existing_ids:
                batch_docs = [
                    doc
                    for doc in batch_docs
                    if str(doc.metadata.get("chunk_id", "")) not in existing_ids
                ]
                batch_ids = [
                    str(doc.metadata.get("chunk_id", ""))
                    for doc in batch_docs
                ]
            if not batch_docs:
                print(
                    f"Skipped batch {start // batch_size + 1}: docs {start + 1}-{end}/{total} "
                    "(all already indexed)",
                    flush=True,
                )
                continue
        batch_started = time.time()
        texts = [doc.page_content for doc in batch_docs]
        metadatas = [doc.metadata for doc in batch_docs]
        embeddings = self_or_embed_documents(vectorstore, texts)
        vectorstore._collection.upsert(  # noqa: SLF001
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
        elapsed = time.time() - batch_started
        overall_elapsed = time.time() - started
        print(
            f"Ingested batch {start // batch_size + 1}: docs {start + 1}-{end}/{total} "
            f"(batch {elapsed:.1f}s, total {overall_elapsed:.1f}s)",
            flush=True,
        )


def self_or_embed_documents(vectorstore: Chroma, texts: list[str]) -> list[list[float]]:
    embedding_function = vectorstore._embedding_function  # noqa: SLF001
    if embedding_function is None:
        raise RuntimeError("No embedding function configured for Chroma vectorstore.")
    return embedding_function.embed_documents(texts)


def _assert_chroma_reachable(chroma_host: str, chroma_port: int) -> None:
    base_url = f"http://{chroma_host}:{chroma_port}"
    heartbeat_url = f"{base_url}/api/v2/heartbeat"
    request = Request(heartbeat_url, method="GET")
    try:
        with urlopen(request, timeout=5) as response:
            status_ok = 200 <= response.status < 300
            if not status_ok:
                raise RuntimeError(
                    f"Chroma heartbeat returned unexpected status {response.status} at {heartbeat_url}."
                )
            body = response.read().decode("utf-8", errors="ignore").strip()
            if body:
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = body
                if isinstance(payload, dict) and payload.get("detail") == "Not Found":
                    raise RuntimeError(
                        "Target service returned FastAPI-style 404 for Chroma heartbeat. "
                        f"Expected Chroma at {base_url}. Check for a port conflict."
                    )
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore").strip()
        raise RuntimeError(
            f"Could not reach Chroma heartbeat at {heartbeat_url}. "
            f"HTTP {exc.code}. Body: {body[:200]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to Chroma at {base_url}. Ensure Docker container is up "
            "and use --chroma-port if your host mapping differs."
        ) from exc


def main() -> None:
    args = parse_args()
    if not args.corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {args.corpus_dir}")

    all_docs: list[Document] = []
    report_paths = list(iter_report_paths(args.corpus_dir, args.glob, args.limit_files))
    if not report_paths:
        raise FileNotFoundError(
            f"No reports found in {args.corpus_dir} matching pattern '{args.glob}'."
        )

    for report_path in report_paths:
        docs = build_documents_for_report(
            report_path=report_path,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
        all_docs.extend(docs)
        print(f"Prepared {len(docs):>4} chunks from {report_path.name}")

    print(f"Prepared {len(all_docs)} chunks across {len(report_paths)} report files.")
    if all_docs:
        sample = all_docs[0]
        print(f"Sample metadata keys: {sorted(sample.metadata.keys())}")

    if args.dry_run:
        print("Dry run enabled; skipped embedding + Chroma ingestion.")
        return

    ingest_documents(
        docs=all_docs,
        collection=args.collection,
        embedding_model=args.embedding_model,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
    )
    print(
        f"Ingested {len(all_docs)} chunks into Chroma collection '{args.collection}' at "
        f"http://{args.chroma_host}:{args.chroma_port}"
    )


if __name__ == "__main__":
    main()
