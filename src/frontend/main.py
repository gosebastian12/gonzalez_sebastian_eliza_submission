from pathlib import Path
import asyncio
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from functools import lru_cache

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

from fastapi import FastAPI, Form, Request

# Load ``.env`` from cwd so ``RAG_*`` overrides apply when not exported in the shell.
if load_dotenv:
    load_dotenv()
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

app = FastAPI()
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
RAG_SERVICE_PATH = (
    Path(__file__).resolve().parents[1] / "backend" / "services" / "llm-user-prompt.py"
)


def _load_rag_service():
    spec = importlib.util.spec_from_file_location("llm_user_prompt", RAG_SERVICE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load RAG service module at {RAG_SERVICE_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _get_rag_module():
    """Import the RAG service once; it is expensive only on first load."""
    return _load_rag_service()


_rag_config_fingerprint: str | None = None
_rag_client: object | None = None


def _fingerprint_rag_config(config) -> str:
    """Stable string so env-driven ``RAG_*`` changes rebuild the client."""
    return json.dumps(asdict(config), sort_keys=True, default=str)


def _get_rag_runtime():
    """Return ``(module, EdgarHybridRAG)``. Recreates the client when ``RAGConfig.from_env()`` changes.

    Previously this used ``lru_cache`` on the whole runtime, so ``RAG_FINAL_K`` and other vars
    were frozen until process restart. Source count is capped by ``RAG_FINAL_K`` (not ``RAG_SEMANTIC_K``).
    """
    global _rag_config_fingerprint, _rag_client
    rag_module = _get_rag_module()
    config = rag_module.RAGConfig.from_env()
    fp = _fingerprint_rag_config(config)
    if fp != _rag_config_fingerprint or _rag_client is None:
        _rag_client = rag_module.EdgarHybridRAG(config)
        _rag_config_fingerprint = fp
    return rag_module, _rag_client


@app.get("/", response_class=HTMLResponse)
def chat_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"page_title": "ELIZA RAG Chat", "messages": []},
    )


@app.post("/chat", response_class=HTMLResponse)
async def submit_prompt(request: Request, prompt: str = Form(...)) -> HTMLResponse:
    prompt_clean = prompt.strip()
    assistant_text = "RAG pipeline is not connected yet. This is a placeholder response."
    source_chunks: list[dict[str, str]] | None = None
    rag_finished_ok = False
    timeout_seconds = float(os.environ.get("RAG_HTTP_TIMEOUT", "180"))

    if prompt_clean:
        try:
            started = time.time()
            _, rag = _get_rag_runtime()
            result = await asyncio.wait_for(
                run_in_threadpool(rag.answer, prompt_clean),
                timeout=timeout_seconds,
            )
            rag_finished_ok = True
            assistant_text = str(result.get("answer", "")).strip() or assistant_text
            elapsed = time.time() - started
            assistant_text = f"{assistant_text}\n\n(Response time: {elapsed:.1f}s)"
            source_chunks = result.get("source_chunks")
            if not source_chunks and result.get("sources"):
                source_chunks = [
                    {"label": label, "content": "(Chunk text unavailable for this response.)"}
                    for label in result["sources"]
                ]
            source_chunks = source_chunks or []
        except (TimeoutError, asyncio.TimeoutError):
            assistant_text = (
                "RAG pipeline timeout: retrieval + generation exceeded "
                f"{timeout_seconds:.0f}s (set RAG_HTTP_TIMEOUT to raise this limit). "
                "Try smaller models (RAG_LLM_MODEL / RAG_EMBEDDING_MODEL), lower "
                "RAG_SEMANTIC_K / RAG_FINAL_K / RAG_NUM_PREDICT / RAG_MAX_CONTEXT_CHARS, "
                "or verify Ollama and Chroma are responsive."
            )
        except Exception as exc:
            assistant_text = f"RAG pipeline error: {exc}"

    assistant_message: dict[str, object] = {
        "role": "assistant",
        "content": assistant_text,
    }
    if rag_finished_ok and source_chunks is not None:
        assistant_message["source_chunks"] = source_chunks

    messages = [
        {"role": "user", "content": prompt_clean},
        assistant_message,
    ]
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"page_title": "ELIZA RAG Chat", "messages": messages, "prompt": prompt},
    )