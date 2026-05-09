from pathlib import Path
import asyncio
import importlib.util
import os
import sys
import time
from functools import lru_cache

from fastapi import FastAPI, Form, Request
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
def _get_rag_runtime():
    rag_module = _load_rag_service()
    config = rag_module.RAGConfig.from_env()
    return rag_module, rag_module.EdgarHybridRAG(config)


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
    timeout_seconds = float(os.environ.get("RAG_HTTP_TIMEOUT", "180"))

    if prompt_clean:
        try:
            started = time.time()
            _, rag = _get_rag_runtime()
            result = await asyncio.wait_for(
                run_in_threadpool(rag.answer, prompt_clean),
                timeout=timeout_seconds,
            )
            assistant_text = str(result.get("answer", "")).strip() or assistant_text
            sources = result.get("sources", [])
            if sources:
                source_lines = "\n".join(f"- {source}" for source in sources)
                assistant_text = f"{assistant_text}\n\nSources:\n{source_lines}"
            elapsed = time.time() - started
            assistant_text = f"{assistant_text}\n\n(Response time: {elapsed:.1f}s)"
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

    messages = [
        {"role": "user", "content": prompt_clean},
        {
            "role": "assistant",
            "content": assistant_text,
        },
    ]
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"page_title": "ELIZA RAG Chat", "messages": messages, "prompt": prompt},
    )