from pathlib import Path
import importlib.util
import sys

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

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


@app.get("/", response_class=HTMLResponse)
def chat_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"page_title": "ELIZA RAG Chat", "messages": []},
    )


@app.post("/chat", response_class=HTMLResponse)
def submit_prompt(request: Request, prompt: str = Form(...)) -> HTMLResponse:
    prompt_clean = prompt.strip()
    assistant_text = "RAG pipeline is not connected yet. This is a placeholder response."

    if prompt_clean:
        try:
            rag_module = _load_rag_service()
            config = rag_module.RAGConfig()
            rag = rag_module.EdgarHybridRAG(config)
            result = rag.answer(prompt_clean)
            assistant_text = str(result.get("answer", "")).strip() or assistant_text
            sources = result.get("sources", [])
            if sources:
                source_lines = "\n".join(f"- {source}" for source in sources[:4])
                assistant_text = f"{assistant_text}\n\nSources:\n{source_lines}"
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