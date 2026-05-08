from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def chat_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"page_title": "ELIZA RAG Chat", "messages": []},
    )


@app.post("/chat", response_class=HTMLResponse)
def submit_prompt(request: Request, prompt: str = Form(...)) -> HTMLResponse:
    messages = [
        {"role": "user", "content": prompt.strip()},
        {
            "role": "assistant",
            "content": "RAG pipeline is not connected yet. This is a placeholder response.",
        },
    ]
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"page_title": "ELIZA RAG Chat", "messages": messages, "prompt": prompt},
    )