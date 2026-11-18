"""FastAPI backend for the CHRIST University AI Student Helpdesk.

Run from the christ_helpdesk folder:
    python -m uvicorn backend.main:app --port 8000
then open http://localhost:8000
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import run_evaluation  # noqa: E402
from src.model1_llm import PlainLLMBot  # noqa: E402
from src.model2_rag import RAGBot, load_documents  # noqa: E402
from src.model3_llmops import LLMOpsRAGBot, compute_metrics  # noqa: E402
from src.utils import load_config, path  # noqa: E402

STATE: dict = {}
EVAL_LOCK = threading.Lock()
LLM_LOCK = threading.Lock()  # one LLM request at a time keeps latency comparable on a laptop
EVAL_JOB = {"status": "idle", "done": 0, "total": 0, "message": "", "error": None,
            "started": None, "finished": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    print(f"[startup] provider={cfg['llm']['provider']} model={cfg['llm']['model']}")
    m2 = RAGBot(cfg, prompt_version="v1")
    m3 = LLMOpsRAGBot(cfg, index=m2.index, verbose=False)
    m3.channel = "web"
    STATE.update(cfg=cfg, bots={"llm": PlainLLMBot(cfg), "rag": m2, "llmops": m3})
    print("[startup] three models loaded")
    yield


app = FastAPI(title="CHRIST University AI Student Helpdesk", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    model: Literal["llm", "rag", "llmops"] = "llmops"


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=1, max_length=64)
    helpful: bool


def _ask(key: str, question: str) -> dict:
    q = question.strip()
    if not q:
        raise HTTPException(422, "Question must not be empty.")
    bot = STATE["bots"][key]
    with LLM_LOCK:
        r = bot.answer(q)
    return {"model_key": key, "model": bot.name, **r}


@app.get("/api/health")
def health():
    cfg = STATE["cfg"]
    bots = STATE["bots"]
    client = bots["llm"].client
    reachable = client.is_reachable()
    return {
        "status": "ok",
        "provider": cfg["llm"]["provider"],
        "model": cfg["llm"]["model"],
        "llm_reachable": reachable,
        "model_available": client.has_model() if reachable else False,
        "embeddings": bots["rag"].index.embedder.label,
        "chunks": len(bots["rag"].index.chunks),
        "prompt_version": cfg["llmops"]["prompt_version"],
        "top_k": cfg["rag"]["top_k"],
        "min_score": cfg["rag"]["min_score"],
        "chunk_size": cfg["rag"]["chunk_size"],
        "chunk_overlap": cfg["rag"]["chunk_overlap"],
        "temperature": cfg["llm"]["temperature"],
        "models": {k: {"name": b.name, "tagline": b.tagline} for k, b in bots.items()},
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    return _ask(req.model, req.question)


@app.post("/api/compare")
def compare(req: CompareRequest):
    # sequential on purpose, so each model's latency is measured without contention
    return {"question": req.question, "results": [_ask(k, req.question) for k in ("llm", "rag", "llmops")]}


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    if not STATE["bots"]["llmops"].feedback(req.trace_id, req.helpful):
        raise HTTPException(404, "Unknown trace id.")
    return {"ok": True, "trace_id": req.trace_id, "feedback": 1 if req.helpful else -1}


@app.get("/api/metrics")
def metrics():
    traces = STATE["bots"]["llmops"].store.fetch()
    m = compute_metrics(traces)
    m["traces"] = traces[:200]
    return m


def _eval_worker():
    def progress(done, total, msg):
        EVAL_JOB.update(done=done, total=total, message=msg)
    try:
        b = STATE["bots"]
        # a separate Model 3 instance: its own (cleared) cache, traces tagged channel="eval",
        # and the web chat keeps its cache and stays usable while the benchmark runs
        m3_eval = LLMOpsRAGBot(STATE["cfg"], index=b["rag"].index, verbose=False)
        run_evaluation([b["llm"], b["rag"], m3_eval], progress=progress)
        EVAL_JOB.update(status="done")
    except Exception as e:  # surfaced to the UI
        EVAL_JOB.update(status="error", error=f"{type(e).__name__}: {e}")
    finally:
        EVAL_JOB["finished"] = time.time()


@app.post("/api/evaluation/run")
def evaluation_run():
    with EVAL_LOCK:
        if EVAL_JOB["status"] == "running":
            return {"started": False, "job": EVAL_JOB}
        EVAL_JOB.update(status="running", done=0, total=0, message="starting", error=None,
                        started=time.time(), finished=None)
        threading.Thread(target=_eval_worker, daemon=True).start()
    return {"started": True, "job": EVAL_JOB}


@app.get("/api/evaluation")
def evaluation():
    f = path("results", "evaluation.json")
    result = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    return {"job": EVAL_JOB, "result": result}


@app.get("/api/kb")
def kb():
    files = []
    for name, text in load_documents(STATE["cfg"]["rag"]["kb_dir"]):
        title = text.splitlines()[0].lstrip("# ").strip() if text else name
        qs = re.findall(r"(?m)^## (.+)$", text)
        files.append({"file": name, "title": title, "questions": qs})
    return {"files": files}


FRONTEND = ROOT / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
