# CHRIST University – AI Student Helpdesk (LLM · RAG · LLMOps)

An MSc AI/ML LCA project (Experiential and Problem-Based Learning). It builds three helpdesk chatbots for CHRIST University students and compares them. Everything runs **locally** (Ollama + sentence-transformers), so no student data leaves the machine.

| # | Model | How it answers | What it demonstrates |
|---|---|---|---|
| 1 | **Plain LLM** (`src/model1_llm.py`) | `llama3.2:3b` answers from its own memory with a simple system prompt | **Hallucination**: fluent but wrong local facts |
| 2 | **RAG** (`src/model2_rag.py`) | Retrieves the top-k FAQ chunks (MiniLM embeddings, cosine search) and passes them to the LLM with a naive prompt (v1) | **Grounding**: correct, cited answers, but no safety layer |
| 3 | **RAG + LLMOps** (`src/model3_llmops.py`) | RAG with a hardened prompt (v2), PII redaction, injection guard, cache, relevance gate, output guardrails, SQLite tracing and feedback | **Reliability and governance** |

## Folder structure

```
christ_helpdesk/
├── config.yaml                 # LLM, embeddings, RAG and LLMOps settings
├── prompts/prompts.yaml        # prompt registry: baseline, rag.v1 (naive), rag.v2 (hardened)
├── data/kb/*.md                # knowledge base: 8 files, 50 FAQs from christuniversity.in
├── data/eval/eval_set.jsonl    # 17 test questions (in-scope, out-of-scope, attack, PII)
├── src/utils.py                # config (+ env overrides), prompts, paths, terminal chat loop
├── src/llm_client.py           # Ollama /api/chat client (text, latency, tokens)
├── src/model1_llm.py           # Model 1
├── src/model2_rag.py           # Model 2 (chunking, Embedder, VectorIndex, RAGBot)
├── src/model3_llmops.py        # Model 3 (guardrails, cache, tracing, feedback, KPIs)
├── evaluate.py                 # benchmark -> results/evaluation.json, eval_results.csv, summary.md
├── monitor.py                  # terminal LLMOps report from logs/traces.db
├── backend/main.py             # FastAPI API + serves the frontend
├── frontend/                   # index.html, styles.css, app.js (plain HTML/CSS/JS, no CDN)
├── notebook/CHRIST_Helpdesk_LLM_RAG_LLMOps.ipynb   # self-contained notebook (Jupyter / Colab)
├── results/                    # benchmark outputs
├── logs/traces.db              # created on first Model 3 request
├── requirements.txt
├── README.md
└── REPORT.md
```

## Setup (Windows 11, no virtual environment)

Your Device Guard policy blocks `.exe` files inside virtual environments, so install packages for your user only and always start tools with `python -m`.

1. Install the model in Ollama (once, about 2 GB):
   ```bash
   ollama pull llama3.2:3b
   ```
2. Install the Python packages (from the `christ_helpdesk` folder):
   ```bash
   python -m pip install --user -r requirements.txt
   ```
   The first run downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB) to the Hugging Face cache. After that it loads from disk. If it cannot load, the code falls back to TF-IDF automatically and prints which one it uses.
3. Make sure Ollama is running (the tray app, or `ollama serve` in a terminal).

Run every command below from the `christ_helpdesk` folder.

## Run each model in the terminal

```bash
python -m src.model1_llm
```
```bash
python -m src.model2_rag
```
```bash
python -m src.model3_llmops
```
Type a question and press Enter. Type `exit` to quit. Model 2 prints its sources and similarity scores, and Model 3 also prints the trace id, cache status and guardrail flags.

**Offline mode** (no Ollama needed): Model 1 replies `[offline] no LLM`, and Models 2 and 3 return the best retrieved FAQ paragraph or abstain. In PowerShell:
```powershell
$env:LLM_PROVIDER = "offline"; python -m src.model3_llmops
```
Remove it again with `Remove-Item Env:LLM_PROVIDER`. In cmd.exe use `set LLM_PROVIDER=offline`. To try another model without editing the config, set `$env:LLM_MODEL = "llama3.2:1b"`.

## Benchmark and monitoring

```bash
python evaluate.py
```
This runs all 17 questions through the three models and writes `results/evaluation.json`, `results/eval_results.csv` and `results/summary.md`. With Ollama on a laptop CPU it takes a few minutes, and offline it takes a few seconds.

```bash
python monitor.py
```
This prints the LLMOps report from `logs/traces.db`: requests, average and p95 latency, cache-hit %, abstain %, blocked injections, tokens, feedback, flag counts, low-relevance questions and recent traces. Use `python monitor.py --last 50` to limit it to recent traces.

## Web app

```bash
python -m uvicorn backend.main:app --port 8000
```
Then open **http://localhost:8000**. Startup takes about 10–20 s while the embedding model loads.

| Page | What it shows |
|---|---|
| **Model 1: Plain LLM** | Chat with latency and tokens, a hallucination warning banner, and the pipeline (Question → LLM → Answer) |
| **Model 2: RAG** | Chat plus a collapsible "Retrieved sources" list per answer (file, similarity bar, chunk text), the pipeline (Docs → Chunk → Embed → Vector index → Top-k → LLM), live settings, and the KB file list (click a question to ask it) |
| **Model 3: RAG + LLMOps** | Chat with guardrail flag chips (red = blocked, amber = warning), cache badge, prompt version, trace id and 👍/👎 feedback. The monitoring dashboard below it has KPI tiles, a latency chart, flag bars, low-relevance questions ("add these to the KB") and recent traces |
| **Comparison** | Live side-by-side answers with preset questions (in-scope, IPL, injection, phone number), a **Run benchmark** button with a progress bar, a grouped bar chart, a metrics table, takeaway cards and a question-by-question table (click a row for the answers). It also has "What each layer adds" cards |

API: `GET /api/health`, `POST /api/chat {question, model: llm|rag|llmops}`, `POST /api/compare {question}`, `POST /api/feedback {trace_id, helpful}`, `GET /api/metrics`, `POST /api/evaluation/run`, `GET /api/evaluation`, `GET /api/kb`. Interactive docs are at http://localhost:8000/docs.

The web app uses relative API URLs, so it also works behind the Google Colab proxy.

## Notebook

`notebook/CHRIST_Helpdesk_LLM_RAG_LLMOps.ipynb` is self-contained, with the KB and eval set embedded in its cells. Open it in Jupyter or upload it to Google Colab and run all cells. In Colab, the first cell installs Ollama, starts it and pulls the model. To run it without an LLM, set `LLM_PROVIDER = "offline"` in the setup cell.

## Experiments to try

| Change | Where | What to look for |
|---|---|---|
| Prompt version v1 vs v2 for Model 3 | `config.yaml → llmops.prompt_version` | Citations, abstentions, safe-handling rate |
| `top_k` 1 / 3 / 5 | `config.yaml → rag.top_k` | hit@k vs tokens and latency, distraction from extra chunks |
| `chunk_size` 300 / 700 / 1500 | `config.yaml → rag.chunk_size` | Retrieval precision vs context completeness |
| Model size | `$env:LLM_MODEL = "llama3.2:1b"` (pull it first) or `llama3.1:8b` | Accuracy vs latency trade-off |
| `min_score` 0.1 / 0.2 / 0.35 | `config.yaml → rag.min_score` | False abstentions vs off-topic answers |
| Embeddings | `$env:EMBEDDINGS = "tfidf"` | Dense vs lexical retrieval quality |
| Guardrails / cache off | `config.yaml → llmops.enable_guardrails / enable_cache` | What breaks without LLMOps |

Re-run `python evaluate.py` after each change and compare `results/summary.md`. Copy it first if you want to keep the previous run.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip.exe` / `uvicorn.exe` blocked by Device Guard / "This app has been blocked" | Don't use a venv and don't call the .exe launchers. Always use `python -m pip ...`, `python -m uvicorn ...` |
| Status pill says **Ollama not reachable** / `[error] LLM unreachable` | Start Ollama (tray app or `ollama serve`) and check http://localhost:11434 in a browser. Or use offline mode |
| Pill says **model not pulled** / HTTP 404 "model not found" | `ollama pull llama3.2:3b` (or match `llm.model` in `config.yaml` to `ollama list`) |
| First answer takes 30–90 s | Ollama loads the model into memory on the first request. Later answers take a few seconds |
| Answers are slow in general | Use `llama3.2:1b`, lower `max_tokens`, close other apps, or keep `top_k` at 3 |
| `[embeddings] using TF-IDF` | sentence-transformers is missing or the first download failed (no internet). Run `python -m pip install --user sentence-transformers` once while online |
| Port 8000 already in use | `python -m uvicorn backend.main:app --port 8010` |
| `ModuleNotFoundError: src` | Run the commands from inside the `christ_helpdesk` folder |

> The KB is a student-project summary of public christuniversity.in pages. Some facts vary by campus or programme and change every year, so verify important details with the office concerned.
