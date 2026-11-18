# AI-Powered Student Helpdesk for CHRIST University using LLM, RAG and LLMOps

**Course:** MSc AI/ML · **Activity:** LCA – Experiential and Problem-Based Learning
**Stack:** Python 3 · Ollama (`llama3.2:3b`) · sentence-transformers (`all-MiniLM-L6-v2`) · NumPy · FastAPI · SQLite · plain HTML/CSS/JS. Everything runs locally.

---

## 1. Problem statement

Students at CHRIST University ask the same operational questions again and again: minimum attendance, leave deadlines, hall tickets, revaluation, counselling hours, scholarships, fee payments and admissions. The answers are spread across many web pages and circulars, so offices handle repetitive queries and students sometimes act on wrong information from peers or from general-purpose chatbots.

**Goal:** design and build a helpdesk chatbot that answers these questions **correctly, with sources, safely and privately**. Implement three models locally (a plain LLM, RAG, and RAG + LLMOps), measure them on the same test set, and explain why each layer matters in GenAI applications.

## 2. Architecture

```
                         ┌───────────────────────── Model 1: Plain LLM ─────────────────────────┐
  Student question ───►  │  baseline prompt ──► Ollama llama3.2:3b ──► answer                    │
                         └────────────────────────────────────────────────────────────────────────┘

                         ┌───────────────────────── Model 2: RAG ────────────────────────────────┐
  data/kb/*.md ──► chunk by "## FAQ" (+title) ──► embed (MiniLM, normalised) ──► NumPy index      │
  Student question ──► embed ──► cosine top-k (k=3) ──► prompt v1 "[source] context" ──► LLM ──► answer + sources
                         └────────────────────────────────────────────────────────────────────────┘

                         ┌───────────────────────── Model 3: RAG + LLMOps ───────────────────────┐
  Student question                                                                               │
     │ 1 PII redaction (phone / email / Aadhaar) ──► flags                                       │
     │ 2 Prompt-injection guard ──────────────► refuse (LLM never called)                        │
     │ 3 Cache (normalised question) ─────────► instant answer, 0 tokens                         │
     │ 4 Retrieval + relevance gate ──────────► abstain if top score < min_score (0.20)          │
     │ 5 LLM with hardened prompt v2 ──► output guardrails (prompt leak, missing citation)       │
     │ 6 Trace to SQLite (latency, tokens, flags, sources) ◄── 7 user feedback 👍/👎             │
     ▼                                                                                           │
  answer + sources + flags + trace_id ──► monitoring dashboard / monitor.py / evaluate.py        │
                         └────────────────────────────────────────────────────────────────────────┘
```

## 3. Comparison of the three models

| Aspect | Model 1: Plain LLM | Model 2: RAG | Model 3: RAG + LLMOps |
|---|---|---|---|
| Knowledge source | Model weights only | Official KB (50 FAQs) | Official KB (50 FAQs) |
| Prompt | `baseline` | `rag.v1` (naive) | `rag.v2` (hardened: context-only, exact abstain phrase, citations, no prompt disclosure) |
| Citations | No | Context includes `[source]` | Required, checked by a guardrail |
| Out-of-scope handling | Answers anyway | Usually answers anyway | Relevance gate abstains |
| Prompt injection | Vulnerable | Vulnerable | Blocked before the LLM |
| PII | Sent to the LLM | Sent to the LLM | Redacted before LLM, cache and logs |
| Cost per query | Lowest tokens | More prompt tokens (context) | Like RAG, 0 for cache hits and blocked requests |
| Observability | None | None | Traces, KPIs, feedback, low-relevance list |

## 4. Implementation summary

- **Knowledge base (`data/kb/`)**: 8 markdown files with 50 FAQs, rewritten in our own words from official christuniversity.in pages: general regulations, counselling centre, campus life, Office of Examinations notices, scholarship pages, the payment portal, and contact details. Every answer ends with a `(Source: URL)` line.
- **Chunking**: one `## question` plus its answer forms one chunk, prefixed with the document title. Sections longer than 700 characters are split with a 100-character overlap.
- **Embeddings and index**: `all-MiniLM-L6-v2` (normalised, 384-d) in an in-memory NumPy matrix, searched by cosine similarity (a dot product). TF-IDF (1–2-grams) is the automatic fallback.
- **LLM**: Ollama `/api/chat` (non-streaming) with temperature 0.1 and a 300-token output limit. The client also returns latency and the prompt and completion token counts.
- **LLMOps**: regex PII redaction; pattern-based injection detection; a dictionary cache keyed on the normalised question; a relevance gate at `min_score = 0.20`; output guardrails (6-word shingle overlap with the system prompt, citation check); a thread-safe SQLite trace store; and ±1 feedback.
- **Prompt registry**: `prompts/prompts.yaml` versions the prompts, and `llmops.prompt_version` selects the one in use.
- **Evaluation (`evaluate.py`)**: 17 questions (14 in-scope, 1 out-of-scope, 1 attack, 1 PII). Metrics: answer accuracy (all expected keywords present), retrieval hit@k, groundedness (share of answer content words found in the retrieved context; Model 1 is scored against the context RAG would retrieve), safe-handling rate, PII-redacted rate, average latency and average tokens.
- **Application**: a FastAPI backend and a 4-page web UI (three model chats, a monitoring dashboard, and live comparison plus benchmark), plus terminal chats, `monitor.py`, and a self-contained notebook.

## 5. Results

### 5.1 Offline pipeline check (`LLM_PROVIDER=offline`, no LLM)

In offline mode Model 1 has no LLM, and Models 2 and 3 return the best retrieved FAQ paragraph. This checks retrieval, guardrails and the evaluation itself, **not** LLM answer quality.

| Metric | Model 1: Plain LLM | Model 2: RAG | Model 3: RAG + LLMOps |
|---|---|---|---|
| Answer accuracy | 0.0% | 100.0% | 100.0% |
| Retrieval hit@k | n/a | 100.0% | 100.0% |
| Groundedness | 6.0% | 100.0% | 100.0% |
| Safe handling | 0.0% | 50.0% | 100.0% |
| PII redacted | 0.0% | 0.0% | 100.0% |
| Avg latency | 0.00 s | 0.01 s | 0.01 s |
| Avg tokens | 0 | 0 | 0 |

Reading: retrieval finds the correct source file for every in-scope question (hit@3 = 100%). Model 2 still returns an unrelated FAQ for the injection attempt, so its safe handling is 50%. Model 3 blocks the attack, abstains on the IPL question and redacts the phone number.

### 5.2 Results with Ollama (`llama3.2:3b`)

Settings: MiniLM embeddings, top_k = 3, min_score = 0.20, prompt v2 for Model 3, 17 questions. These are the numbers from `results/summary.md`.

| Metric | Model 1: Plain LLM | Model 2: RAG | Model 3: RAG + LLMOps |
|---|---|---|---|
| Answer accuracy | 20.0% | **100.0%** | **100.0%** |
| Retrieval hit@k | n/a | **100.0%** | **100.0%** |
| Groundedness | 35.2% | 88.0% | **88.6%** |
| Safe handling | 50.0% | 50.0% | **100.0%** |
| PII redacted | 0.0% | 0.0% | **100.0%** |
| Avg latency | 4.61 s | 2.71 s | **2.67 s** |
| Avg tokens | **248** | 373 | 487 |

Hardware used: AMD Ryzen 7 260 (Radeon 780M) with an NVIDIA GeForce RTX 5050 Laptop GPU, 24 GB RAM, Windows 11 · Date of run: 18 November 2026

**Reading the results:**

- **Model 1 (plain LLM)** answered only 3 of the 15 in-scope and PII questions correctly. It gave 75% instead of 85% for the attendance rule and invented library hours. One of its 3 "correct" answers was a lucky keyword match: "9:00" appears only because it made up an opening time of 9:00 AM.
- **RAG (Models 2 and 3)** lifted accuracy by 80 points and groundedness by about 53 points. Retrieval found the right source file for every question.
- **The injection attack** broke both Model 1 and Model 2. Only Model 3 refused it and kept the phone number away from the LLM.
- **Token cost:** RAG uses about 1.5× the plain LLM's tokens, and Model 3 about 2× because of its longer v2 instructions. Both RAG models were still faster on average than Model 1, which writes longer answers.

## 6. Interpretation: why each model matters in GenAI applications

*This is the expected outcome. Verify each statement against your own numbers in 5.2 and edit it where your results differ.*

### Model 1: Plain LLM shows hallucination
The 3B model writes confident, well-formed answers, but it has never seen CHRIST's rules. Expect wrong thresholds (for example "75%" instead of 85% attendance), invented URLs or email domains (for example `christuniversity.ac.in`), and generic exam procedures. Accuracy and groundedness should be the lowest of the three, and it may obey the injection prompt. **Why it matters:** a fluent answer is not a correct answer. In any domain-specific GenAI application, a bare LLM must not be treated as a source of truth.

### Model 2: RAG provides grounding
Retrieval gives the model the right paragraph, so accuracy and groundedness should rise sharply, typically by 50+ percentage points over Model 1, and answers can point to a source. Knowledge is updated by editing a markdown file, with no retraining. Costs: several times more prompt tokens per query (the context) and somewhat higher latency. Because the naive prompt never tells the model to stay inside the context or to refuse, Model 2 still answers out-of-scope questions, may follow injected instructions, and sends phone numbers to the LLM. **Why it matters:** RAG is the standard way to connect LLMs to private, changing organisational knowledge, but it solves *knowledge*, not *safety*.

### Model 3: RAG + LLMOps provides reliability and governance
Accuracy should match or slightly exceed Model 2. The hardened prompt makes answers shorter and cited, but a small model sometimes abstains too cautiously. The differences show up in safety and operations: 100% safe handling of the attack and out-of-scope questions, PII redacted before it reaches the LLM or the logs, zero-token cache hits, explicit abstention when retrieval is weak, and a trace for every answer with user feedback. **Why it matters:** production GenAI systems are judged on trust, privacy, cost and maintainability as well as accuracy. LLMOps turns a demo into a service that can be monitored, audited and improved.

## 7. Recommendations

1. **Deploy Model 3.** Use only the RAG + LLMOps design for a student-facing helpdesk. Models 1 and 2 are not safe to expose.
2. **Data ownership.** Each office (Examinations, CCHS, Scholarship Cell, Admissions, Accounts) owns its KB file, reviews it every semester and at each new circular, and keeps the source URL on every answer.
3. **Feedback loop.** Review 👎 answers and the "low-relevance questions" list weekly, add missing FAQs to the KB, and correct wrong ones.
4. **Evaluation-gated releases.** Treat prompts, KB and model versions like code. Re-run the benchmark (extended to 100+ questions) on every change and block a release if accuracy or safe handling drops.
5. **Tune `min_score`** on a larger labelled set. Too high and the bot abstains on valid questions; too low and it answers off-topic ones. Review the abstain rate on the dashboard.
6. **Scale-up path:** a persistent vector DB (FAISS / Chroma / pgvector); hybrid BM25 + dense retrieval for exact terms like "CIA" or "KP"; a cross-encoder re-ranker; RAGAS or LLM-as-judge evaluation (faithfulness, answer relevance); multilingual support (Kannada, Hindi, Malayalam); and **human hand-off** that routes unanswered questions to the right office's ticket queue.
7. **Keep sensitive topics human.** For counselling, mental health, harassment, medical or disciplinary matters, the bot should only share CCHS contact details, timings and emergency numbers, and hand over to a person. It must not give advice.
8. **On-campus hosting and privacy.** Host the LLM and embeddings on university servers, keep traces redacted with a defined retention period, restrict dashboard access, and align with India's **Digital Personal Data Protection (DPDP) Act, 2023** (purpose limitation, data minimisation, consent and notice).

## 8. Limitations

- A small knowledge base (50 FAQs) and test set (17 questions), so the results are indicative and not statistically significant.
- Keyword-based accuracy can mark correct paraphrases as wrong (or lucky matches as right). Groundedness is lexical overlap, not semantic entailment.
- Some KB facts (hall-ticket steps, library hours, the selection process, CIA/ESE split) vary by campus or programme and change each year. Some came from the project brief because the official page did not show them when fetched.
- Regex guardrails catch common patterns only. Paraphrased jailbreaks and other PII formats can get through. A classifier-based guard would be stronger.
- Even with the right context, a 3B model can add wrong reasoning on top of a correct fact (for example, miscalculating a leave deadline after correctly quoting "within 10 days"). Keyword accuracy does not catch this, so human review of samples is still needed.
- A 3B model on a laptop CPU is slow (seconds per answer, longer on the first call), and latency figures depend on hardware.
- The cache is in memory and exact-match only (after normalisation). A semantic cache would raise the hit rate.

## 9. Conclusion

The three models show the layers every GenAI application needs. The **LLM** supplies language ability but hallucinates institution-specific facts. **RAG** supplies grounded, citable and easily updated knowledge. **LLMOps** supplies the guardrails, privacy, cost control, observability and feedback that make the system trustworthy and maintainable. For CHRIST University, the recommended solution is a locally hosted RAG + LLMOps helpdesk with office-owned knowledge, evaluation-gated updates and a human hand-off for sensitive cases.
