"""Model 3: RAG + LLMOps.

Wraps the RAG bot (hardened prompt v2) with: PII redaction, prompt-injection
detection, response cache, relevance gate (abstain), output guardrails,
SQLite tracing and user feedback.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

from .model2_rag import ABSTAIN_MSG, RAGBot, VectorIndex
from .utils import chat_loop, load_config, path

REFUSAL_MSG = "I can only help with CHRIST University student support questions."

# ---------------------------------------------------------------- PII redaction
PII_PATTERNS = [  # order matters: +91 phones, then Aadhaar-like 12 digits, then 10-digit mobiles
    ("pii_email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<EMAIL_REDACTED>"),
    ("pii_phone", re.compile(r"(?<!\d)(?:\+|00)91[\s-]?[6-9]\d{4}[\s-]?\d{5}(?!\d)"), "<PHONE_REDACTED>"),
    ("pii_aadhaar", re.compile(r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"), "<AADHAAR_REDACTED>"),
    ("pii_phone", re.compile(r"(?<!\d)0?[6-9]\d{4}[\s-]?\d{5}(?!\d)"), "<PHONE_REDACTED>"),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    for flag, rx, repl in PII_PATTERNS:
        text, n = rx.subn(repl, text)
        if n and flag not in flags:
            flags.append(flag)
    return text, flags


# ---------------------------------------------------------------- prompt injection
INJECTION_PATTERNS = [re.compile(p, re.I) for p in [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|your)\s+(instructions|rules)",
    r"(reveal|print|show|repeat|display|output|tell me)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions|rules)",
    r"system\s+prompt",
    r"you\s+are\s+now\b",
    r"developer\s+mode",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"act\s+as\s+(an?\s+)?(unrestricted|unfiltered)",
    r"pretend\s+(you\s+are|to\s+be)\s+",
]]


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


def normalise_question(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s<>@.%:]", " ", q.lower())).strip()


CITATION_RX = re.compile(r"\[[^\]]+\.(md|txt)\]", re.I)
FLAG_LABELS = {
    "pii_phone": "Phone number redacted", "pii_email": "Email redacted", "pii_aadhaar": "Aadhaar redacted",
    "prompt_injection": "Prompt injection blocked", "low_relevance_abstain": "Low relevance - abstained",
    "output_prompt_leak": "Prompt leak blocked", "missing_citation": "Missing citation",
    "cache_hit": "Cache hit", "llm_error": "LLM error",
}


# ---------------------------------------------------------------- tracing
class TraceStore:
    COLUMNS = ["id", "ts", "model", "prompt_version", "question", "answer", "sources", "top_score",
               "latency_s", "prompt_tokens", "completion_tokens", "cache_hit", "flags", "feedback", "channel"]

    def __init__(self, db_path: str):
        p = path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(p)
        self.lock = threading.Lock()
        with self.lock, self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY, ts TEXT, model TEXT, prompt_version TEXT, question TEXT,
                answer TEXT, sources TEXT, top_score REAL, latency_s REAL, prompt_tokens INTEGER,
                completion_tokens INTEGER, cache_hit INTEGER, flags TEXT, feedback INTEGER,
                channel TEXT)""")

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def log(self, rec: dict) -> None:
        row = [rec.get(k) for k in self.COLUMNS]
        with self.lock, self._conn() as c:
            c.execute(f"INSERT INTO traces VALUES ({','.join('?' * len(self.COLUMNS))})", row)

    def set_feedback(self, trace_id: str, value: int) -> bool:
        with self.lock, self._conn() as c:
            cur = c.execute("UPDATE traces SET feedback=? WHERE id=?", (value, trace_id))
            return cur.rowcount > 0

    def fetch(self, limit: int | None = None) -> list[dict]:
        with self.lock, self._conn() as c:
            c.row_factory = sqlite3.Row
            sql = "SELECT * FROM traces ORDER BY ts DESC" + (f" LIMIT {int(limit)}" if limit else "")
            rows = [dict(r) for r in c.execute(sql)]
        for r in rows:
            r["sources"] = json.loads(r["sources"] or "[]")
            r["flags"] = json.loads(r["flags"] or "[]")
            r["cache_hit"] = bool(r["cache_hit"])
        return rows


# ---------------------------------------------------------------- the bot
class LLMOpsRAGBot:
    name = "Model 3: RAG + LLMOps"
    tagline = "RAG with guardrails, caching, tracing and feedback"

    def __init__(self, cfg: dict | None = None, index: VectorIndex | None = None, verbose: bool = True):
        self.cfg = cfg or load_config()
        ops = self.cfg["llmops"]
        self.prompt_version = ops.get("prompt_version", "v2")
        self.rag = RAGBot(self.cfg, prompt_version=self.prompt_version, index=index, verbose=verbose)
        self.min_score = float(self.cfg["rag"]["min_score"])
        self.enable_cache = bool(ops.get("enable_cache", True))
        self.enable_guardrails = bool(ops.get("enable_guardrails", True))
        self.store = TraceStore(ops.get("trace_db", "logs/traces.db"))
        self.cache: dict[str, dict] = {}
        self.channel = "app"
        # 6-word shingles of the system prompt, to detect leakage in outputs
        words = re.findall(r"\w+", self.rag.system_prompt.lower())
        self._prompt_shingles = {" ".join(words[i:i + 6]) for i in range(len(words) - 5)}

    def clear_cache(self) -> None:
        self.cache.clear()

    def _leaks_prompt(self, answer: str) -> bool:
        words = re.findall(r"\w+", answer.lower())
        return any(" ".join(words[i:i + 6]) in self._prompt_shingles for i in range(len(words) - 5))

    def answer(self, question: str) -> dict:
        t0 = time.perf_counter()
        flags: list[str] = []
        hits: list[dict] = []
        tokens = (0, 0)
        cache_hit = False

        # 1. PII redaction (the LLM, the cache and the trace only ever see the redacted text)
        q, pii_flags = redact_pii(question) if self.enable_guardrails else (question, [])
        flags += pii_flags
        key = normalise_question(q)

        # 2. prompt-injection detection -> refuse without calling the LLM
        if self.enable_guardrails and detect_injection(q):
            flags.append("prompt_injection")
            answer = REFUSAL_MSG
        # 3. cache
        elif self.enable_cache and key in self.cache:
            c = self.cache[key]
            cache_hit = True
            answer, hits = c["answer"], c["hits"]
            flags += [f for f in c["flags"] if f not in flags]
        else:
            hits = self.rag.retrieve(q)
            # 4. relevance gate
            if not hits or hits[0]["score"] < self.min_score:
                flags.append("low_relevance_abstain")
                answer = ABSTAIN_MSG
            else:
                # 5. LLM call + output guardrails
                out = self.rag.generate(q, hits)
                answer = out["text"]
                tokens = (out["prompt_tokens"], out["completion_tokens"])
                if out.get("error"):
                    flags.append("llm_error")
                elif self.enable_guardrails:
                    if self._leaks_prompt(answer):
                        flags.append("output_prompt_leak")
                        answer = REFUSAL_MSG
                    elif ABSTAIN_MSG.lower()[:30] not in answer.lower() and not CITATION_RX.search(answer):
                        flags.append("missing_citation")
            if self.enable_cache and "llm_error" not in flags:
                self.cache[key] = {"answer": answer, "hits": hits,
                                   "flags": [f for f in flags if not f.startswith("pii_")]}
        if cache_hit:
            flags.append("cache_hit")

        latency = round(time.perf_counter() - t0, 3)
        trace_id = uuid.uuid4().hex[:12]
        sources = [h["source"] for h in hits]
        top = hits[0]["score"] if hits else None
        # 6. tracing
        self.store.log({
            "id": trace_id, "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "model": self.name, "prompt_version": self.prompt_version, "question": q,
            "answer": answer, "sources": json.dumps(sources), "top_score": top, "latency_s": latency,
            "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "cache_hit": int(cache_hit),
            "flags": json.dumps(flags), "feedback": None, "channel": self.channel,
        })
        return {
            "answer": answer, "contexts": [h["text"] for h in hits], "sources": sources,
            "scores": [h["score"] for h in hits], "latency_s": latency,
            "prompt_tokens": tokens[0], "completion_tokens": tokens[1],
            "trace_id": trace_id, "cache_hit": cache_hit, "flags": flags,
            "flag_labels": [FLAG_LABELS.get(f, f) for f in flags],
            "redacted_question": q, "pii_redacted": bool(pii_flags),
            "top_score": top, "prompt_version": self.prompt_version,
        }

    # 7. feedback
    def feedback(self, trace_id: str, helpful: bool) -> bool:
        return self.store.set_feedback(trace_id, 1 if helpful else -1)


def compute_metrics(traces: list[dict]) -> dict:
    """KPIs for the monitoring dashboard / monitor.py (traces newest first)."""
    import numpy as np
    n = len(traces)
    lat = [t["latency_s"] for t in traces if t["latency_s"] is not None]
    llm_calls = [t for t in traces if (t["prompt_tokens"] or 0) > 0]
    fb = [t["feedback"] for t in traces if t["feedback"] is not None]
    flag_counts: dict[str, int] = {}
    for t in traces:
        for f in t["flags"]:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    def pct(k):
        return round(100 * k / n, 1) if n else 0.0
    return {
        "kpis": {
            "requests": n,
            "avg_latency_s": round(float(np.mean(lat)), 3) if lat else 0.0,
            "p95_latency_s": round(float(np.percentile(lat, 95)), 3) if lat else 0.0,
            "cache_hit_pct": pct(sum(t["cache_hit"] for t in traces)),
            "abstain_pct": pct(sum("low_relevance_abstain" in t["flags"] for t in traces)),
            "blocked_injection_pct": pct(sum("prompt_injection" in t["flags"] for t in traces)),
            "avg_tokens": round(float(np.mean([(t["prompt_tokens"] or 0) + (t["completion_tokens"] or 0)
                                               for t in llm_calls])), 1) if llm_calls else 0.0,
            "feedback_count": len(fb),
            "positive_feedback_pct": round(100 * sum(1 for v in fb if v > 0) / len(fb), 1) if fb else 0.0,
        },
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda kv: -kv[1])),
        "flag_labels": FLAG_LABELS,
        "low_relevance": [{"ts": t["ts"], "question": t["question"], "top_score": t["top_score"]}
                          for t in traces if "low_relevance_abstain" in t["flags"]][:10],
    }


if __name__ == "__main__":
    bot = LLMOpsRAGBot()
    bot.channel = "terminal"
    chat_loop(bot, show=lambda r: f"trace={r['trace_id']} cache={r['cache_hit']} flags={r['flags']} sources={r['sources']}")
