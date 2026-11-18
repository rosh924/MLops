"""Terminal LLMOps report from the Model 3 trace database.

    python monitor.py            (all traces)
    python monitor.py --last 50  (most recent 50)
"""
from __future__ import annotations

import argparse

from src.model3_llmops import FLAG_LABELS, TraceStore, compute_metrics
from src.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="LLMOps monitoring report")
    ap.add_argument("--last", type=int, default=None, help="only the most recent N traces")
    args = ap.parse_args()
    cfg = load_config()
    store = TraceStore(cfg["llmops"]["trace_db"])
    traces = store.fetch(args.last)
    print("=" * 66)
    print(" CHRIST Helpdesk - LLMOps monitoring report")
    print(f" trace DB: {store.db_path}")
    print("=" * 66)
    if not traces:
        print("No traces yet. Chat with Model 3 (python -m src.model3_llmops) or run the web app.")
        return
    m = compute_metrics(traces)
    k = m["kpis"]
    rows = [("Requests", k["requests"]), ("Avg latency", f"{k['avg_latency_s']:.2f} s"),
            ("p95 latency", f"{k['p95_latency_s']:.2f} s"), ("Cache hit rate", f"{k['cache_hit_pct']}%"),
            ("Abstain rate (low relevance)", f"{k['abstain_pct']}%"),
            ("Blocked injections", f"{k['blocked_injection_pct']}%"),
            ("Avg tokens per LLM call", k["avg_tokens"]), ("Feedback received", k["feedback_count"]),
            ("Positive feedback", f"{k['positive_feedback_pct']}%")]
    print("\nKPIs")
    for name, val in rows:
        print(f"  {name:<30} {val}")
    print("\nGuardrail / pipeline flags")
    for f, c in m["flag_counts"].items() or {}:
        print(f"  {FLAG_LABELS.get(f, f):<30} {c:>4}  {'#' * min(c, 30)}")
    if not m["flag_counts"]:
        print("  (none)")
    print("\nLow-relevance questions (consider adding these to the KB)")
    for t in m["low_relevance"] or []:
        print(f"  {t['ts'][:19]}  score={t['top_score'] if t['top_score'] is None else round(t['top_score'], 2)}  {t['question'][:70]}")
    if not m["low_relevance"]:
        print("  (none)")
    print("\nRecent traces")
    print(f"  {'time':<19} {'ch':<8} {'lat':>6} {'fb':>3}  flags / question")
    for t in traces[:10]:
        fb = {1: "+1", -1: "-1"}.get(t["feedback"], "")
        print(f"  {t['ts'][:19]:<19} {(t.get('channel') or ''):<8} {t['latency_s']:>5.2f}s {fb:>3}  "
              f"{','.join(t['flags']) or '-'} | {t['question'][:50]}")


if __name__ == "__main__":
    main()
