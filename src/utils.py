"""Shared helpers: configuration, prompt registry and paths."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def path(*parts: str) -> Path:
    """Absolute path inside the project (relative paths in config resolve from here)."""
    p = Path(*parts)
    return p if p.is_absolute() else ROOT / p


def load_config(config_path: str | None = None) -> dict:
    """Load config.yaml, then apply environment overrides LLM_PROVIDER / LLM_MODEL."""
    with open(config_path or path("config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if os.getenv("LLM_PROVIDER"):
        cfg["llm"]["provider"] = os.environ["LLM_PROVIDER"].strip().lower()
    if os.getenv("LLM_MODEL"):
        cfg["llm"]["model"] = os.environ["LLM_MODEL"].strip()
    if os.getenv("OLLAMA_BASE_URL"):
        cfg["llm"]["base_url"] = os.environ["OLLAMA_BASE_URL"].strip()
    return cfg


def load_prompts(prompts_path: str | None = None) -> dict:
    with open(prompts_path or path("prompts", "prompts.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_offline(cfg: dict) -> bool:
    return cfg["llm"].get("provider", "ollama") == "offline"


def chat_loop(bot, show=None) -> None:
    """Simple terminal chat used by each model module."""
    print(f"\n{bot.name}  (type 'exit' to quit)\n" + "-" * 60)
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        r = bot.answer(q)
        print(f"\nBot: {r['answer']}")
        meta = f"[{r.get('latency_s', 0):.2f}s | tokens {r.get('prompt_tokens', 0)}+{r.get('completion_tokens', 0)}]"
        if show:
            meta += " " + show(r)
        print(meta)
