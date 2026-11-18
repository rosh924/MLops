"""Model 1: Plain LLM - no retrieval. The baseline that shows hallucination."""
from __future__ import annotations

import time

from .llm_client import OllamaClient
from .utils import chat_loop, load_config, load_prompts

OFFLINE_MSG = "[offline] no LLM - start Ollama and set LLM_PROVIDER=ollama to get Model 1 answers."


class PlainLLMBot:
    name = "Model 1: Plain LLM"
    tagline = "LLM only - no access to university data"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.prompts = load_prompts()
        self.client = OllamaClient(self.cfg)

    def answer(self, question: str) -> dict:
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": self.prompts["baseline"]["system"]},
            {"role": "user", "content": question},
        ]
        out = self.client.chat(messages)
        if out is None:
            text = OFFLINE_MSG if self.client.offline else (
                f"[error] LLM unreachable at {self.client.base_url} ({self.client.last_error}). "
                "Is Ollama running?")
            out = {"text": text, "prompt_tokens": 0, "completion_tokens": 0}
        return {
            "answer": out["text"],
            "contexts": [],
            "sources": [],
            "latency_s": round(time.perf_counter() - t0, 3),
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
        }


if __name__ == "__main__":
    chat_loop(PlainLLMBot())
