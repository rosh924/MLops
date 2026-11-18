"""Minimal Ollama REST client (local only - no data leaves the machine)."""
from __future__ import annotations

import time

import requests


class OllamaClient:
    def __init__(self, cfg: dict):
        llm = cfg["llm"]
        self.base_url = llm.get("base_url", "http://localhost:11434").rstrip("/")
        self.model = llm.get("model", "llama3.2:3b")
        self.temperature = float(llm.get("temperature", 0.1))
        self.max_tokens = int(llm.get("max_tokens", 300))
        self.timeout = float(llm.get("timeout", 120))
        self.offline = llm.get("provider", "ollama") == "offline"
        self.last_error: str | None = None

    def is_reachable(self) -> bool:
        if self.offline:
            return False
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def has_model(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(n == self.model or n.split(":")[0] == self.model for n in names)
        except Exception:
            return False

    def chat(self, messages: list[dict]) -> dict | None:
        """Call /api/chat (non-streaming). Returns dict(text, latency_s, prompt_tokens,
        completion_tokens) or None if offline / unreachable / error."""
        if self.offline:
            return None
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            if r.status_code != 200:
                self.last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                return None
            data = r.json()
        except requests.RequestException as e:
            self.last_error = str(e)
            return None
        self.last_error = None
        return {
            "text": (data.get("message", {}).get("content") or "").strip(),
            "latency_s": round(time.perf_counter() - t0, 3),
            "prompt_tokens": int(data.get("prompt_eval_count") or 0),
            "completion_tokens": int(data.get("eval_count") or 0),
        }
