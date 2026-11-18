"""Model 2: Retrieval-Augmented Generation (naive prompt v1).

Pipeline: KB docs -> chunk by FAQ heading -> embed -> in-memory vector index
-> top-k cosine search -> prompt with [source] context -> LLM.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .llm_client import OllamaClient
from .utils import chat_loop, load_config, load_prompts, path

ABSTAIN_MSG = "I don't have that information. Please contact the relevant office through the student portal."


# ---------------------------------------------------------------- loading + chunking
@dataclass
class Chunk:
    text: str       # title-prefixed text that gets embedded and shown as context
    source: str     # file name, e.g. 03_examinations.md
    title: str      # document title
    question: str   # FAQ heading ("" for intro text)
    body: str       # FAQ answer paragraph (used for extractive offline answers)


def load_documents(kb_dir: str | Path) -> list[tuple[str, str]]:
    kb = path(str(kb_dir))
    docs = []
    for f in sorted(kb.iterdir()):
        if f.suffix.lower() in {".md", ".txt"}:
            docs.append((f.name, f.read_text(encoding="utf-8")))
    return docs


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):  # prefer to break at a sentence / space
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return parts


def chunk_document(name: str, text: str, size: int = 700, overlap: int = 100) -> list[Chunk]:
    """One FAQ (## heading + answer) ~= one chunk, prefixed with the document title."""
    lines = text.strip().splitlines()
    title = name
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]
    sections = re.split(r"(?m)^## ", "\n".join(lines))
    chunks: list[Chunk] = []
    for i, sec in enumerate(sections):
        sec = sec.strip()
        if not sec:
            continue
        if i == 0:
            question, body = "", sec  # intro text before the first "## " heading
        else:
            question, _, body = sec.partition("\n")
            question, body = question.strip(), body.strip()
        for piece in _split_long(body, size, overlap):
            head = f"{title}\n## {question}\n" if question else f"{title}\n"
            chunks.append(Chunk(text=head + piece, source=name, title=title, question=question, body=piece))
    return chunks


def build_chunks(cfg: dict) -> list[Chunk]:
    rag = cfg["rag"]
    out: list[Chunk] = []
    for name, text in load_documents(rag["kb_dir"]):
        out.extend(chunk_document(name, text, int(rag["chunk_size"]), int(rag["chunk_overlap"])))
    return out


# ---------------------------------------------------------------- embeddings + index
class Embedder:
    """sentence-transformers (normalised) with a TF-IDF fallback."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", verbose: bool = True):
        self.kind = "tfidf"
        self.model_name = model_name
        self._st = None
        self._tfidf = None
        if os.getenv("EMBEDDINGS", "").lower() != "tfidf":
            try:
                import logging
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                for name in ("sentence_transformers", "transformers", "huggingface_hub"):
                    logging.getLogger(name).setLevel(logging.ERROR)
                from sentence_transformers import SentenceTransformer
                try:  # use the local cache first (no network), download only the first time
                    self._st = SentenceTransformer(model_name, device="cpu", local_files_only=True)
                except Exception:
                    self._st = SentenceTransformer(model_name, device="cpu")
                self.kind = "sentence-transformers"
            except Exception as e:  # no package / no internet for first download / etc.
                if verbose:
                    print(f"[embeddings] sentence-transformers unavailable ({type(e).__name__}: {str(e)[:120]})")
        if verbose:
            label = f"sentence-transformers ({model_name})" if self._st else "TF-IDF (1-2 grams, fallback)"
            print(f"[embeddings] using {label}")

    @property
    def label(self) -> str:
        return f"sentence-transformers/{self.model_name.split('/')[-1]}" if self._st else "tfidf"

    def fit(self, texts: list[str]) -> np.ndarray:
        if self._st is not None:
            return self.encode(texts)
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        m = self._tfidf.fit_transform(texts).toarray()
        return _normalise(m)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._st is not None:
            return np.asarray(self._st.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        return _normalise(self._tfidf.transform(texts).toarray())


def _normalise(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class VectorIndex:
    """In-memory numpy matrix; cosine similarity = dot product of normalised vectors."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = embedder.fit([c.text for c in chunks])

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q = self.embedder.encode([query])[0]
        scores = self.matrix @ q
        idx = np.argsort(-scores)[:top_k]
        return [
            {"text": self.chunks[i].text, "source": self.chunks[i].source,
             "score": round(float(scores[i]), 4), "body": self.chunks[i].body,
             "question": self.chunks[i].question}
            for i in idx
        ]


def build_index(cfg: dict, verbose: bool = True) -> VectorIndex:
    chunks = build_chunks(cfg)
    emb = Embedder(cfg["embeddings"]["model"], verbose=verbose)
    idx = VectorIndex(chunks, emb)
    if verbose:
        print(f"[index] {len(chunks)} chunks from {cfg['rag']['kb_dir']}")
    return idx


def extractive_answer(hit: dict) -> str:
    """Offline mode: return the best FAQ paragraph with its citation."""
    return f"{hit['body'].strip()} [{hit['source']}]"


# ---------------------------------------------------------------- the bot
class RAGBot:
    name = "Model 2: RAG"
    tagline = "Retrieves KB passages and grounds the LLM in them"

    def __init__(self, cfg: dict | None = None, prompt_version: str = "v1",
                 index: VectorIndex | None = None, verbose: bool = True):
        self.cfg = cfg or load_config()
        self.prompts = load_prompts()
        self.prompt_version = prompt_version
        self.client = OllamaClient(self.cfg)
        self.top_k = int(self.cfg["rag"]["top_k"])
        self.min_score = float(self.cfg["rag"]["min_score"])
        self.index = index or build_index(self.cfg, verbose=verbose)

    @property
    def system_prompt(self) -> str:
        return self.prompts["rag"][self.prompt_version]["system"]

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        return self.index.search(question, top_k or self.top_k)

    def build_prompt(self, question: str, hits: list[dict]) -> list[dict]:
        context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
        p = self.prompts["rag"][self.prompt_version]
        return [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"].format(context=context, question=question)},
        ]

    def generate(self, question: str, hits: list[dict]) -> dict:
        """LLM call (or extractive answer / abstain in offline mode)."""
        out = self.client.chat(self.build_prompt(question, hits))
        if out is not None:
            return out
        if self.client.offline:
            if hits and hits[0]["score"] >= self.min_score:
                text = extractive_answer(hits[0])
            else:
                text = ABSTAIN_MSG
            return {"text": text, "prompt_tokens": 0, "completion_tokens": 0, "offline": True}
        return {"text": f"[error] LLM unreachable at {self.client.base_url} ({self.client.last_error}). Is Ollama running?",
                "prompt_tokens": 0, "completion_tokens": 0, "error": True}

    def answer(self, question: str) -> dict:
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        out = self.generate(question, hits)
        return {
            "answer": out["text"],
            "contexts": [h["text"] for h in hits],
            "sources": [h["source"] for h in hits],
            "scores": [h["score"] for h in hits],
            "latency_s": round(time.perf_counter() - t0, 3),
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "prompt_version": self.prompt_version,
        }


if __name__ == "__main__":
    chat_loop(RAGBot(), show=lambda r: "sources: " + ", ".join(f"{s} ({sc:.2f})" for s, sc in zip(r["sources"], r["scores"])))
