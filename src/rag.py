"""The RAG pipeline: retrieve passages, build a grounded prompt, generate a cited answer.

Generators
  ExtractiveGenerator        no LLM; quotes the best passages with citations (works offline)
  AnthropicGenerator         Messages API over plain HTTP (needs ANTHROPIC_API_KEY)
  OpenAICompatibleGenerator  /chat/completions (OpenAI, Ollama, Groq, vLLM, ...)

The HTTP generators were tested against a local mock server (tests/test_rag.py),
not against the live APIs, because the sandbox this was built in has no API keys.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import Chunk, load_corpus, make_chunks
from .retrievers import BM25Retriever, Retriever

ABSTAIN = "I don't know based on the provided documents."

SYSTEM_PROMPT = (
    "You answer questions about the scikit-learn user guide using ONLY the numbered passages provided.\n"
    "Rules:\n"
    "1. Cite the passages that support each claim, like [1] or [2][3], immediately after the claim.\n"
    "2. Do not use outside knowledge. Do not cite a passage that does not support the claim.\n"
    f"3. If the passages do not contain the answer, reply exactly: {ABSTAIN}\n"
    "4. Be concise. Keep class, function and parameter names exactly as written."
)


@dataclass
class Passage:
    n: int  # 1-based citation number
    chunk: Chunk
    score: float


@dataclass
class Answer:
    question: str
    text: str
    passages: list[Passage]
    citations: dict = field(default_factory=dict)
    mode: str = ""


def build_prompt(question: str, passages: list[Passage]) -> str:
    ctx = "\n\n".join(f"[{p.n}] (source: {p.chunk.doc}.rst, section: {p.chunk.heading})\n{p.chunk.text}" for p in passages)
    return f"Passages:\n\n{ctx}\n\nQuestion: {question}\n\nAnswer (with [n] citations):"


_CITE = re.compile(r"\[(\d+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`(])")


def verify_citations(answer: str, n_passages: int) -> dict:
    """Cheap structural checks that catch many hallucination patterns:
    cited numbers that do not exist, and substantive sentences with no citation."""
    cited = sorted({int(m) for m in _CITE.findall(answer)})
    invalid = [c for c in cited if not 1 <= c <= n_passages]
    abstained = answer.strip().startswith(ABSTAIN)
    sentences = [s for s in _SENT.split(answer.strip()) if len(s.split()) >= 5]
    uncited = [s for s in sentences if not _CITE.search(s)]
    return {"cited": cited, "invalid": invalid, "abstained": abstained,
            "uncited_sentences": len(uncited), "n_sentences": len(sentences),
            "ok": not invalid and (abstained or (bool(cited) and not uncited))}


# ----------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------
class ExtractiveGenerator:
    """No LLM. Returns the opening sentences of the top passages, cited. Useful as an
    offline fallback and as the baseline an LLM answer has to beat."""

    name = "extractive"

    def __init__(self, n_passages: int = 2, n_sentences: int = 3):
        self.n_passages, self.n_sentences = n_passages, n_sentences

    def generate(self, question: str, passages: list[Passage]) -> str:
        if not passages:
            return ABSTAIN
        parts = []
        for p in passages[: self.n_passages]:
            lines = p.chunk.text.split("\n")[1:]  # drop the heading-path line
            if lines and lines[0].strip() == p.chunk.heading.split(" > ")[-1].strip():
                lines = lines[1:]  # ...and the repeated section title
            body = re.sub(r"\s+", " ", " ".join(lines)).strip()
            sents = [s for s in _SENT.split(body) if len(s.split()) >= 4][: self.n_sentences]
            parts.append(" ".join(f"{s} [{p.n}]" for s in sents))
        return "\n\n".join(x for x in parts if x) or ABSTAIN


def _post_json(url: str, headers: dict, body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class AnthropicGenerator:
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str = "https://api.anthropic.com", max_tokens: int = 600):
        self.model = model or os.environ.get("RAG_MODEL", "claude-sonnet-4-5")
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.base_url, self.max_tokens = base_url.rstrip("/"), max_tokens

    def generate(self, question: str, passages: list[Passage]) -> str:
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0, "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": build_prompt(question, passages)}]}
        out = _post_json(f"{self.base_url}/v1/messages",
                         {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, body)
        return "".join(b.get("text", "") for b in out["content"] if b.get("type") == "text").strip()


class OpenAICompatibleGenerator:
    name = "openai-compatible"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, max_tokens: int = 600):
        self.model = model or os.environ.get("RAG_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.max_tokens = max_tokens

    def generate(self, question: str, passages: list[Passage]) -> str:
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": build_prompt(question, passages)}]}
        out = _post_json(f"{self.base_url}/chat/completions", {"authorization": f"Bearer {self.api_key}"}, body)
        return out["choices"][0]["message"]["content"].strip()


def generator_from_env():
    """ANTHROPIC_API_KEY -> Anthropic; OPENAI_API_KEY or OPENAI_BASE_URL -> OpenAI-compatible; else extractive."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicGenerator()
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        return OpenAICompatibleGenerator()
    return ExtractiveGenerator()


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
class RAG:
    def __init__(self, chunks: list[Chunk], retriever: Retriever, generator=None, top_k: int = 5):
        self.chunks, self.retriever, self.top_k = chunks, retriever, top_k
        self.generator = generator or ExtractiveGenerator()

    @classmethod
    def from_config(cls, root: str | Path, generator=None, top_k: int = 5) -> "RAG":
        """Build the index with the configuration selected on the dev split (results/selected_config.json)."""
        root = Path(root)
        cfg = json.loads((root / "results" / "selected_config.json").read_text())
        docs = load_corpus(root / "data" / "corpus")
        c = cfg["chunking"]
        chunks = make_chunks(docs, c["strategy"], c["size"], c["with_heading"])
        bm = cfg["bm25"]
        retr = BM25Retriever(bm["k1"], bm["b"], bm["stem"]).fit([ch.text for ch in chunks])
        return cls(chunks, retr, generator or generator_from_env(), top_k)

    def retrieve(self, question: str, k: int | None = None) -> list[Passage]:
        hits = self.retriever.search(question, k or self.top_k)
        return [Passage(n + 1, self.chunks[i], s) for n, (i, s) in enumerate(hits)]

    def answer(self, question: str, k: int | None = None) -> Answer:
        passages = self.retrieve(question, k)
        text = self.generator.generate(question, passages)
        return Answer(question, text, passages, verify_citations(text, len(passages)), getattr(self.generator, "name", "?"))
