import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.ingest import Chunk
from src.rag import (ABSTAIN, RAG, AnthropicGenerator, ExtractiveGenerator, OpenAICompatibleGenerator, Passage,
                     build_prompt, generator_from_env, verify_citations)
from src.retrievers import BM25Retriever

CHUNKS = [
    Chunk(0, "ensemble", 0, 50, "Ensemble > Fitting additional trees\nFitting additional trees\nSet warm_start to True to add more trees to a forest. It keeps the existing trees.", "Ensemble > Fitting additional trees"),
    Chunk(1, "svm", 0, 50, "SVM > Tips\nScale your data before fitting a support vector machine to avoid slow convergence.", "SVM > Tips"),
]


def passages():
    return [Passage(1, CHUNKS[0], 1.0), Passage(2, CHUNKS[1], 0.5)]


def test_prompt_numbers_and_sources():
    p = build_prompt("how to add trees?", passages())
    assert "[1] (source: ensemble.rst, section: Ensemble > Fitting additional trees)" in p
    assert "[2] (source: svm.rst" in p and p.rstrip().endswith("Answer (with [n] citations):")


def test_verify_citations():
    ok = verify_citations("Set warm_start to add more trees to the forest [1].", 2)
    assert ok["ok"] and ok["cited"] == [1]
    assert verify_citations("This sentence cites a passage that does not exist [7].", 2)["invalid"] == [7]
    assert not verify_citations("This long claim has no citation at all in it.", 2)["ok"]
    ab = verify_citations(ABSTAIN, 2)
    assert ab["abstained"] and ab["ok"]


def test_extractive_generator_cites_every_sentence_and_skips_title():
    text = ExtractiveGenerator().generate("q", passages())
    assert "[1]" in text and "[2]" in text
    assert "Fitting additional trees Set" not in text  # repeated title removed
    assert verify_citations(text, 2)["ok"]
    assert ExtractiveGenerator().generate("q", []) == ABSTAIN


def test_pipeline_end_to_end_with_stub_generator():
    class Stub:
        name = "stub"
        def generate(self, q, ps):
            return f"Use warm_start [{ps[0].n}]."
    rag = RAG(CHUNKS, BM25Retriever().fit([c.text for c in CHUNKS]), Stub(), top_k=2)
    a = rag.answer("how do I add more trees to a forest")
    assert a.passages[0].chunk.doc == "ensemble" and a.citations["ok"] and a.mode == "stub"


class _Handler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _Handler.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if self.path == "/v1/messages":
            out = {"content": [{"type": "text", "text": "Anthropic says [1]."}]}
        else:
            out = {"choices": [{"message": {"content": "OpenAI says [1]."}}]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    _Handler.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_anthropic_generator_request_shape(server):
    g = AnthropicGenerator(model="m-1", api_key="k-123", base_url=server)
    assert g.generate("how?", passages()) == "Anthropic says [1]."
    path, headers, body = _Handler.seen[0]
    assert path == "/v1/messages"
    assert headers["x-api-key"] == "k-123" and headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "m-1" and body["temperature"] == 0 and "ONLY the numbered passages" in body["system"]
    assert body["messages"][0]["role"] == "user" and "Question: how?" in body["messages"][0]["content"]


def test_openai_compatible_generator_request_shape(server):
    g = OpenAICompatibleGenerator(model="m-2", api_key="sk-x", base_url=server + "/v1")
    assert g.generate("how?", passages()) == "OpenAI says [1]."
    path, headers, body = _Handler.seen[0]
    assert path == "/v1/chat/completions" and headers["authorization"] == "Bearer sk-x"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_generator_from_env(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    assert isinstance(generator_from_env(), ExtractiveGenerator)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert isinstance(generator_from_env(), OpenAICompatibleGenerator)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(generator_from_env(), AnthropicGenerator)
