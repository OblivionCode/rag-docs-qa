from src import rag

def test_prompt_selector(monkeypatch):
    monkeypatch.delenv("RAG_PROMPT", raising=False)
    assert rag.get_system_prompt() == rag.SYSTEM_PROMPT
    monkeypatch.setenv("RAG_PROMPT", "v2")
    assert rag.get_system_prompt() == rag.SYSTEM_PROMPT_V2
