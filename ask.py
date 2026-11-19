"""Ask a question from the command line.

    python ask.py "How do I add more trees to a trained forest?" [-k 5]

Uses an LLM if ANTHROPIC_API_KEY (or OPENAI_API_KEY / OPENAI_BASE_URL) is set,
otherwise prints the best passages (extractive mode, no LLM).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.rag import RAG  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("-k", type=int, default=5, help="passages to retrieve")
args = ap.parse_args()

rag = RAG.from_config(Path(__file__).resolve().parent, top_k=args.k)
a = rag.answer(args.question)
print(f"[mode: {a.mode}]\n")
print(a.text, "\n")
print("Sources:")
for p in a.passages:
    print(f"  [{p.n}] {p.chunk.doc}.rst > {p.chunk.heading}  (score {p.score:.2f})")
c = a.citations
print(f"\nCitation check: cited={c['cited']} invalid={c['invalid']} uncited_sentences={c['uncited_sentences']} ok={c['ok']}")
