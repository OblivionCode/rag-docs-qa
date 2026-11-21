"""Structural evaluation of generated answers (does not need a judge model).

    python evaluate_generation.py              # uses ANTHROPIC_API_KEY / OPENAI_* if set, else extractive baseline

Reports, on the 51 test questions:
  citation_ok      answer cites existing passages and every substantive sentence has a citation
  cited_gold       at least one cited passage is a gold hit (is the answer grounded in the right section?)
  abstained        answer says it does not know (should be rare on answerable questions)
and on 10 off-topic questions (eval/unanswerable.jsonl):
  abstain_rate     answer abstains instead of making something up (should be high)

It cannot judge whether the prose is *correct*; read results/generation_*.csv for that.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.evaluation import is_hit, load_questions  # noqa: E402
from src.ingest import load_corpus  # noqa: E402
from src.rag import RAG, generator_from_env  # noqa: E402

SPLIT = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "test"
PROMPT = os.environ.get("RAG_PROMPT", "v1")
TAG = (f"_{SPLIT}" if SPLIT != "test" else "") + (f"_{PROMPT}" if PROMPT != "v1" else "")
gen = generator_from_env()
rag = RAG.from_config(ROOT, generator=gen)
docs = load_corpus(ROOT / "data" / "corpus")
test = [q for q in load_questions(ROOT / "eval" / "questions.jsonl", docs) if q.split == SPLIT]

rows = []
for q in test:
    a = rag.answer(q.question)
    cited_gold = any(is_hit(p.chunk, q) for p in a.passages if p.n in a.citations["cited"])
    rows.append({"id": q.id, "question": q.question, "answer": a.text, "citation_ok": a.citations["ok"],
                 "cited_gold": cited_gold, "abstained": a.citations["abstained"]})
df = pd.DataFrame(rows)

off = [json.loads(l) for l in (ROOT / "eval" / "unanswerable.jsonl").read_text().splitlines()]
ab = []
for r in off:
    a = rag.answer(r["question"])
    ab.append({"id": r["id"], "question": r["question"], "answer": a.text, "abstained": a.citations["abstained"]})
ab = pd.DataFrame(ab)

mode = gen.name
df.to_csv(ROOT / "results" / f"generation_{mode}{TAG}_answerable.csv", index=False)
ab.to_csv(ROOT / "results" / f"generation_{mode}{TAG}_unanswerable.csv", index=False)
summary = {"mode": mode, "split": SPLIT, "prompt": PROMPT, "n_answerable": len(df), "citation_ok": float(df.citation_ok.mean()),
           "cited_gold": float(df.cited_gold.mean()), "abstained_on_answerable": float(df.abstained.mean()),
           "n_unanswerable": len(ab), "abstain_rate_on_unanswerable": float(ab.abstained.mean())}
(ROOT / "results" / f"generation_{mode}{TAG}_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
