"""Manual correctness review of LLM answers.
    python review_answers.py [N]     # rate N random answers (default 20); resumable
"""
import sys, random
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.evaluation import load_questions
from src.ingest import load_corpus

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ans = pd.read_csv(ROOT / "results" / "generation_openai-compatible_answerable.csv")
docs = load_corpus(ROOT / "data" / "corpus")
qs = {q.id: q for q in load_questions(ROOT / "eval" / "questions.jsonl", docs)}

out = ROOT / "results" / "generation_review.csv"
done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["id", "rating"])
ids = ans.id.tolist()
random.Random(42).shuffle(ids)
ids = ids[:N]
rows = done.to_dict("records")
seen = set(done.id)

for n, qid in enumerate(ids, 1):
    if qid in seen:
        continue
    r = ans[ans.id == qid].iloc[0]
    q = qs.get(qid)
    print("\n" + "=" * 70)
    print(f"[{n}/{len(ids)}] {qid}   cited_gold={r.cited_gold}  citation_ok={r.citation_ok}")
    print("QUESTION:", r.question)
    print("GOLD SECTION:", getattr(q, "gold_heading", "?"))
    print("ANSWER:", r.answer)
    while True:
        x = input("rate: c=correct  p=partly  w=wrong  a=abstained-fairly  q=quit > ").strip().lower()
        if x in ("c", "p", "w", "a", "q"):
            break
    if x == "q":
        break
    rows.append({"id": qid, "rating": {"c": "correct", "p": "partly", "w": "wrong", "a": "abstained_fairly"}[x]})
    pd.DataFrame(rows).to_csv(out, index=False)

d = pd.DataFrame(rows)
print("\n", d.rating.value_counts().to_string())
print(f"saved {out}")
