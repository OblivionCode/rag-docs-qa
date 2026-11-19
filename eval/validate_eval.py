"""Check every gold label: the section exists (uniquely) and contains the evidence regex."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.ingest import load_corpus  # noqa: E402


def resolve(docs, doc, section):
    d = docs[doc]
    sec_l = section.lower()
    exact = [s for s in d.sections if s.title.lower() == sec_l]
    if len(exact) == 1:
        return d, exact
    hits = [s for s in d.sections if " > ".join(s.path).lower().endswith(sec_l) or s.title.lower().startswith(sec_l)]
    return d, hits


def main():
    docs = load_corpus(ROOT / "data" / "corpus")
    bad = 0
    for line in (ROOT / "eval" / "questions.jsonl").read_text().splitlines():
        q = json.loads(line)
        if q["doc"] not in docs:
            print(f"{q['id']} MISSING DOC {q['doc']}"); bad += 1; continue
        d, hits = resolve(docs, q["doc"], q["section"])
        if len(hits) != 1:
            print(f"{q['id']} {q['doc']} / {q['section']!r}: {len(hits)} section matches", [' > '.join(h.path) for h in hits][:4]); bad += 1; continue
        s = hits[0]
        if not re.search(q["evidence"], d.text[s.start:s.end], re.I):
            print(f"{q['id']} {q['doc']} / {q['section']!r}: evidence {q['evidence']!r} NOT in section"); bad += 1
    print("OK" if not bad else f"{bad} problems")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
