"""Retrieval evaluation: gold-section hit judgement, ranking metrics, bootstrap CIs."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HIT_FRACTION = 0.5  # a chunk is relevant if it overlaps the gold section by >= 50% of the smaller of the two


@dataclass
class Question:
    id: str
    split: str
    question: str
    doc: str
    section: str
    start: int = 0
    end: int = 0
    gold_heading: str = ""


def load_questions(path: str | Path, docs) -> list[Question]:
    qs = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        d = docs[r["doc"]]
        sec_l = r["section"].lower()
        exact = [s for s in d.sections if s.title.lower() == sec_l]
        hits = exact if len(exact) == 1 else [
            s for s in d.sections if " > ".join(s.path).lower().endswith(sec_l) or s.title.lower().startswith(sec_l)]
        if len(hits) != 1:
            raise ValueError(f"{r['id']}: section {r['section']!r} matched {len(hits)} sections in {r['doc']}")
        s = hits[0]
        qs.append(Question(r["id"], r["split"], r["question"], r["doc"], r["section"], s.start, s.end, " > ".join(s.path)))
    return qs


def is_hit(chunk, q: Question) -> bool:
    if chunk.doc != q.doc:
        return False
    overlap = min(chunk.end, q.end) - max(chunk.start, q.start)
    if overlap <= 0:
        return False
    return overlap >= HIT_FRACTION * min(chunk.end - chunk.start, q.end - q.start)


def relevance_table(chunks, questions) -> dict[str, np.ndarray]:
    """Boolean vector over all chunks per question (is that chunk a hit?)."""
    return {q.id: np.fromiter((is_hit(c, q) for c in chunks), dtype=bool, count=len(chunks)) for q in questions}


def per_question_metrics(ranked: list[int], rel: np.ndarray, ks=(1, 3, 5, 10)) -> dict[str, float]:
    flags = [bool(rel[i]) for i in ranked[:10]]
    first = next((r for r, f in enumerate(flags, 1) if f), None)
    out = {f"recall@{k}": float(any(flags[:k])) for k in ks}
    out["mrr@10"] = 1.0 / first if first else 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(flags) if f)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(int(rel.sum()), 10)))
    out["ndcg@10"] = dcg / ideal if ideal else 0.0
    out["first_hit_rank"] = float(first) if first else float("nan")
    return out


def run_retriever(retriever, chunks, questions, rel, k=10):
    """Returns (per-question DataFrame-like list of dicts, mean latency ms)."""
    rows, t_total = [], 0.0
    for q in questions:
        t0 = time.perf_counter()
        ranked = [i for i, _ in retriever.search(q.question, k)]
        t_total += time.perf_counter() - t0
        m = per_question_metrics(ranked, rel[q.id])
        m["id"] = q.id
        m["top1_doc"] = chunks[ranked[0]].doc
        m["top1_heading"] = chunks[ranked[0]].heading
        rows.append(m)
    return rows, 1000 * t_total / max(1, len(questions))


METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]


def summarize(rows, n_boot=1000, seed=42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(rows)
    out = {}
    idx = rng.integers(0, n, size=(n_boot, n))
    for m in METRICS:
        v = np.array([r[m] for r in rows])
        boots = v[idx].mean(axis=1)
        out[m] = float(v.mean())
        out[m + "_lo"], out[m + "_hi"] = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return out


def paired_diff(rows_a, rows_b, metric="mrr@10", n_boot=2000, seed=42):
    """Mean(a - b) over questions with a bootstrap 95% CI (questions are resampled jointly)."""
    a = np.array([r[metric] for r in rows_a])
    b = np.array([r[metric] for r in rows_b])
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
