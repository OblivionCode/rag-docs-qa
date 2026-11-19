"""Retrieval evaluation with a dev/test protocol.

    python evaluate.py                       # full protocol, no model downloads needed
    python evaluate.py --dense BAAI/bge-small-en-v1.5   # also run a sentence-transformers retriever

Phase 1 (dev)  choose chunking by mean dev MRR over the four base retrievers
Phase 2 (dev)  tune BM25 (k1, b, stemming) and LSA dimension
Phase 3 (test) evaluate every retriever and hybrid once, with bootstrap 95% CIs

The 52 dev and 51 test questions are disjoint; nothing is chosen on the test split.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.evaluation import (METRICS, load_questions, paired_diff, relevance_table,  # noqa: E402
                            run_retriever, summarize)
from src.ingest import load_corpus, make_chunks  # noqa: E402
from src.retrievers import (BM25Retriever, GloveRetriever, LSARetriever, RRFRetriever,  # noqa: E402
                            SentenceTransformerRetriever, TfidfRetriever)

RESULTS = ROOT / "results"
GLOVE = ROOT / "data" / "glove" / "glove-wiki-gigaword-100.gz"
SEED = 42
INK, INK2, MUTED, GRID, AXIS, SURFACE, BLUE, ORANGE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#2a78d6", "#eb6834"

CHUNK_GRID = [("fixed", 100, None), ("fixed", 200, None), ("fixed", 300, None),
              ("section", 100, True), ("section", 200, True), ("section", 300, True),
              ("section", 200, False)]


def chunk_label(strategy, size, heading):
    return f"{strategy}-{size}" + ("-noheading" if strategy == "section" and heading is False else "")


def base_retrievers(texts, bm25=(1.5, 0.75, False), lsa_dim=256):
    k1, b, stem = bm25
    rs = {
        "BM25": BM25Retriever(k1, b, stem).fit(texts),
        "TF-IDF": TfidfRetriever().fit(texts),
        f"LSA-{lsa_dim}": LSARetriever(lsa_dim).fit(texts),
        "GloVe-avg": GloveRetriever(GLOVE).fit(texts),
    }
    return rs


class RandomRetriever:
    name = "random"

    def __init__(self, n, seed=SEED):
        self.rng, self.n = np.random.default_rng(seed), n

    def search(self, query, k=10):
        return [(int(i), 0.0) for i in self.rng.choice(self.n, size=k, replace=False)]


def hybrids(rs, n):
    bm, lsa, gl, tf = rs["BM25"], next(v for k, v in rs.items() if k.startswith("LSA")), rs["GloVe-avg"], rs["TF-IDF"]
    mk = lambda members: RRFRetriever(members).fit([None] * n)  # noqa: E731
    out = {
        "Hybrid BM25+LSA": mk([bm, lsa]),
        "Hybrid BM25+GloVe": mk([bm, gl]),
        "Hybrid BM25+LSA+GloVe": mk([bm, lsa, gl]),
        "Hybrid all four": mk([bm, tf, lsa, gl]),
    }
    return out


def eval_split(rs, chunks, qs, rel):
    res = {}
    for name, r in rs.items():
        rows, ms = run_retriever(r, chunks, qs, rel)
        res[name] = (rows, ms)
    return res


def table(res):
    recs = []
    for name, (rows, ms) in res.items():
        s = summarize(rows)
        s.update(retriever=name, latency_ms=ms, n_questions=len(rows))
        recs.append(s)
    return pd.DataFrame(recs).set_index("retriever")


def fmt_ci(df, m):
    return df.apply(lambda r: f"{r[m]:.3f} ({r[m + '_lo']:.2f}-{r[m + '_hi']:.2f})", axis=1)


def plot_test(df, path):
    d = df.sort_values("mrr@10")
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(d) + 1.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, m, title in zip(axes, ["mrr@10", "recall@5"], ["MRR@10", "Recall@5"]):
        err = np.array([d[m] - d[m + "_lo"], d[m + "_hi"] - d[m]])
        colors = [MUTED if n == "random" else BLUE for n in d.index]
        ax.barh(d.index, d[m], xerr=err, color=colors, height=0.55, error_kw={"ecolor": INK2, "elinewidth": 1})
        for y, hi, v in zip(range(len(d)), d[m + "_hi"], d[m]):
            ax.text(hi + 0.015, y, f"{v:.2f}", va="center", ha="left", color=INK2, fontsize=9)  # past the CI whisker
        ax.set_xlim(0, 1)
        ax.set_title(f"{title} (test, n={int(d['n_questions'].iloc[0])}, 95% bootstrap CI)", loc="left", color=INK, fontsize=11)
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, length=0, labelsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", help="optional sentence-transformers model name (needs weights; not run in the sandbox)")
    ap.add_argument("--dense-query-prefix", default="Represent this sentence for searching relevant passages: ")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    docs = load_corpus(ROOT / "data" / "corpus")
    qs_all = load_questions(ROOT / "eval" / "questions.jsonl", docs)
    dev = [q for q in qs_all if q.split == "dev"]
    test = [q for q in qs_all if q.split == "test"]
    print(f"{len(docs)} docs, {len(dev)} dev / {len(test)} test questions")

    # ---------------- Phase 1: chunking sweep on dev ----------------
    print("\n== Phase 1: chunking sweep (dev) ==")
    sweep = []
    for strat, size, heading in CHUNK_GRID:
        chunks = make_chunks(docs, strat, size, heading)
        texts = [c.text for c in chunks]
        rel = relevance_table(chunks, dev)
        res = eval_split(base_retrievers(texts), chunks, dev, rel)
        for name, (rows, ms) in res.items():
            s = summarize(rows, n_boot=200)
            sweep.append({"chunking": chunk_label(strat, size, heading), "n_chunks": len(chunks), "retriever": name,
                          "mrr@10": s["mrr@10"], "recall@5": s["recall@5"], "recall@10": s["recall@10"]})
    sw = pd.DataFrame(sweep)
    sw.to_csv(RESULTS / "dev_chunking_sweep.csv", index=False)
    piv = sw.pivot(index="chunking", columns="retriever", values="mrr@10")
    piv["mean"] = piv.mean(axis=1)
    print(piv.round(3).sort_values("mean", ascending=False).to_string())
    best_label = piv["mean"].idxmax()
    best_cfg = next(c for c in CHUNK_GRID if chunk_label(*c) == best_label)
    print(f"Selected chunking: {best_label}")

    chunks = make_chunks(docs, *best_cfg)
    texts = [c.text for c in chunks]
    rel_dev, rel_test = relevance_table(chunks, dev), relevance_table(chunks, test)

    # ---------------- Phase 2: tune BM25 and LSA on dev ----------------
    print("\n== Phase 2: BM25 / LSA tuning (dev) ==")
    tune = []
    for k1, b, stem in itertools.product([0.9, 1.2, 1.5, 2.0], [0.4, 0.6, 0.75, 0.9], [False, True]):
        rows, _ = run_retriever(BM25Retriever(k1, b, stem).fit(texts), chunks, dev, rel_dev)
        s = summarize(rows, n_boot=100)
        tune.append({"retriever": "BM25", "params": f"k1={k1},b={b},stem={stem}", "k1": k1, "b": b, "stem": stem,
                     "mrr@10": s["mrr@10"], "recall@5": s["recall@5"]})
    for dim in [64, 128, 256, 384]:
        rows, _ = run_retriever(LSARetriever(dim).fit(texts), chunks, dev, rel_dev)
        s = summarize(rows, n_boot=100)
        tune.append({"retriever": "LSA", "params": f"dim={dim}", "dim": dim, "mrr@10": s["mrr@10"], "recall@5": s["recall@5"]})
    tn = pd.DataFrame(tune)
    tn.to_csv(RESULTS / "dev_tuning.csv", index=False)
    bm_best = tn[tn.retriever == "BM25"].sort_values("mrr@10", ascending=False).iloc[0]
    lsa_best = tn[tn.retriever == "LSA"].sort_values("mrr@10", ascending=False).iloc[0]
    print("BM25 default (k1=1.5,b=0.75,stem=False): dev MRR",
          round(tn[(tn.retriever == "BM25") & (tn.k1 == 1.5) & (tn.b == 0.75) & (~tn.stem.astype(bool))]["mrr@10"].iloc[0], 3))
    print("BM25 best:", bm_best["params"], "dev MRR", round(bm_best["mrr@10"], 3))
    print("LSA best:", lsa_best["params"], "dev MRR", round(lsa_best["mrr@10"], 3))
    cfg = {"chunking": {"strategy": best_cfg[0], "size": best_cfg[1], "with_heading": best_cfg[2]},
           "n_chunks": len(chunks),
           "bm25": {"k1": float(bm_best.k1), "b": float(bm_best.b), "stem": bool(bm_best.stem)},
           "lsa_dim": int(lsa_best.dim), "seed": SEED}
    (RESULTS / "selected_config.json").write_text(json.dumps(cfg, indent=2))

    # ---------------- Phase 3: test ----------------
    print("\n== Phase 3: test (run once) ==")
    rs = base_retrievers(texts, (cfg["bm25"]["k1"], cfg["bm25"]["b"], cfg["bm25"]["stem"]), cfg["lsa_dim"])
    allr = dict(rs)
    allr.update(hybrids(rs, len(chunks)))
    if args.dense:
        dense = SentenceTransformerRetriever(args.dense, args.dense_query_prefix).fit(texts)
        allr[dense.name] = dense
        allr[f"Hybrid BM25+{dense.name}"] = RRFRetriever([rs["BM25"], dense]).fit([None] * len(chunks))
    allr["random"] = RandomRetriever(len(chunks))
    res = eval_split(allr, chunks, test, rel_test)
    df = table(res)
    df.to_csv(RESULTS / "test_results.csv")

    ordered = df.sort_values("mrr@10", ascending=False)
    md = ["| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | ms/query |", "|---|---|---|---|---|---|---|"]
    for name, r in ordered.iterrows():
        md.append(f"| {name} | {r['recall@1']:.2f} | {r['recall@5']:.2f} ({r['recall@5_lo']:.2f}-{r['recall@5_hi']:.2f}) | "
                  f"{r['recall@10']:.2f} | {r['mrr@10']:.3f} ({r['mrr@10_lo']:.2f}-{r['mrr@10_hi']:.2f}) | {r['ndcg@10']:.3f} | {r['latency_ms']:.1f} |")
    (RESULTS / "test_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    diffs = []
    for name in ordered.index:
        if name in ("BM25", "random"):
            continue
        d, lo, hi = paired_diff(res[name][0], res["BM25"][0], "mrr@10")
        diffs.append({"retriever": name, "delta_mrr_vs_bm25": d, "ci_lo": lo, "ci_hi": hi, "significant": (lo > 0 or hi < 0)})
    dd = pd.DataFrame(diffs)
    dd.to_csv(RESULTS / "test_paired_diff_vs_bm25.csv", index=False)
    print("\nPaired MRR@10 difference vs BM25 (bootstrap 95% CI):")
    print(dd.round(3).to_string(index=False))

    pq = []
    for name, (rows, _) in res.items():
        for r in rows:
            pq.append({"retriever": name, **{k: r[k] for k in ("id", "first_hit_rank", "recall@5", "top1_heading")}})
    pqd = pd.DataFrame(pq)
    qtext = {q.id: (q.question, q.gold_heading) for q in test}
    pqd["question"] = pqd["id"].map(lambda i: qtext[i][0])
    pqd["gold_section"] = pqd["id"].map(lambda i: qtext[i][1])
    pqd.to_csv(RESULTS / "test_per_question.csv", index=False)

    plot_test(df, RESULTS / "test_retrieval_comparison.png")
    print("\nSaved results to", RESULTS)


if __name__ == "__main__":
    main()
