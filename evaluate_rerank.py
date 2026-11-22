"""Cross-encoder reranking evaluation: tune on dev first, then run test once.

    python evaluate_rerank.py                  # Phase A: tune on dev, writes results/rerank_selected.json
    python evaluate_rerank.py --test           # Phase B: run once on test with the dev-selected setting
    python evaluate_rerank.py --model BAAI/bge-reranker-base    # try another cross-encoder (dev only)

Reuses the chunking and BM25 settings that evaluate.py selected on dev (results/selected_config.json);
run `python evaluate.py` first if that file is missing.
Needs: pip install sentence-transformers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluate import eval_split, table  # noqa: E402
from src.evaluation import load_questions, paired_diff, relevance_table, run_retriever, summarize  # noqa: E402
from src.ingest import load_corpus, make_chunks  # noqa: E402
from src.retrievers import BM25Retriever, RRFRetriever, SentenceTransformerRetriever  # noqa: E402

RESULTS = ROOT / "results"
DEPTHS = [10, 20, 30, 50]
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DENSE = "BAAI/bge-small-en-v1.5"
DENSE_PREFIX = "Represent this sentence for searching relevant passages: "


class RerankRetriever:
    """Re-score the top `depth` candidates of a first-stage retriever with a cross-encoder."""

    def __init__(self, first, texts, model, depth, name):
        self.first, self.texts, self.model, self.depth, self.name = first, texts, model, depth, name

    def search(self, query, k=10):
        cands = [int(i) for i, _ in self.first.search(query, self.depth)]
        scores = self.model.predict([(query, self.texts[i]) for i in cands], show_progress_bar=False)
        order = sorted(zip(cands, scores), key=lambda t: -float(t[1]))
        return [(i, float(s)) for i, s in order[:k]]


def load_cross_encoder(name):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        sys.exit("sentence-transformers is not installed: pip install sentence-transformers")
    return CrossEncoder(name)


def setup(dense_name):
    cfg_path = RESULTS / "selected_config.json"
    if not cfg_path.exists():
        sys.exit("results/selected_config.json not found - run `python evaluate.py` first.")
    cfg = json.loads(cfg_path.read_text())
    docs = load_corpus(ROOT / "data" / "corpus")
    qs = load_questions(ROOT / "eval" / "questions.jsonl", docs)
    c = cfg["chunking"]
    chunks = make_chunks(docs, c["strategy"], c["size"], c["with_heading"])
    if len(chunks) != cfg["n_chunks"]:
        sys.exit(f"chunk count {len(chunks)} != {cfg['n_chunks']} in selected_config.json; rerun evaluate.py")
    texts = [ch.text for ch in chunks]
    bm = cfg["bm25"]
    bm25 = BM25Retriever(bm["k1"], bm["b"], bm["stem"]).fit(texts)
    dense = SentenceTransformerRetriever(dense_name, DENSE_PREFIX).fit(texts)
    hybrid = RRFRetriever([bm25, dense]).fit([None] * len(chunks))
    return qs, chunks, texts, {"BM25": bm25, "Hybrid BM25+dense": hybrid}


def phase_dev(args):
    qs, chunks, texts, firsts = setup(args.dense)
    ce = load_cross_encoder(args.model)
    dev = [q for q in qs if q.split == "dev"]
    rel = relevance_table(chunks, dev)
    recs = []
    for fname, first in firsts.items():
        rows, ms = run_retriever(first, chunks, dev, rel)
        s = summarize(rows, n_boot=100)
        recs.append({"first_stage": fname, "depth": 0, "mrr@10": s["mrr@10"], "recall@5": s["recall@5"],
                     "recall@10": s["recall@10"], "ms_per_query": ms})
        for depth in DEPTHS:
            r = RerankRetriever(first, texts, ce, depth, f"{fname} + rerank@{depth}")
            rows, ms = run_retriever(r, chunks, dev, rel)
            s = summarize(rows, n_boot=100)
            recs.append({"first_stage": fname, "depth": depth, "mrr@10": s["mrr@10"], "recall@5": s["recall@5"],
                         "recall@10": s["recall@10"], "ms_per_query": ms})
    df = pd.DataFrame(recs)
    df.to_csv(RESULTS / "rerank_dev_grid.csv", index=False)
    print(f"\nDev grid ({len(dev)} questions, depth 0 = no reranking, model {args.model}):")
    print(df.round(3).to_string(index=False))

    rr = df[df.depth > 0].sort_values(["mrr@10", "depth"], ascending=[False, True]).iloc[0]
    base = df[(df.first_stage == rr.first_stage) & (df.depth == 0)].iloc[0]
    sel = {"model": args.model, "dense": args.dense, "first_stage": rr.first_stage, "depth": int(rr.depth)}
    (RESULTS / "rerank_selected.json").write_text(json.dumps(sel, indent=2))
    print(f"\nSelected on dev: {rr.first_stage} + rerank@{int(rr.depth)} (dev MRR {rr['mrr@10']:.3f} "
          f"vs {base['mrr@10']:.3f} without reranking)")
    if rr["mrr@10"] <= base["mrr@10"]:
        print("Reranking did not beat its first stage on dev. Consider reporting this as a negative result "
              "instead of running the test split.")
    print("Saved results/rerank_dev_grid.csv and results/rerank_selected.json")
    print("When you are happy with the dev result, run the test split once:  python evaluate_rerank.py --test")


def phase_test(args):
    sel_path = RESULTS / "rerank_selected.json"
    if not sel_path.exists():
        sys.exit("Run `python evaluate_rerank.py` (dev tuning) first.")
    out = RESULTS / "rerank_test_results.csv"
    if out.exists() and not args.force:
        sys.exit(f"{out.name} exists: the test split is meant to be run once. Use --force to override.")
    sel = json.loads(sel_path.read_text())
    qs, chunks, texts, firsts = setup(sel["dense"])
    ce = load_cross_encoder(sel["model"])
    test = [q for q in qs if q.split == "test"]
    rel = relevance_table(chunks, test)
    rr_name = f"{sel['first_stage']} + rerank@{sel['depth']}"
    allr = {"BM25": firsts["BM25"], "Hybrid BM25+dense": firsts["Hybrid BM25+dense"],
            rr_name: RerankRetriever(firsts[sel["first_stage"]], texts, ce, sel["depth"], rr_name)}
    res = eval_split(allr, chunks, test, rel)
    df = table(res)
    df.to_csv(out)
    print(f"\nTest ({len(test)} questions, model {sel['model']}), 95% bootstrap CIs:")
    for name, r in df.iterrows():
        print(f"{name:34s} MRR@10 {r['mrr@10']:.3f} ({r['mrr@10_lo']:.2f}-{r['mrr@10_hi']:.2f})  "
              f"Recall@5 {r['recall@5']:.3f} ({r['recall@5_lo']:.2f}-{r['recall@5_hi']:.2f})  "
              f"Recall@1 {r['recall@1']:.3f}  {r['latency_ms']:.1f} ms/query")
    diffs = []
    print("\nPaired differences of the reranked retriever (bootstrap 95% CI):")
    for base in dict.fromkeys(["BM25", sel["first_stage"]]):
        for metric in ("mrr@10", "recall@5"):
            d, lo, hi = paired_diff(res[rr_name][0], res[base][0], metric)
            diffs.append({"vs": base, "metric": metric, "delta": d, "ci_lo": lo, "ci_hi": hi,
                          "significant": bool(lo > 0 or hi < 0)})
            print(f"  vs {base:20s} {metric:9s} {d:+.3f} ({lo:+.3f} to {hi:+.3f})"
                  + ("  significant" if (lo > 0 or hi < 0) else ""))
    pd.DataFrame(diffs).to_csv(RESULTS / "rerank_test_paired_diff.csv", index=False)
    print("\nSaved results/rerank_test_results.csv and results/rerank_test_paired_diff.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="cross-encoder model name (dev phase)")
    ap.add_argument("--dense", default=DENSE, help="dense model used in the hybrid first stage")
    ap.add_argument("--test", action="store_true", help="run the test split once with the dev-selected setting")
    ap.add_argument("--force", action="store_true", help="allow re-running the test phase")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    (phase_test if args.test else phase_dev)(args)


if __name__ == "__main__":
    main()
