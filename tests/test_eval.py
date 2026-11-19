import math

import numpy as np

from src.evaluation import Question, is_hit, paired_diff, per_question_metrics, summarize
from src.ingest import Chunk


def chunk(doc, s, e):
    return Chunk(0, doc, s, e, "x", "h")


Q = Question("q1", "dev", "?", "tree", "S", start=100, end=200)


def test_is_hit_requires_same_doc_and_enough_overlap():
    assert is_hit(chunk("tree", 100, 200), Q)
    assert is_hit(chunk("tree", 120, 180), Q)  # fully inside
    assert is_hit(chunk("tree", 50, 400), Q)  # section fully inside big chunk (overlap = section len)
    assert not is_hit(chunk("svm", 100, 200), Q)
    assert not is_hit(chunk("tree", 190, 400), Q)  # 10 chars of 100 is an edge touch, not a hit
    assert not is_hit(chunk("tree", 0, 100), Q)  # adjacent, no overlap


def test_metrics_known_ranking():
    rel = np.array([0, 0, 1, 0, 1, 0], dtype=bool)  # chunks 2 and 4 relevant
    m = per_question_metrics([0, 1, 2, 3, 4, 5], rel)
    assert m["first_hit_rank"] == 3 and math.isclose(m["mrr@10"], 1 / 3)
    assert (m["recall@1"], m["recall@3"], m["recall@5"]) == (0.0, 1.0, 1.0)
    dcg = 1 / math.log2(4) + 1 / math.log2(6)
    ideal = 1 + 1 / math.log2(3)
    assert math.isclose(m["ndcg@10"], dcg / ideal)


def test_metrics_perfect_and_total_miss():
    rel = np.array([1, 0, 0], dtype=bool)
    assert per_question_metrics([0, 1, 2], rel)["ndcg@10"] == 1.0
    miss = per_question_metrics([1, 2], rel)
    assert miss["mrr@10"] == 0.0 and miss["recall@10"] == 0.0 and math.isnan(miss["first_hit_rank"])


def test_summarize_ci_brackets_mean_and_paired_diff_sign():
    rows_a = [{m: float(i % 2) for m in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]} for i in range(60)]
    s = summarize(rows_a)
    assert s["mrr@10_lo"] <= s["mrr@10"] <= s["mrr@10_hi"] and math.isclose(s["mrr@10"], 0.5)
    rows_b = [{**r, "mrr@10": 0.0} for r in rows_a]
    d, lo, hi = paired_diff(rows_a, rows_b)
    assert d > 0 and lo > 0
