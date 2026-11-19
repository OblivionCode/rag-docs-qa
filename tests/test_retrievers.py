import gzip

import numpy as np

from src.retrievers import (BM25Retriever, CrossEncoderReranker, GloveRetriever, LSARetriever, RRFRetriever,
                            SentenceTransformerRetriever, TfidfRetriever, load_glove, make_tokenizer)

TEXTS = [
    "forest ensemble warm_start adds more trees to an already trained model",
    "support vector machine class_weight handles an unbalanced rare class",
    "k-means clustering groups samples around centroids",
    "decision tree pruning with ccp_alpha reduces overfitting",
    "imputation fills missing values using the mean of each column",
]


def test_tokenizer_splits_identifiers_and_drops_stop_words():
    tok = make_tokenizer()
    assert tok("The class_weight option") == ["class_weight", "class", "weight", "option"]
    stem = make_tokenizer(stem=True)
    assert stem("weights") == stem("weight")


def test_bm25_ranks_the_matching_text_first_including_subtokens():
    r = BM25Retriever().fit(TEXTS)
    assert r.search("how to add trees to a trained model", 1)[0][0] == 0
    assert r.search("unbalanced class weight", 1)[0][0] == 1  # matches via class_weight subtokens
    assert r.search("pruning ccp_alpha", 1)[0][0] == 3
    assert BM25Retriever(stem=True).fit(TEXTS).search("class weights", 1)[0][0] == 1


def test_scores_shape_and_search_k_bounds():
    for r in (BM25Retriever(), TfidfRetriever(), LSARetriever(dim=3)):
        r.fit(TEXTS)
        assert r.scores("trees").shape == (len(TEXTS),)
        assert len(r.search("trees", 100)) == len(TEXTS)
        assert r.search("trees", 2)[0][1] >= r.search("trees", 2)[1][1]


def test_rrf_fuses_rankings():
    a = BM25Retriever().fit(TEXTS)
    b = TfidfRetriever().fit(TEXTS)
    f = RRFRetriever([a, b]).fit(TEXTS)
    top = f.search("missing values imputation", 1)[0][0]
    assert top == 4
    s = f.scores("missing values imputation")
    assert s.argmax() == 4 and s[4] > s[0]


def test_glove_loader_and_retriever(tmp_path):
    p = tmp_path / "g.txt.gz"
    words = {"trees": [1, 0, 0], "forest": [0.9, 0.1, 0], "clustering": [0, 1, 0], "means": [0, 0.9, 0.1],
             "missing": [0, 0, 1], "values": [0.1, 0, 0.9]}
    with gzip.open(p, "wt") as f:
        f.write(f"{len(words)} 3\n")
        for w, v in words.items():
            f.write(w + " " + " ".join(map(str, v)) + "\n")
    assert set(load_glove(p, {"trees", "zzz"})) == {"trees"}
    texts = ["forest trees", "clustering means", "missing values"]
    r = GloveRetriever(p).fit(texts)
    assert r.search("trees", 1)[0][0] == 0 and r.search("missing", 1)[0][0] == 2


def test_dense_retriever_with_injected_encoder():
    def enc(xs):  # bag-of-letters toy embedding, deterministic
        return np.array([[x.lower().count(c) for c in "aeiou"] + [1.0] for x in xs], dtype=np.float32)

    r = SentenceTransformerRetriever("fake/model", query_prefix="", encode_fn=enc).fit(TEXTS)
    assert r.name == "dense:model" and len(r.search("anything", 3)) == 3
    assert r.search(TEXTS[2], 1)[0][0] == 2  # a text is its own nearest neighbour
    seen = []
    pr = SentenceTransformerRetriever("fake/model", query_prefix="query: ", encode_fn=lambda xs: (seen.append(xs[0]), enc(xs))[1])
    pr.fit(TEXTS)
    pr.scores("hello")
    assert seen[-1] == "query: hello"  # prefix applied to queries only


def test_cross_encoder_reranker_with_injected_scorer():
    rr = CrossEncoderReranker(score_fn=lambda pairs: [len(t) for _, t in pairs])
    cands = [(0, 0.9), (1, 0.8), (2, 0.7)]
    out = rr.rerank("q", cands, ["a", "abcdef", "abc"], top_k=2)
    assert [i for i, _ in out] == [1, 2]
