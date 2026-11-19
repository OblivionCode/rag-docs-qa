"""Retrievers. Every retriever implements fit(texts) and scores(query) -> np.ndarray
over all chunks; `search` just takes the top-k of those scores.

Runnable with no model downloads: BM25, TF-IDF, LSA, GloVe-average, hybrid (RRF).
Optional (need weights from Hugging Face, so not run in the sandbox this repo was
built in): SentenceTransformerRetriever and CrossEncoderReranker.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

SEED = 42
_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def make_tokenizer(stem: bool = False, drop_stop: bool = True):
    """Lowercase tokens. Identifiers like `class_weight` also emit `class` and
    `weight`, so natural-language questions can match code-style names."""
    stemmer = None
    if stem:
        import snowballstemmer

        stemmer = snowballstemmer.stemmer("english")

    def tokenize(text: str) -> list[str]:
        out: list[str] = []
        for tok in _TOKEN.findall(text.lower()):
            parts = [tok]
            if "_" in tok:
                parts += [p for p in tok.split("_") if p]
            for p in parts:
                if drop_stop and p in ENGLISH_STOP_WORDS:
                    continue
                out.append(p)
        if stemmer is not None:
            out = stemmer.stemWords(out)
        return out

    return tokenize


class Retriever:
    name = "base"

    def fit(self, texts: list[str]) -> "Retriever":
        raise NotImplementedError

    def scores(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        s = self.scores(query)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return [(int(i), float(s[i])) for i in top]


class BM25Retriever(Retriever):
    """Okapi BM25 with the non-negative idf variant, vectorized with scipy sparse."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, stem: bool = False):
        self.k1, self.b, self.stem = k1, b, stem
        self.name = f"bm25{'+stem' if stem else ''}"
        self._tok = make_tokenizer(stem=stem)

    def fit(self, texts):
        from sklearn.feature_extraction.text import CountVectorizer

        self.cv = CountVectorizer(tokenizer=self._tok, lowercase=False, token_pattern=None)
        tf = self.cv.fit_transform(texts).tocsr().astype(np.float64)
        n_docs = tf.shape[0]
        df = np.diff(tf.tocsc().indptr)
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avg = dl.mean()
        # precompute the saturated tf part: tf*(k1+1) / (tf + k1*(1-b+b*dl/avg))
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[rows] / avg)
        w = tf.copy()
        w.data = tf.data * (self.k1 + 1) / denom
        self.W = w.tocsc()
        return self

    def scores(self, query):
        q = self.cv.transform([query]).tocoo()
        s = np.zeros(self.W.shape[0])
        for j in q.col:  # query term frequency is ignored (standard for short queries)
            col = self.W.getcol(j)
            s[col.indices] += col.data * self.idf[j]
        return s


class TfidfRetriever(Retriever):
    name = "tfidf"

    def __init__(self, ngram=(1, 2), stem: bool = False):
        self.ngram, self.stem = ngram, stem
        self._tok = make_tokenizer(stem=stem)

    def fit(self, texts):
        self.vec = TfidfVectorizer(tokenizer=self._tok, lowercase=False, token_pattern=None,
                                   ngram_range=self.ngram, sublinear_tf=True, min_df=1)
        self.X = self.vec.fit_transform(texts)
        return self

    def scores(self, query):
        return (self.X @ self.vec.transform([query]).T).toarray().ravel()


class LSARetriever(Retriever):
    """TF-IDF then truncated SVD: a classic semantic baseline that can match
    synonyms that co-occur, with no pretrained model."""

    def __init__(self, dim: int = 256, stem: bool = False):
        self.dim, self.stem = dim, stem
        self.name = f"lsa{dim}"
        self._tok = make_tokenizer(stem=stem)

    def fit(self, texts):
        self.vec = TfidfVectorizer(tokenizer=self._tok, lowercase=False, token_pattern=None,
                                   sublinear_tf=True, min_df=2 if len(texts) >= 50 else 1)
        X = self.vec.fit_transform(texts)
        dim = min(self.dim, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=dim, random_state=SEED)
        self.Z = normalize(self.svd.fit_transform(X))
        return self

    def scores(self, query):
        q = normalize(self.svd.transform(self.vec.transform([query])))
        return (self.Z @ q.T).ravel()


def load_glove(path: str | Path, vocab: set[str]) -> dict[str, np.ndarray]:
    """Read only the words we need from a GloVe/word2vec text file (.gz ok)."""
    vecs: dict[str, np.ndarray] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            word, _, rest = line.partition(" ")
            if word in vocab:
                vecs[word] = np.asarray(rest.split(), dtype=np.float32)
    return vecs


class GloveRetriever(Retriever):
    """Static word embeddings: idf-weighted mean of GloVe vectors for each chunk.
    Captures some word similarity (e.g. 'trim' ~ 'prune') but no word order or context."""

    name = "glove"

    def __init__(self, glove_path: str | Path):
        self.path = glove_path
        self._tok = make_tokenizer(drop_stop=True)

    def fit(self, texts):
        toks = [self._tok(t) for t in texts]
        vocab = {w for t in toks for w in t}
        self.vecs = load_glove(self.path, vocab)
        self.dim = len(next(iter(self.vecs.values())))
        n = len(texts)
        df: dict[str, int] = {}
        for t in toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: np.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        self.Z = normalize(np.vstack([self._embed(t) for t in toks]))
        return self

    def _embed(self, toks: list[str]) -> np.ndarray:
        acc, wsum = np.zeros(self.dim, dtype=np.float64), 0.0
        for w in toks:
            v = self.vecs.get(w)
            if v is not None:
                wt = self.idf.get(w, 1.0)
                acc += wt * v
                wsum += wt
        return acc / wsum if wsum else acc

    def scores(self, query):
        q = self._embed(self._tok(query))
        norm = np.linalg.norm(q)
        return self.Z @ (q / norm) if norm else np.zeros(self.Z.shape[0])


class SentenceTransformerRetriever(Retriever):
    """Dense retrieval with a sentence-embedding model.

    NOT run in the sandbox this repo was built in (no access to model weights);
    run it locally with `python evaluate.py --dense BAAI/bge-small-en-v1.5`.
    `encode_fn(list[str]) -> np.ndarray` can be injected for tests.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", query_prefix: str = "", encode_fn=None):
        self.model_name, self.query_prefix = model_name, query_prefix
        self.name = "dense:" + model_name.split("/")[-1]
        self._encode = encode_fn

    def _ensure_model(self):
        if self._encode is None:
            from sentence_transformers import SentenceTransformer  # lazy: heavy optional dependency

            model = SentenceTransformer(self.model_name)
            self._encode = lambda xs: model.encode(xs, normalize_embeddings=True, show_progress_bar=False)

    def fit(self, texts):
        self._ensure_model()
        self.Z = normalize(np.asarray(self._encode(texts), dtype=np.float32))
        return self

    def scores(self, query):
        q = normalize(np.asarray(self._encode([self.query_prefix + query]), dtype=np.float32))
        return (self.Z @ q.T).ravel()


class RRFRetriever(Retriever):
    """Reciprocal rank fusion: score = sum_r 1 / (c + rank_r). Parameter-free apart
    from c, needs no score calibration across retrievers."""

    def __init__(self, members: list[Retriever], c: int = 60, depth: int = 100, weights: list[float] | None = None):
        self.members, self.c, self.depth = members, c, depth
        self.weights = weights or [1.0] * len(members)
        self.name = "rrf(" + "+".join(m.name for m in members) + ")"

    def fit(self, texts):
        self.n = len(texts)
        return self  # members are assumed to be fitted already

    def scores(self, query):
        out = np.zeros(self.n)
        for m, w in zip(self.members, self.weights):
            s = m.scores(query)
            top = np.argsort(-s, kind="stable")[: self.depth]
            out[top] += w / (self.c + np.arange(1, len(top) + 1))
        return out


class CrossEncoderReranker:
    """Re-score the top-n candidates with a cross-encoder. Optional and not run in
    the sandbox (needs weights). `score_fn(list[(query, text)]) -> list[float]` can be injected."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", score_fn=None):
        self.model_name, self._score = model_name, score_fn

    def rerank(self, query: str, candidates: list[tuple[int, float]], texts: list[str], top_k: int = 5):
        if self._score is None:
            from sentence_transformers import CrossEncoder

            ce = CrossEncoder(self.model_name)
            self._score = lambda pairs: ce.predict(pairs)
        s = self._score([(query, texts[i]) for i, _ in candidates])
        order = np.argsort(-np.asarray(s), kind="stable")[:top_k]
        return [(candidates[j][0], float(s[j])) for j in order]
