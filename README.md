# RAG over the scikit-learn docs: building and *measuring* a retrieval pipeline

A question-answering system over the scikit-learn 1.6.1 user guide (42 files, 128k words) that
retrieves the relevant sections and generates cited answers. The point of the project is the
evaluation: 103 questions with gold sections, a dev/test protocol, bootstrap confidence
intervals, and an honest look at what worked and what did not.

![Retrieval comparison on the test split](results/test_retrieval_comparison.png)

**Headline:** tuned BM25 over section-aware chunks was the strongest retriever I could run:
**Recall@5 = 0.78, MRR@10 = 0.67** on 51 held-out questions (95% CI 0.55 to 0.78). It beat TF-IDF,
LSA, averaged GloVe vectors and three of four hybrids with statistical significance. A hybrid of
BM25 and LSA was statistically indistinguishable from it. Dense transformer retrievers and LLM answer
generation are implemented but **not evaluated here** (see [What I did not run](#what-i-did-not-run)).

## What is in the pipeline

| Stage | Implementation |
|---|---|
| Ingest | Parse reStructuredText, strip Sphinx markup, find sections, keep character offsets (`src/ingest.py`) |
| Chunking | Fixed sliding word windows vs structure-aware chunks that never cross a heading and carry their heading path |
| Retrieval | BM25 (written from scratch, sparse-matrix), TF-IDF, LSA, GloVe average, reciprocal rank fusion; optional sentence-transformers retriever and cross-encoder reranker (`src/retrievers.py`) |
| Generation | Grounded prompt with numbered passages, required `[n]` citations, abstention rule; Anthropic, OpenAI-compatible (OpenAI, Ollama, Groq, vLLM...) and an offline extractive mode (`src/rag.py`) |
| Checks | Citation verifier: flags cited passages that do not exist and substantive sentences with no citation |
| Evaluation | Gold-section relevance, Recall@k, MRR, nDCG, bootstrap CIs, paired comparisons (`src/evaluation.py`, `evaluate.py`) |
| Demo | Streamlit app showing the answer, citation check and expandable source passages (`app.py`) |

## How retrieval was evaluated

- **103 questions**, each written about one section of the guide and phrased in natural language
  (not copying the heading) so keyword matching gets no free ride. Each has a gold section and an
  `evidence` regex; `eval/validate_eval.py` checks that the regex really appears inside the gold
  section, which caught 7 labeling problems while building the set (3 evidence phrases that were not in the section, 4 ambiguous section names).
- **Split:** 52 dev / 51 test, disjoint. Everything was chosen on dev; the test split was used only
  to report (the protocol is in `evaluate.py`).
- **Relevance** is judged on character offsets: a retrieved chunk is a hit if it overlaps the gold
  section by at least 50% of the smaller of the two. This makes results comparable across chunking
  strategies.
- **Metrics:** Recall@k (any gold hit in the top k), MRR@10, nDCG@10, with 1,000-sample bootstrap
  confidence intervals over questions and paired bootstrap differences between systems.
- Two full runs give identical metrics and the same selected configuration (everything is seeded; only latency varies).

## Results

### Test split (n = 51), configuration chosen on dev

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|
| **BM25** (k1=1.5, b=0.6, stemming) | 0.57 | 0.78 (0.65-0.88) | 0.88 | **0.671** (0.55-0.78) | 0.566 | 0.5 |
| Hybrid BM25 + LSA | 0.49 | 0.80 (0.69-0.90) | 0.88 | 0.627 (0.51-0.73) | 0.554 | 2.2 |
| LSA (256 dims) | 0.33 | 0.73 (0.61-0.84) | 0.84 | 0.494 (0.39-0.60) | 0.466 | 1.7 |
| Hybrid all four | 0.33 | 0.73 (0.59-0.84) | 0.86 | 0.482 (0.37-0.58) | 0.448 | 3.9 |
| TF-IDF (1-2 grams) | 0.29 | 0.71 (0.57-0.82) | 0.84 | 0.458 (0.36-0.56) | 0.416 | 1.0 |
| Hybrid BM25 + LSA + GloVe | 0.33 | 0.61 (0.47-0.75) | 0.80 | 0.450 (0.33-0.56) | 0.397 | 2.3 |
| Hybrid BM25 + GloVe | 0.33 | 0.47 (0.33-0.61) | 0.61 | 0.408 (0.28-0.53) | 0.327 | 0.7 |
| GloVe average | 0.08 | 0.29 (0.18-0.41) | 0.33 | 0.163 (0.09-0.25) | 0.144 | 0.1 |
| Random | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.0 |

Paired MRR@10 difference against BM25 (bootstrap 95% CI): LSA -0.18 (-0.29 to -0.08), TF-IDF -0.21
(-0.31 to -0.12), GloVe average -0.51 (-0.63 to -0.39), hybrid BM25+LSA **-0.04 (-0.12 to +0.02, not
significant)**. Full table: `results/test_paired_diff_vs_bm25.csv`.

### What I take from it

1. **BM25 is a very strong baseline on technical documentation.** The questions and the docs share
   identifiers and vocabulary (`warm_start`, `class_weight`, `RidgeCV`). The tokenizer also splits
   `class_weight` into `class` and `weight`, which lets natural-language questions hit code-style names.
2. **Fusing in a weak retriever hurts.** Equal-weight rank fusion of BM25 and averaged GloVe dropped
   MRR from 0.67 to 0.41. Hybrid search is not automatically better than its best member.
3. **Chunking mattered less than I expected, except for very large chunks** (dev, mean MRR over the four
   base retrievers): section-200 0.456, fixed-100 0.454, section-100 0.451, section-300 0.446,
   section-200 without heading prefix 0.435, fixed-200 0.416, fixed-300 0.378. The top four are within
   noise of each other. Prefixing each chunk with its heading path helped the embedding-style
   retrievers (LSA 0.445 to 0.488, GloVe 0.241 to 0.282) and was neutral for BM25.
4. **Tuning bought little.** Over a 32-point BM25 grid, dev MRR ranged only 0.54 to 0.59; the selected
   setting (stemming, b=0.6) is 0.589 against 0.548 for the defaults. A difference that size on 52
   questions is within noise.
5. **50-question evaluations are noisy.** On dev, BM25 led TF-IDF by 0.04 MRR and LSA trailed TF-IDF;
   on test BM25 led by 0.21 and LSA beat TF-IDF. The ordering at the top held, the magnitudes did not.
   Read the confidence intervals, not the third decimal.

### Error analysis

BM25 put its first relevant chunk below rank 3 (or outside the top 10) on 14 of 51 test questions.
Five questions were missed at k=5 by all four base retrievers. Typical causes:

- **Vocabulary mismatch:** "which model mixes two kinds of penalty" (the docs say "L1 and L2-norm
  regularization"); "discard columns that barely change" (the docs say "low variance").
  This is where a dense model should help, and where GloVe averaging did not.
- **Label noise (strict gold labels):** for "which data structures speed up neighbor lookups" the top hit
  was the *Ball Tree* section, a defensible answer that the gold label (the *KDTree and BallTree
  Classes* section) does not credit; similar for the hashing-trick section on a feature-hashing
  question, and for parent or overview sections on a few others. I did not relabel after seeing test
  failures, so the numbers above are a **lower bound** on usefulness.
- **Wrong level of section:** the right content sometimes sits in an intro chunk that is judged
  against a more specific gold subsection.

Per-question results: `results/test_per_question.csv`.

### Generation (structural checks only)

`evaluate_generation.py` checks citation structure and abstention without needing a judge model. With no
API key in the build environment, only the offline extractive baseline was run:

| Mode | Cited a gold section (51 test Qs) | Abstains on 10 off-topic Qs |
|---|---|---|
| Extractive baseline (top-2 passages quoted, no LLM) | 71% | 0% (it always answers) |

That 0% is the point of a grounded LLM with an abstention rule: a retriever alone will happily return
passages for "Who won the 2018 World Cup?". Run the script with your own key to fill in the LLM row.

## What I did not run

Being explicit, because these are the natural next questions:

- **Dense transformer retrieval and cross-encoder reranking** are implemented
  (`SentenceTransformerRetriever`, `CrossEncoderReranker`) and unit-tested with stub encoders, but the
  environment this was built in could not download model weights, so **there are no results for them.**
  Run `pip install sentence-transformers && python evaluate.py --dense BAAI/bge-small-en-v1.5` to add
  dense and BM25+dense rows to the test table. I expect dense models to help on the vocabulary-mismatch
  failures above; that is a hypothesis, not a result.
- **LLM answer quality.** The Anthropic and OpenAI-compatible generators were tested against a local mock
  server (request shape, headers, response parsing), not the live APIs. Whether answers are *correct*
  needs a human or judge-model review of `results/generation_*.csv`.
- Single corpus (one library's docs), one question-writer, ~50 test questions.

## Run it

```bash
pip install -r requirements.txt
python fetch_corpus.py              # scikit-learn 1.6.1 user guide + GloVe vectors (134 MB)
python eval/validate_eval.py        # checks every gold label against the corpus
python evaluate.py                  # full dev/test protocol, about 1 minute
python -m pytest -q                 # 23 tests
python ask.py "How do I add more trees to an already trained forest?"
streamlit run app.py
```

For written answers instead of quoted passages, set a key before `ask.py` or the app:

```bash
export ANTHROPIC_API_KEY=...                       # optionally RAG_MODEL=<model name>
# or any OpenAI-compatible server, including a local one:
export OPENAI_BASE_URL=http://localhost:11434/v1 RAG_MODEL=llama3.1
```

## Repository layout

```
src/ingest.py         RST cleaning, section parsing, chunkers (offsets preserved)
src/retrievers.py     BM25, TF-IDF, LSA, GloVe, RRF, optional dense + cross-encoder
src/rag.py            prompt, generators, citation verifier, pipeline
src/evaluation.py     relevance judging, metrics, bootstrap
evaluate.py           dev/test protocol; writes results/
evaluate_generation.py structural checks of generated answers
eval/                 questions.jsonl (103), unanswerable.jsonl (10), builder and validator
results/              every table and figure in this README
tests/                23 tests (cleaning, chunk invariants, metrics, retrievers, mock-server generators)
app.py, ask.py        demo UI and CLI
fetch_corpus.py       downloads the corpus at a pinned release tag (not committed)
```

## Data and licensing

scikit-learn documentation: BSD-3-Clause (<https://github.com/scikit-learn/scikit-learn>), fetched at
tag 1.6.1 and not redistributed here. GloVe 6B vectors (Pennington et al., 2014) are downloaded from the
gensim-data release under the Public Domain Dedication and License.

## Development notes

Built with AI assistance (Claude). The eval questions and gold labels were drafted with that help and
machine-validated against the corpus, but a human should review and rewrite some of them before treating
the numbers as final. The dev/test protocol, code and results in this repo are all reproducible from the commands above.
