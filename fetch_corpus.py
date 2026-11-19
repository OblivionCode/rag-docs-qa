"""Download the scikit-learn user guide (RST sources) at a pinned release tag, plus GloVe vectors.

The corpus is not committed to the repo; this script recreates it exactly.
scikit-learn is BSD-3-Clause licensed: https://github.com/scikit-learn/scikit-learn
"""
import sys
import urllib.request
from pathlib import Path

TAG = "1.6.1"
BASE = f"https://raw.githubusercontent.com/scikit-learn/scikit-learn/{TAG}/doc/modules/"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "corpus"
GLOVE_URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"
GLOVE_OUT = ROOT / "data" / "glove" / "glove-wiki-gigaword-100.gz"  # 134 MB, GloVe 6B (Pennington et al., PDDL license)

FILES = """
calibration clustering compose covariance cross_decomposition cross_validation
decomposition density ensemble feature_extraction feature_selection gaussian_process
grid_search impute isotonic kernel_approximation kernel_ridge lda_qda learning_curve
linear_model manifold metrics mixture model_evaluation model_persistence multiclass
naive_bayes neighbors neural_networks outlier_detection partial_dependence
permutation_importance pipeline preprocessing preprocessing_targets random_projection
semi_supervised sgd svm tree unsupervised_reduction classification_threshold
biclustering array_api
""".split()


def fetch_glove():
    if GLOVE_OUT.exists():
        print(f"GloVe already at {GLOVE_OUT}")
        return
    GLOVE_OUT.parent.mkdir(parents=True, exist_ok=True)
    print("downloading GloVe 100d vectors (134 MB)...")
    urllib.request.urlretrieve(GLOVE_URL, GLOVE_OUT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok, missing = 0, []
    for name in FILES:
        try:
            with urllib.request.urlopen(BASE + name + ".rst", timeout=30) as r:
                (OUT / f"{name}.rst").write_bytes(r.read())
            ok += 1
        except Exception as e:  # noqa: BLE001
            missing.append(f"{name} ({e})")
    print(f"downloaded {ok}/{len(FILES)} files to {OUT}")
    if missing:
        print("missing:", *missing, sep="\n  ")
    if "--no-glove" not in sys.argv:
        fetch_glove()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
