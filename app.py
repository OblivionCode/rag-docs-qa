"""Streamlit demo:  streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.rag import RAG, generator_from_env  # noqa: E402

st.set_page_config(page_title="scikit-learn docs Q&A", page_icon="📚", layout="centered")


@st.cache_resource(show_spinner="Building the index (about 5 seconds)...")
def load_rag():
    return RAG.from_config(ROOT, generator=generator_from_env())


rag = load_rag()
st.title("scikit-learn docs Q&A")
st.caption("Retrieval-augmented answers over the scikit-learn 1.6.1 user guide, with citations to the exact section.")

with st.sidebar:
    k = st.slider("Passages to retrieve", 1, 10, 5)
    st.markdown(f"**Answer mode:** `{rag.generator.name}`")
    if rag.generator.name == "extractive":
        st.info("No LLM key found, so answers are quotes from the best passages. "
                "Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` / `OPENAI_BASE_URL`) for written answers.")
    st.markdown("**Retriever:** tuned BM25 over section-aware chunks (see README for the evaluation).")

# Illustrative examples taken from the dev split (the retriever gets these right). They are NOT
# a measure of quality: see the README for the held-out test numbers, including the failures.
EXAMPLES = ["How should I split chronologically ordered data so the model never trains on the future?",
            "Which method learns a distance metric to improve nearest-neighbor accuracy?",
            "How can a classifier use its own confident predictions on unlabeled data to improve itself?"]
if "q" not in st.session_state:
    st.session_state.q = ""
cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    if c.button(ex, use_container_width=True):
        st.session_state.q = ex
question = st.text_input("Ask a question about scikit-learn", key="q")

if question.strip():
    a = rag.answer(question.strip(), k)
    st.subheader("Answer")
    st.write(a.text)
    c = a.citations
    if c["invalid"]:
        st.error(f"The answer cites passages that do not exist: {c['invalid']}")
    elif c["abstained"]:
        st.warning("The model says the documents do not contain the answer.")
    elif not c["ok"]:
        st.warning(f"{c['uncited_sentences']} sentence(s) have no citation. Check them against the sources.")
    st.subheader("Sources")
    for p in a.passages:
        cited = p.n in c["cited"]
        with st.expander(f"[{p.n}] {p.chunk.doc}.rst · {p.chunk.heading.split(' > ')[-1]}" + ("  ✅ cited" if cited else ""), expanded=cited):
            st.caption(p.chunk.heading + f"  ·  BM25 score {p.score:.2f}")
            body = p.chunk.text.split("\n")[1:]
            if body and body[0].strip() == p.chunk.heading.split(" > ")[-1].strip():
                body = body[1:]  # the section title is already in the expander label
            st.text("\n".join(body).strip())
