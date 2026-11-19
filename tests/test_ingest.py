from src.ingest import (Doc, chunk_fixed, chunk_sections, clean_rst, innermost_section, parse_sections)

RST = """.. _demo:

==========
Main Title
==========

.. currentmodule:: sklearn.tree

Intro text with a :ref:`wrapped reference
<some_label>` and :class:`~sklearn.tree.DecisionTreeClassifier` plus ``code``.

.. figure:: ../images/x.png
   :target: ../y.html
   :scale: 75

Alpha
-----

Alpha body paragraph one has enough words to count as real content here.

Alpha body paragraph two also has several words in it for the chunker.

Beta
~~~~

Beta is nested inside Alpha and talks about class_weight and warm_start.

Gamma
-----

Gamma body is a separate section after Alpha, with its own words too.
"""


def test_clean_rst_strips_markup_but_keeps_prose():
    t = clean_rst(RST)
    assert ":ref:" not in t and "<some_label>" not in t
    assert "wrapped reference" in t
    assert "DecisionTreeClassifier" in t and "sklearn.tree.DecisionTree" not in t
    assert "figure" not in t and ":target:" not in t and "x.png" not in t
    assert "``" not in t and "`code`" in t
    assert ".. _demo" not in t


def test_sections_levels_ranges_paths():
    t = clean_rst(RST)
    secs = parse_sections(t)
    by = {s.title: s for s in secs}
    assert [s.title for s in secs] == ["Main Title", "Alpha", "Beta", "Gamma"]
    assert by["Main Title"].level == 1 and by["Alpha"].level == 2 and by["Beta"].level == 3
    assert by["Beta"].path == ("Main Title", "Alpha", "Beta")
    # Alpha contains Beta; Gamma starts where Alpha ends
    assert by["Alpha"].start < by["Beta"].start < by["Alpha"].end == by["Gamma"].start
    assert innermost_section(secs, by["Beta"].start + 5).title == "Beta"


def _doc():
    t = clean_rst(RST)
    return {"demo": Doc("demo", t, parse_sections(t))}


def test_section_chunks_never_cross_headings():
    docs = _doc()
    d = docs["demo"]
    starts = [s.start for s in d.sections]
    for c in chunk_sections(docs, max_words=30):
        assert not any(c.start < st < c.end for st in starts), c
        assert c.heading.startswith("Main Title")


def test_chunks_carry_valid_offsets_and_heading_prefix():
    docs = _doc()
    d = docs["demo"]
    for ch in chunk_sections(docs, max_words=20) + chunk_fixed(docs, size=15, overlap=3):
        assert 0 <= ch.start < ch.end <= len(d.text)
        words = [w for w in d.text[ch.start:ch.end].split() if not set(w) <= set("=-~")]  # underlines are stripped by design
        assert words[0] in ch.text and words[-1] in ch.text
    assert chunk_sections(docs, 20, with_heading=True)[0].text.startswith("Main Title")


def test_fixed_chunks_overlap_and_cover_everything():
    docs = _doc()
    chunks = chunk_fixed(docs, size=10, overlap=2)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(docs["demo"].text.rstrip())
    assert all(b.start < a.end for a, b in zip(chunks, chunks[1:]))
