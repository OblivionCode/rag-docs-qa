"""Parse reStructuredText docs into clean text, sections and chunks.

Everything downstream works on the *cleaned* text of each file, and every
section and chunk carries character offsets into it. That is what lets the
evaluation decide whether a retrieved chunk overlaps the section that
actually answers a question, regardless of how the chunk was cut.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
# Directives whose body is layout/metadata, not prose: drop the whole block.
_DROP_DIRECTIVES = {
    "figure", "image", "currentmodule", "toctree", "include", "literalinclude",
    "plot", "autoclass", "autofunction", "automodule", "rst-class", "raw",
    "only", "glossary", "index", "tab", "tabs", "grid", "sidebar", "contents",
}
_RE_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+([\w-]+)::\s*(.*)$")
_RE_LABEL = re.compile(r"^\s*\.\.\s+_[^:]+:.*$")  # .. _label:  /  .. _name: url
_RE_COMMENT = re.compile(r"^\s*\.\.(\s.*)?$")  # bare ".." comment line
_RE_SUBST = re.compile(r"^\s*\.\.\s+\|[^|]+\|.*$")
_RE_ROLE_TEXT_TARGET = re.compile(r":[\w:.-]+:`([^`<]+?)\s*<[^>`]+>`")
_RE_ROLE = re.compile(r":[\w:.-]+:`([^`]+)`")
_RE_LINK = re.compile(r"`([^`<]+?)\s*<[^>`]+>`_{1,2}")
_RE_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+<>_.:])\1{2,}\s*$")


def _clean_inline(text: str) -> str:
    text = _RE_ROLE_TEXT_TARGET.sub(r"\1", text)

    def role(m: re.Match) -> str:
        inner = m.group(1).lstrip("~!")
        return inner.split(".")[-1] if m.group(0).startswith((":class:", ":func:", ":meth:", ":mod:")) else inner

    text = _RE_ROLE.sub(role, text)
    text = _RE_LINK.sub(r"\1", text)
    text = text.replace("``", "`")
    text = re.sub(r"\s+\|\w+\|", "", text)  # drop |substitutions|
    return text


def clean_rst(raw: str) -> str:
    """Strip Sphinx/RST markup that is noise for retrieval, keep headings and prose."""
    lines = raw.splitlines()
    out: list[str] = []
    skip_indent: int | None = None  # indentation of a dropped directive block

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if skip_indent is not None:
            if stripped == "" or indent > skip_indent:
                continue
            skip_indent = None

        if _RE_LABEL.match(line) or _RE_SUBST.match(line):
            continue
        m = _RE_DIRECTIVE.match(line)
        if m:
            name = m.group(2).lower()
            if name in _DROP_DIRECTIVES:
                skip_indent = len(m.group(1))
                continue
            # keep admonition / math / code bodies; keep the one-line argument as text
            arg = m.group(3).strip()
            if name in {"note", "warning", "topic", "dropdown", "versionadded", "versionchanged",
                        "deprecated", "seealso", "tip", "important", "rubric"} and arg:
                out.append(arg)
            continue
        if _RE_COMMENT.match(line):
            continue
        if re.match(r"^\s*:[\w-]+:(\s|$)", line) and indent > 0:  # directive option lines (:target: ...)
            continue
        out.append(line)

    text = _clean_inline("\n".join(out))  # whole text, so roles wrapped across lines are cleaned too
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------
@dataclass
class Section:
    title: str
    level: int  # 1 = top
    start: int  # char offset of heading line
    end: int  # char offset where the section (incl. children) ends
    path: tuple[str, ...] = ()


def parse_sections(text: str) -> list[Section]:
    """Find headings via RST underline (optionally overline) styles."""
    lines = text.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    found: list[tuple[int, str, str]] = []  # (line_index, title, style)
    i = 0
    while i < len(lines) - 1:
        cur, nxt = lines[i].strip(), lines[i + 1]
        # underline-only style
        if cur and not _RE_UNDERLINE.match(cur) and _RE_UNDERLINE.match(nxt.strip()) \
                and len(nxt.strip()) >= len(cur) - 1 and len(cur) < 120:
            overlined = i > 0 and _RE_UNDERLINE.match(lines[i - 1].strip()) is not None
            style = ("o" if overlined else "u") + nxt.strip()[0]
            found.append((i, cur, style))
            i += 2
            continue
        i += 1

    levels: dict[str, int] = {}
    heads = []
    for idx, title, style in found:
        levels.setdefault(style, len(levels) + 1)
        start_line = idx - 1 if style[0] == "o" and idx > 0 else idx
        heads.append((offsets[start_line], title, levels[style]))

    sections: list[Section] = []
    for k, (start, title, level) in enumerate(heads):
        end = len(text)
        for later_start, _, later_level in heads[k + 1:]:
            if later_level <= level:
                end = later_start
                break
        sections.append(Section(title=title, level=level, start=start, end=end))

    stack: list[Section] = []
    for s in sections:
        while stack and stack[-1].level >= s.level:
            stack.pop()
        stack.append(s)
        s.path = tuple(x.title for x in stack)
    return sections


def innermost_section(sections: list[Section], offset: int) -> Section | None:
    best = None
    for s in sections:
        if s.start <= offset < s.end and (best is None or s.level >= best.level):
            best = s
    return best


# ----------------------------------------------------------------------------
# Documents and chunks
# ----------------------------------------------------------------------------
@dataclass
class Doc:
    name: str
    text: str
    sections: list[Section] = field(default_factory=list)


@dataclass
class Chunk:
    id: int
    doc: str
    start: int
    end: int
    text: str  # what the retriever indexes and the LLM reads (may include heading prefix)
    heading: str  # "Decision Trees > Classification"


def load_corpus(corpus_dir: str | Path) -> dict[str, Doc]:
    docs = {}
    for p in sorted(Path(corpus_dir).glob("*.rst")):
        text = clean_rst(p.read_text(encoding="utf-8"))
        docs[p.stem] = Doc(p.stem, text, parse_sections(text))
    if not docs:
        raise FileNotFoundError(f"no .rst files in {corpus_dir}; run `python fetch_corpus.py`")
    return docs


_WORD = re.compile(r"\S+")
_RE_UNDERLINE_LINE = re.compile(r"^([=\-~^\"'`#*+<>_.:])\1{2,}[ \t]*\n?", re.M)


def _body(text: str) -> str:
    """Chunk text shown to retrievers/LLM: drop RST heading underlines (offsets are unaffected)."""
    return _RE_UNDERLINE_LINE.sub("", text).strip()


def _word_spans(text: str):
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def chunk_fixed(docs: dict[str, Doc], size: int = 200, overlap: int = 40, with_heading: bool = False) -> list[Chunk]:
    """Sliding window over words, ignoring document structure (the naive baseline)."""
    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    for d in docs.values():
        spans = _word_spans(d.text)
        for i in range(0, max(1, len(spans)), step):
            window = spans[i:i + size]
            if not window:
                break
            s, e = window[0][0], window[-1][1]
            sec = innermost_section(d.sections, s)
            heading = " > ".join(sec.path) if sec else d.name
            body = _body(d.text[s:e])
            chunks.append(Chunk(len(chunks), d.name, s, e, f"{heading}\n{body}" if with_heading else body, heading))
            if i + size >= len(spans):
                break
    return chunks


def chunk_sections(docs: dict[str, Doc], max_words: int = 200, overlap: int = 0, with_heading: bool = True) -> list[Chunk]:
    """Structure-aware: never cross a heading; split long sections on paragraph
    boundaries to ~max_words; prefix each chunk with its heading path."""
    chunks: list[Chunk] = []
    for d in docs.values():
        bounds = sorted({0, len(d.text), *[s.start for s in d.sections]})
        for a, b in zip(bounds, bounds[1:]):
            region = d.text[a:b]
            if not region.strip():
                continue
            sec = innermost_section(d.sections, a)
            heading = " > ".join(sec.path) if sec else d.name
            paras = [(m.start() + a, m.end() + a) for m in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", region)]
            cur: list[tuple[int, int]] = []
            words = 0
            groups: list[list[tuple[int, int]]] = []
            for p in paras:
                w = len(d.text[p[0]:p[1]].split())
                if cur and words + w > max_words:
                    groups.append(cur)
                    cur, words = (cur[-1:] if overlap else []), (len(d.text[cur[-1][0]:cur[-1][1]].split()) if overlap and cur else 0)
                cur.append(p)
                words += w
            if cur:
                groups.append(cur)
            for g in groups:
                s, e = g[0][0], g[-1][1]
                body = _body(d.text[s:e])
                if len(body.split()) < 8 and len(groups) > 1:
                    continue
                chunks.append(Chunk(len(chunks), d.name, s, e, f"{heading}\n{body}" if with_heading else body, heading))
    return chunks


def make_chunks(docs: dict[str, Doc], strategy: str, size: int, with_heading: bool | None = None) -> list[Chunk]:
    """strategy: 'fixed' (sliding word window) or 'section' (structure-aware).
    with_heading prepends the heading path to the chunk text (default: off for fixed, on for section)."""
    if strategy == "fixed":
        return chunk_fixed(docs, size=size, overlap=size // 5, with_heading=bool(with_heading))
    if strategy == "section":
        return chunk_sections(docs, max_words=size, with_heading=True if with_heading is None else with_heading)
    raise ValueError(f"unknown chunking strategy {strategy!r}")
