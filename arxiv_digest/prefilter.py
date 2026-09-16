"""Stage 1.5: cheap semantic prefilter.

A week of cs.LG + cs.CV + eess.IV is ~1400 abstracts. Scoring all of them with
a frontier model costs a few dollars a run and mostly buys you 1300 confident
rejections. This module cuts the candidate set with local embeddings first, so
the model only reads papers that are plausibly relevant.

It is also the baseline stage 4 has to beat. If agentic ranking cannot
outperform cosine similarity against your profile, that is the finding -- which
is why this runs as real infrastructure rather than as a comparison script
written at the end.

Recall is the thing to watch. A paper that lands 151st here never reaches the
model, so a prefilter miss is invisible in the digest. `--keep` is the knob;
measure recall against a labeled set before you tighten it.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # numpy arrives with sentence-transformers, imported lazily
    import numpy as np

from .models import Paper

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING = re.compile(r"^\s*#+\s*(?P<text>.*?)\s*$")
_BULLET = re.compile(r"^\s*[-*+]\s+")
# Headings that mark a section as things to push *down*, not up.
_NEGATIVE_HEADING = re.compile(
    r"do\s*n[o']?t\s+(need|want|care)|not\s+interested|exclude|avoid|ignore|irrelevant",
    re.IGNORECASE,
)

INSTALL_HINT = (
    "The prefilter needs sentence-transformers:\n"
    "    pip install sentence-transformers\n"
    "Or skip it for this run with --no-prefilter (scores every abstract, "
    "which on a full week costs real money)."
)


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(INSTALL_HINT) from exc
    # stderr, not stdout. This module is imported by the MCP server, which
    # speaks JSON-RPC over stdout -- a single stray print corrupts the stream
    # and drops the connection. Progress messages are diagnostics, not data.
    print(f"  loading {model_name} (first run downloads ~80MB)...", file=sys.stderr)
    return SentenceTransformer(model_name)


def document_text(paper: Paper) -> str:
    """How a paper is turned into text for embedding.

    Shared so that vectors cached by the digest and vectors used by search are
    built identically. Change this and stored embeddings become stale.
    """
    return f"{paper.title}. {paper.abstract}"


def embed_texts(
    texts: list[str], model_name: str, batch_size: int = 64
) -> "np.ndarray":
    """Encode text to L2-normalised vectors. Loads the model on first use."""
    model = _load_model(model_name)
    return model.encode(
        texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False
    )


def embed_papers(
    papers: list[Paper], model_name: str, batch_size: int = 64
) -> dict[str, "np.ndarray"]:
    """Vectors for papers, keyed by arXiv id, ready for `Store.set_embeddings`."""
    if not papers:
        return {}
    vectors = embed_texts([document_text(p) for p in papers], model_name, batch_size)
    return {paper.arxiv_id: vectors[i] for i, paper in enumerate(papers)}


def parse_profile(profile: str) -> tuple[list[str], list[str]]:
    """Split the profile into (wanted, unwanted) statements.

    Three things this has to get right, each of which was a bug first:

    - Markdown is hard-wrapped, so a sentence spans several lines. Embedding
      the lines separately embeds fragments. Blank lines and bullets end a
      statement; a bare newline does not.
    - Headings are labels, not content. "What I work on" describes nothing.
    - The "what I do not need" section is the opposite of a search query.
      Embedded as a positive it surfaces exactly the papers you excluded.

    Statements are kept separate rather than averaged into one vector: a paper
    only has to match one thing you care about, and averaging turns a sharp
    profile into a vague one.
    """
    wanted: list[str] = []
    unwanted: list[str] = []
    target = wanted
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        statement = " ".join(" ".join(buffer).split())
        buffer.clear()
        # Fragments shorter than this carry no usable signal.
        if len(statement.split()) >= 4:
            target.append(statement)

    for raw in _COMMENT.sub("", profile).splitlines():
        heading = _HEADING.match(raw)
        if heading:
            flush()
            target = unwanted if _NEGATIVE_HEADING.search(heading.group("text")) else wanted
            continue
        if not raw.strip():
            flush()
            continue
        if _BULLET.match(raw):
            flush()
            buffer.append(_BULLET.sub("", raw))
            continue
        buffer.append(raw)
    flush()

    return wanted, unwanted


def profile_chunks(profile: str) -> list[str]:
    """Just the positive statements. Used to sanity-check a filled-in profile."""
    return parse_profile(profile)[0]


def rank(
    papers: list[Paper],
    profile: str,
    keep: int | None,
    model_name: str,
    batch_size: int = 64,
) -> list[tuple[Paper, float]]:
    """Return the `keep` best papers as (paper, score), best first.

    `keep=None` returns the full ranking, which is what the label sampler
    needs -- it has to see the papers this filter would have thrown away.

    Score is `max similarity to anything you want` minus `max similarity to
    anything you excluded`, so a paper that looks like both is ranked below one
    that only looks like the first.
    """
    wanted, unwanted = parse_profile(profile)
    if not wanted:
        raise ValueError(
            "No usable content in profile.md -- it is all headings and comments. "
            "Copy profile.example.md and fill in the sections."
        )
    if not papers:
        return []

    model = _load_model(model_name)
    documents = [document_text(p) for p in papers]

    print(
        f"  embedding {len(wanted)} wanted / {len(unwanted)} unwanted statements "
        f"and {len(papers)} abstracts...",
        file=sys.stderr,
    )
    doc_vectors = model.encode(
        documents, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False
    )
    want_vectors = model.encode(wanted, normalize_embeddings=True, batch_size=batch_size)

    # Normalized vectors, so a dot product is cosine similarity.
    scores = (doc_vectors @ want_vectors.T).max(axis=1)
    if unwanted:
        avoid_vectors = model.encode(unwanted, normalize_embeddings=True, batch_size=batch_size)
        scores = scores - (doc_vectors @ avoid_vectors.T).max(axis=1)

    ranked = sorted(zip(papers, (float(s) for s in scores)), key=lambda x: -x[1])
    return ranked[:keep]
