"""Tests for preference learning.

Synthetic 2-D vectors throughout, so every expected ranking can be worked out
by hand. The important test is `test_self_match_inflates_naive_evaluation`: a
rated paper sits in its own liked pool, matches itself at similarity 1.0, and
scores brilliantly. An evaluation that forgets to hold it out reports a result
that looks excellent and means nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from arxiv_digest import preferences
from arxiv_digest.preferences import Pools


def unit(x, y):
    v = np.array([x, y], dtype=np.float32)
    return v / np.linalg.norm(v)


# Three documents: one purely "A", one purely "B", one in between.
DOC_IDS = ["doc_a", "doc_mid", "doc_b"]
DOCS = np.stack([unit(1, 0), unit(1, 1), unit(0, 1)])

A = unit(1, 0).reshape(1, -1)
B = unit(0, 1).reshape(1, -1)
EMPTY = np.zeros((0, 2), dtype=np.float32)


def pools(**kw):
    base = dict(wanted=A, unwanted=EMPTY, liked=EMPTY, liked_ids=[],
                disliked=EMPTY, disliked_ids=[])
    base.update(kw)
    return Pools(**base)


# --- the baseline is a special case ---------------------------------------


def test_zero_weights_reproduce_the_plain_prefilter():
    """alpha=beta=0 must be byte-identical to profile-only scoring."""
    p = pools(liked=B, liked_ids=["x"], disliked=B, disliked_ids=["y"])
    learned = preferences.score(DOCS, p, alpha=0.0, beta=0.0)
    profile_only = (DOCS @ A.T).max(axis=1)
    assert np.allclose(learned, profile_only)


def test_liked_examples_raise_similar_papers():
    plain = preferences.score(DOCS, pools(), alpha=0.0)
    learned = preferences.score(DOCS, pools(liked=B, liked_ids=["x"]), alpha=1.0)

    # doc_b is identical to the liked example, so it gains the most.
    assert learned[2] - plain[2] > learned[0] - plain[0]


def test_disliked_examples_push_similar_papers_down():
    plain = preferences.score(DOCS, pools(), beta=0.0)
    learned = preferences.score(DOCS, pools(disliked=B, disliked_ids=["x"]), beta=1.0)
    assert learned[2] < plain[2]
    assert learned[0] == pytest.approx(plain[0], abs=1e-6)


def test_ranking_can_be_flipped_by_ratings_alone():
    """The whole premise: feedback overrides what the profile says."""
    p = pools(wanted=A, liked=B, liked_ids=["x"])
    assert np.argmax(preferences.score(DOCS, p, alpha=0.0)) == 0
    assert np.argmax(preferences.score(DOCS, p, alpha=3.0)) == 2


# --- exclusion ------------------------------------------------------------


def test_exclude_removes_a_paper_from_its_own_pool():
    p = pools(liked=np.stack([unit(0, 1)]), liked_ids=["doc_b"])
    with_self = preferences.score(DOCS, p, alpha=1.0)
    without = preferences.score(DOCS, p, alpha=1.0, exclude_ids={"doc_b"})
    assert without[2] < with_self[2]


def test_excluding_everything_falls_back_to_the_profile():
    p = pools(liked=B, liked_ids=["only"])
    scores = preferences.score(DOCS, p, alpha=1.0, exclude_ids={"only"})
    assert np.allclose(scores, (DOCS @ A.T).max(axis=1))


def test_empty_pools_do_not_crash():
    scores = preferences.score(DOCS, pools(), alpha=1.0, beta=1.0)
    assert len(scores) == 3 and np.isfinite(scores).all()


# --- the trap this module exists to avoid ---------------------------------


def test_self_match_inflates_naive_evaluation():
    """Scoring a rated paper without holding it out is meaningless.

    doc_b is rated 5 and sits in the liked pool. Without exclusion it matches
    itself at 1.0 and ranks first despite the profile pointing at A. Held out,
    it falls back. Any eval that skips this reports a flattering fiction.
    """
    p = pools(wanted=A, liked=np.stack([unit(0, 1)]), liked_ids=["doc_b"])

    def rank_of_doc_b(scores):
        return int(np.where(np.argsort(-scores) == 2)[0][0])  # 0 = best

    naive = preferences.score(DOCS, p, alpha=2.0)
    honest = preferences.score(DOCS, p, alpha=2.0, exclude_ids={"doc_b"})

    # Its own vector contributes the entire alpha term: 2.0 of score for free.
    assert naive[2] == pytest.approx(2.0, abs=1e-5)
    assert honest[2] == pytest.approx(0.0, abs=1e-5)
    assert rank_of_doc_b(honest) > rank_of_doc_b(naive), \
        "held out, it drops -- an eval that skips this reports a fiction"


def test_leave_one_out_holds_each_paper_out():
    labels = {"doc_b": 5, "doc_a": 1}
    p = pools(wanted=A, liked=np.stack([unit(0, 1)]), liked_ids=["doc_b"])

    result = preferences.leave_one_out(DOC_IDS, DOCS, p, labels, alpha=2.0, beta=0.0)
    assert result["positives"] == 1
    assert result["ranks"]["doc_b"] > 0, "doc_b must not rank first off its own vector"


def test_leave_one_out_ignores_labels_for_unknown_papers():
    labels = {"doc_a": 5, "not_in_corpus": 5}
    result = preferences.leave_one_out(DOC_IDS, DOCS, pools(), labels, 0.0, 0.0)
    assert result["positives"] == 1


def test_leave_one_out_without_positives_is_nan_not_zero():
    """No positives means 'unmeasurable', which must not look like a score."""
    result = preferences.leave_one_out(DOC_IDS, DOCS, pools(), {"doc_a": 1}, 0.0, 0.0)
    assert np.isnan(result["mean_positive_rank"])


# --- sweep ----------------------------------------------------------------


def test_sweep_includes_the_unlearned_baseline():
    labels = {"doc_b": 5, "doc_a": 1}
    p = pools(wanted=A, liked=np.stack([unit(0, 1)]), liked_ids=["doc_b"])
    rows = preferences.sweep(DOC_IDS, DOCS, p, labels, [0.0, 1.0], [0.0])

    assert any(r["alpha"] == 0.0 and r["beta"] == 0.0 for r in rows), \
        "without a baseline row there is nothing to beat"


def test_sweep_is_ordered_best_first():
    labels = {"doc_b": 5, "doc_a": 1}
    p = pools(wanted=A, liked=np.stack([unit(0, 1)]), liked_ids=["doc_b"])
    rows = preferences.sweep(DOC_IDS, DOCS, p, labels, [0.0, 1.0, 2.0], [0.0, 1.0])

    ranks = [r["mean_positive_rank"] for r in rows]
    assert ranks == sorted(ranks)
