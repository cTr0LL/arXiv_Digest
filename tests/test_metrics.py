"""Tests for the IPW-weighted metrics.

Raw counts would flatter the prefilter, because the top of the ranking is
sampled thirty times more densely than the tail. These tests pin the weighting
with hand-computable numbers.
"""

from __future__ import annotations

from arxiv_digest import metrics


def row(rank, stratum, size, sampled, score):
    return {
        "arxiv_id": f"id{rank}",
        "label": "y" if score >= 4 else "n",
        "score": str(score),
        "rank": str(rank),
        "stratum": stratum,
        "stratum_size": str(size),
        "stratum_sampled": str(sampled),
        "prefilter_score": "0.5",
        "title": "t",
        "labeled_at": "2026-09-16T00:00:00+00:00",
    }


def test_weight_is_inverse_sampling_rate():
    # 50 papers, 25 labeled -> each stands in for 2.
    assert metrics._weight(row(0, "top", 50, 25, 5)) == 2.0
    # 1000 papers, 20 labeled -> each stands in for 50.
    assert metrics._weight(row(900, "tail", 1000, 20, 5)) == 50.0


def test_recall_counts_weighted_not_raw():
    rows = [
        row(10, "top", 50, 25, 5),    # weight 2, inside k=150
        row(20, "top", 50, 25, 5),    # weight 2, inside k=150
        row(900, "tail", 1000, 20, 5),  # weight 50, outside k=150
    ]
    result = metrics.recall_at(rows, 150)
    # 4 of 54 estimated relevant papers survive the cut.
    assert result["estimated_relevant_in_pool"] == 54.0
    assert result["estimated_relevant_kept"] == 4.0
    assert abs(result["recall"] - 4 / 54) < 1e-9


def test_one_tail_positive_dominates_the_estimate():
    """The documented high-variance behaviour, pinned so it cannot surprise."""
    dense = [row(i, "top", 50, 30, 5) for i in range(10)]
    assert metrics.recall_at(dense, 150)["recall"] == 1.0

    with_tail = dense + [row(900, "tail", 1000, 20, 5)]
    assert metrics.recall_at(with_tail, 150)["recall"] < 0.30
    assert metrics.recall_at(with_tail, 150)["tail_positives"] == 1


def test_maybe_scores_are_not_positives():
    # 'm' maps to 3, below RELEVANT_AT; it must not count as relevant.
    assert metrics.recall_at([row(1, "top", 50, 25, 3)], 150)["recall"] == 0.0
    assert metrics.precision_at([row(1, "top", 50, 25, 3)], 150)["precision"] == 0.0


def test_precision_is_unweighted_and_scoped_to_the_cut():
    rows = [
        row(1, "top", 50, 25, 5),
        row(2, "top", 50, 25, 1),
        row(900, "tail", 1000, 20, 5),  # outside k, must be ignored entirely
    ]
    result = metrics.precision_at(rows, 150)
    assert result["labeled_in_top_k"] == 2
    assert result["positives"] == 1
    assert result["precision"] == 0.5


def test_no_labels_inside_cut_is_not_a_crash():
    assert metrics.precision_at([row(900, "tail", 1000, 20, 5)], 150)["precision"] == 0.0
    assert metrics.recall_at([], 150)["recall"] == 0.0


def test_report_renders_without_labels():
    assert "No labels yet" in metrics.report([])


def test_report_warns_about_positives_below_the_cut():
    rows = [row(1, "top", 50, 30, 5), row(900, "tail", 1000, 20, 5)]
    text = metrics.report(rows)
    assert "below the default cut" in text
    assert "tail" in text


# --- comparing ranking engines -------------------------------------------


def test_positive_positions_reports_missing_as_none():
    order = ["a", "b", "c"]
    assert metrics.positive_positions(order, {"b", "z"}) == [("b", 1), ("z", None)]


def test_precision_over_labeled_ignores_unlabeled_papers():
    order = ["a", "u1", "b", "u2", "c"]          # u* were never labeled
    labels = {"a": 5, "b": 1, "c": 5}
    precision, hits, seen = metrics.precision_over_labeled(order, labels, k=5)
    assert (hits, seen) == (2, 3)
    assert precision == 2 / 3


def test_precision_is_zero_when_nothing_in_the_cut_was_labeled():
    assert metrics.precision_over_labeled(["u1", "u2"], {"a": 5}, k=2) == (0.0, 0, 0)


def test_compare_shows_each_engine_and_flags_sample_size():
    rankings = {
        "prefilter": ["x", "y", "pos1", "pos2"],
        "baseline": ["pos1", "pos2", "x", "y"],
    }
    labels = {"pos1": 5, "pos2": 5, "x": 1, "y": 1}
    text = metrics.compare_rankings(rankings, labels, {"pos1", "pos2"}, set(), ks=(2,))

    assert "prefilter" in text and "baseline" in text
    # The baseline put both positives first; the prefilter put them last.
    assert "not a measurement" in text
    lines = [l for l in text.splitlines() if "pos1" in l]
    assert lines and "2" in lines[0] and "0" in lines[0]


def test_compare_marks_papers_an_engine_never_ranked():
    rankings = {"agent": ["a"]}
    text = metrics.compare_rankings(rankings, {"a": 5, "b": 5}, {"a", "b"}, set(), ks=(1,))
    assert "--" in text
