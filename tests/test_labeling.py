"""Tests for stratified sampling and the label store.

The sampling design is the part of stage 0 that decides whether recall is
measurable at all, so the properties worth pinning are: the tail actually gets
sampled, the order reveals nothing about rank, and resuming never re-asks a
question you already answered.
"""

from __future__ import annotations

from collections import Counter

from arxiv_digest import labeling
from conftest import make_paper


def ranking(n=1400):
    return [(make_paper(i), 1.0 - i / n) for i in range(n)]


def test_samples_every_stratum_including_the_tail():
    tasks = labeling.build_sample(ranking(), seed=1)
    counts = Counter(t.stratum for t in tasks)
    assert set(counts) == {"top", "near", "mid", "tail"}
    assert sum(counts.values()) == 80
    assert counts["top"] > counts["tail"], "the top must be sampled more densely"


def test_tail_is_sampled_sparsely_but_represented():
    tasks = labeling.build_sample(ranking(), seed=1, keep=150)
    tail = [t for t in tasks if t.stratum == "tail"]
    assert tail, "a miss deep in the ranking is invisible without a tail sample"
    assert all(t.rank >= 150 for t in tail)
    rate = tail[0].stratum_sampled / tail[0].stratum_size
    assert rate < 0.05, "the tail is meant to be cheap, not thorough"


def test_strata_straddle_the_cut_at_every_pool_size():
    """A 200-paper pool with a cut at 40 must still measure both sides."""
    for pool, keep in [(1400, 150), (600, 150), (400, 80), (200, 40), (120, 25)]:
        strata = labeling.strata_for(pool, keep)
        above = [s for s in strata if s[1] < keep]
        below = [s for s in strata if s[1] >= keep]
        assert above and below, f"pool={pool} keep={keep} has no labels on one side"
        assert all(end <= pool for _, _, end, _ in strata)


def test_a_cut_past_the_pool_yields_nothing_below_it():
    """The degenerate case label.py warns about: recall would be 100% for free."""
    strata = labeling.strata_for(pool_size=100, keep=150)
    assert not [s for s in strata if s[1] >= 150]


def test_sample_never_exceeds_the_label_budget():
    tasks = labeling.build_sample(ranking(), seed=1, total=40)
    assert len(tasks) <= 40


def test_order_does_not_leak_rank():
    """If the sample were ordered by rank, the labeler would anchor on it."""
    ranks = [t.rank for t in labeling.build_sample(ranking(), seed=1)]
    assert ranks != sorted(ranks)
    assert ranks != sorted(ranks, reverse=True)


def test_sampling_is_reproducible_across_runs():
    a = [t.paper.arxiv_id for t in labeling.build_sample(ranking(), seed=5)]
    b = [t.paper.arxiv_id for t in labeling.build_sample(ranking(), seed=5)]
    assert a == b


def test_already_labeled_papers_are_not_offered_again():
    first = labeling.build_sample(ranking(), seed=1)
    done = {t.paper.arxiv_id for t in first[:10]}
    second = labeling.build_sample(ranking(), seed=1, exclude=done)
    assert done.isdisjoint({t.paper.arxiv_id for t in second})


def test_small_pool_still_gets_usable_strata():
    """A 60-paper pool must not crash, and must still straddle a scaled cut."""
    tasks = labeling.build_sample(ranking(60), seed=1, keep=15)
    strata = {t.stratum for t in tasks}
    assert "top" in strata
    assert any(t.rank >= 15 for t in tasks), "need labels below the cut"
    assert all(t.rank < 60 for t in tasks)


def test_labels_written_immediately_with_full_schema(tmp_path):
    path = tmp_path / "labels.csv"
    task = labeling.build_sample(ranking(), seed=1)[0]

    labeling.append_label(path, task, "y")

    rows = labeling.load_labels(path)
    assert len(rows) == 1
    assert list(rows[0].keys()) == labeling.FIELDNAMES
    assert rows[0]["label"] == "y"
    assert int(rows[0]["score"]) == labeling.LABEL_VALUES["y"]
    assert rows[0]["arxiv_id"] == task.paper.arxiv_id


def test_appending_preserves_earlier_rows(tmp_path):
    path = tmp_path / "labels.csv"
    tasks = labeling.build_sample(ranking(), seed=1)[:3]
    for task, key in zip(tasks, ["y", "n", "m"]):
        labeling.append_label(path, task, key)

    rows = labeling.load_labels(path)
    assert [r["label"] for r in rows] == ["y", "n", "m"]
    assert labeling.labeled_ids(path) == {t.paper.arxiv_id for t in tasks}


def test_missing_label_file_is_not_an_error(tmp_path):
    assert labeling.load_labels(tmp_path / "nope.csv") == []
    assert labeling.labeled_ids(tmp_path / "nope.csv") == set()


def test_pool_cache_round_trips(tmp_path):
    path = tmp_path / "pool.json"
    papers = [make_paper(i, days_ago=i) for i in range(5)]
    labeling.save_pool(path, papers, {"fetched_at": "now"})

    loaded, meta = labeling.load_pool(path)
    assert meta["fetched_at"] == "now"
    assert [p.arxiv_id for p in loaded] == [p.arxiv_id for p in papers]
    assert loaded[0].published == papers[0].published
    assert loaded[0] == papers[0]


def test_missing_pool_returns_none(tmp_path):
    assert labeling.load_pool(tmp_path / "nope.json") is None
