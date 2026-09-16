"""Tests for the SQLite store.

Two properties matter more than the rest. Writes must be idempotent, because a
digest gets re-run and a labeling session gets resumed, and duplicated rows
would quietly corrupt every count downstream. And ratings must replace rather
than accumulate, because "I changed my mind about this paper" is a normal thing
to do and the store has to end up with one answer.
"""

from __future__ import annotations

import pytest

from arxiv_digest.models import Assessment
from arxiv_digest.store import Store
from conftest import make_paper


@pytest.fixture
def store():
    with Store(":memory:") as s:
        yield s


@pytest.fixture
def stocked(store):
    store.add_papers([make_paper(i, days_ago=i) for i in range(5)])
    return store


# --- papers ---------------------------------------------------------------


def test_papers_round_trip_with_types_intact(store):
    original = make_paper(1, days_ago=2, category="eess.IV")
    store.add_papers([original])

    loaded = store.get_paper(original.arxiv_id)
    assert loaded == original, "dataclass equality covers every field"
    assert loaded.authors == ["A. Author", "B. Author"]
    assert loaded.published == original.published


def test_adding_the_same_papers_twice_is_a_no_op(store):
    papers = [make_paper(i) for i in range(3)]
    assert store.add_papers(papers) == 3
    assert store.add_papers(papers) == 0
    assert store.count_papers() == 3


def test_first_seen_is_not_overwritten_on_re_add(store):
    paper = make_paper(1)
    store.add_papers([paper])
    first = store.conn.execute("SELECT first_seen FROM papers").fetchone()[0]

    store.add_papers([paper])
    assert store.conn.execute("SELECT first_seen FROM papers").fetchone()[0] == first


def test_missing_paper_is_none(store):
    assert store.get_paper("9999.99999") is None


def test_seen_ids_reports_only_what_is_stored(stocked):
    ids = [make_paper(i).arxiv_id for i in range(8)]
    seen = stocked.seen_ids(ids)
    assert len(seen) == 5
    assert stocked.seen_ids([]) == set()


def test_search_matches_title_and_abstract(stocked):
    assert stocked.search_papers("Paper 3")
    assert stocked.search_papers("Abstract for paper 4")
    assert stocked.search_papers("quantum chromodynamics") == []


def test_search_respects_limit(stocked):
    assert len(stocked.search_papers("Paper", limit=2)) == 2


# --- ratings --------------------------------------------------------------


def test_rating_a_paper_twice_replaces_it(stocked):
    arxiv_id = make_paper(0).arxiv_id
    stocked.set_rating(arxiv_id, 2)
    stocked.set_rating(arxiv_id, 5)

    assert stocked.get_rating(arxiv_id) == 5
    assert sum(stocked.rating_counts().values()) == 1, "one row, not two"


def test_rating_an_unknown_paper_is_refused(store):
    with pytest.raises(ValueError, match="not in the store"):
        store.set_rating("9999.99999", 5)


@pytest.mark.parametrize("given,expected", [(0, 1), (-2, 1), (9, 5), (3, 3)])
def test_ratings_are_clamped(stocked, given, expected):
    arxiv_id = make_paper(0).arxiv_id
    stocked.set_rating(arxiv_id, given)
    assert stocked.get_rating(arxiv_id) == expected


def test_rated_papers_filters_and_joins(stocked):
    stocked.set_rating(make_paper(0).arxiv_id, 5)
    stocked.set_rating(make_paper(1).arxiv_id, 1)

    high = stocked.rated_papers(min_rating=4)
    assert len(high) == 1
    assert high[0]["title"] == "Paper 0"
    assert high[0]["abs_url"].endswith(make_paper(0).arxiv_id)
    assert len(stocked.rated_papers(min_rating=1)) == 2


def test_unrated_paper_has_no_rating(stocked):
    assert stocked.get_rating(make_paper(0).arxiv_id) is None


# --- assessments ----------------------------------------------------------


def test_engines_do_not_overwrite_each_other(stocked):
    paper = make_paper(0)
    run = stocked.start_run("compare", ["cs.LG"], 7, 200, 40)
    for engine, score in [("baseline", 4), ("agent", 2)]:
        stocked.add_assessments(
            run, engine,
            [(paper, Assessment(arxiv_id=paper.arxiv_id, score=score, method="m", relevance="r"))],
        )

    found = {a["engine"]: a["score"] for a in stocked.assessments_for(paper.arxiv_id)}
    assert found == {"baseline": 4, "agent": 2}


def test_rerunning_the_same_engine_replaces_its_verdict(stocked):
    paper = make_paper(0)
    run = stocked.start_run("digest", ["cs.LG"], 7, 200, 40)
    for score in (2, 5):
        stocked.add_assessments(
            run, "baseline",
            [(paper, Assessment(arxiv_id=paper.arxiv_id, score=score, method="m", relevance="r"))],
        )
    assert len(stocked.assessments_for(paper.arxiv_id)) == 1
    assert stocked.assessments_for(paper.arxiv_id)[0]["score"] == 5


def test_rank_is_recorded_from_list_position(stocked):
    papers = [make_paper(i) for i in range(3)]
    run = stocked.start_run("digest", ["cs.LG"], 7, 200, 40)
    stocked.add_assessments(
        run, "baseline",
        [(p, Assessment(arxiv_id=p.arxiv_id, score=3, method="m", relevance="r")) for p in papers],
    )
    assert stocked.assessments_for(papers[2].arxiv_id)[0]["rank"] == 2


# --- profile --------------------------------------------------------------


def test_identical_profile_saves_do_not_create_versions(store):
    first = store.save_profile("Replay methods for medical imaging.")
    again = store.save_profile("Replay methods for medical imaging.")
    assert first == again
    assert len(store.profile_history()) == 1


def test_changed_profile_creates_a_new_version(store):
    store.save_profile("Version one.")
    second = store.save_profile("Version two.")
    assert second == 2
    assert store.current_profile()["content"] == "Version two."
    assert len(store.profile_history()) == 2, "history is what stage 4 measures drift against"


def test_no_profile_yet(store):
    assert store.current_profile() is None


# --- stats ----------------------------------------------------------------


def test_stats_summarise_everything(stocked):
    stocked.set_rating(make_paper(0).arxiv_id, 5)
    stocked.set_rating(make_paper(1).arxiv_id, 2)
    stocked.save_profile("A profile.")
    stocked.start_run("baseline", ["cs.LG"], 7, 200, 40)

    stats = stocked.stats()
    assert stats["papers"] == 5
    assert stats["rated"] == 2
    assert stats["positives"] == 1
    assert stats["runs"] == {"baseline": 1}
    assert stats["profile_versions"] == 1


def test_persists_across_reopen(tmp_path):
    path = tmp_path / "digest.db"
    with Store(path) as first:
        first.add_papers([make_paper(1)])
        first.set_rating(make_paper(1).arxiv_id, 4)

    with Store(path) as second:
        assert second.count_papers() == 1
        assert second.get_rating(make_paper(1).arxiv_id) == 4


# --- import guard ---------------------------------------------------------


def test_conversational_ratings_are_distinguishable_from_labels(stocked):
    """init_store relies on `source` to avoid reverting chat ratings."""
    a, b = make_paper(0).arxiv_id, make_paper(1).arxiv_id
    stocked.set_rating(a, 5, source="label")
    stocked.set_rating(b, 4, source="chat")

    from_chat = {
        r["arxiv_id"] for r in stocked.conn.execute(
            "SELECT arxiv_id FROM ratings WHERE source = 'chat'"
        ).fetchall()
    }
    assert from_chat == {b}


# --- embeddings -----------------------------------------------------------


def test_vectors_round_trip_and_are_normalised(stocked):
    import numpy as np

    raw = {make_paper(0).arxiv_id: np.array([3.0, 4.0], dtype=np.float32)}  # norm 5
    stocked.set_embeddings("m", raw)

    ids, matrix = stocked.get_embeddings("m")
    assert ids == [make_paper(0).arxiv_id]
    assert abs(float(np.linalg.norm(matrix[0])) - 1.0) < 1e-6
    assert abs(float(matrix[0][0]) - 0.6) < 1e-6


def test_models_are_kept_apart(stocked):
    """Vectors from different models are not comparable; mixing them is silent."""
    import numpy as np

    vector = {make_paper(0).arxiv_id: np.array([1.0, 0.0], dtype=np.float32)}
    stocked.set_embeddings("model-a", vector)
    stocked.set_embeddings("model-b", {make_paper(1).arxiv_id: np.array([0.0, 1.0], dtype=np.float32)})

    assert stocked.count_embeddings("model-a") == 1
    assert stocked.get_embeddings("model-a")[0] == [make_paper(0).arxiv_id]


def test_backfill_is_incremental(stocked):
    import numpy as np

    assert len(stocked.papers_missing_embeddings("m")) == 5
    stocked.set_embeddings("m", {make_paper(0).arxiv_id: np.array([1.0, 0.0], dtype=np.float32)})
    assert len(stocked.papers_missing_embeddings("m")) == 4


def test_semantic_search_orders_by_similarity(stocked):
    import numpy as np

    stocked.set_embeddings("m", {
        make_paper(0).arxiv_id: np.array([1.0, 0.0], dtype=np.float32),
        make_paper(1).arxiv_id: np.array([0.7, 0.7], dtype=np.float32),
        make_paper(2).arxiv_id: np.array([0.0, 1.0], dtype=np.float32),
    })
    hits = stocked.semantic_search(np.array([1.0, 0.0], dtype=np.float32), "m", limit=3)

    assert [p.arxiv_id for p, _ in hits] == [
        make_paper(0).arxiv_id, make_paper(1).arxiv_id, make_paper(2).arxiv_id
    ]
    assert hits[0][1] > hits[-1][1]


def test_semantic_search_without_vectors_is_empty(stocked):
    import numpy as np
    assert stocked.semantic_search(np.array([1.0, 0.0], dtype=np.float32), "m") == []
