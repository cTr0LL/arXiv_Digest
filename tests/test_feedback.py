"""Tests for the weekly feedback loop.

These cover the two things that turn a scheduled job from 52 newsletters into
something cumulative: ratings flowing back out of digests, and papers already
featured not reappearing.
"""

from __future__ import annotations

from datetime import date

import pytest

from arxiv_digest import feedback, render
from arxiv_digest.models import Assessment
from arxiv_digest.store import Store
from conftest import make_paper


@pytest.fixture
def store():
    with Store(":memory:") as s:
        s.add_papers([make_paper(i) for i in range(6)])
        yield s


def digest_with_ratings(tmp_path, ratings: dict[str, str], name="2026-09-16.md"):
    ranked = [
        (make_paper(i), Assessment(arxiv_id=make_paper(i).arxiv_id, score=3,
                                   method="m", relevance="r"))
        for i in range(3)
    ]
    md = render.render(ranked=ranked, top_k=3, scanned=10, categories=["cs.LG"],
                       window_days=7, run_date=date(2026, 9, 16))
    for arxiv_id, value in ratings.items():
        md = md.replace(f"**{arxiv_id}**: `?`", f"**{arxiv_id}**: `{value}`")
    (tmp_path / name).write_text(md, encoding="utf-8")
    return tmp_path


# --- ratings flow back ----------------------------------------------------


def test_ratings_are_read_out_of_digest_files(store, tmp_path):
    a, b = make_paper(0).arxiv_id, make_paper(1).arxiv_id
    digest_with_ratings(tmp_path, {a: "5", b: "2"})

    imported = feedback.ingest_digest_ratings(store, tmp_path)

    assert imported == {a: 5, b: 2}
    assert store.get_rating(a) == 5
    assert store.get_rating(b) == 2


def test_unrated_entries_are_not_imported(store, tmp_path):
    digest_with_ratings(tmp_path, {})
    assert feedback.ingest_digest_ratings(store, tmp_path) == {}


def test_reimporting_is_idempotent(store, tmp_path):
    a = make_paper(0).arxiv_id
    digest_with_ratings(tmp_path, {a: "4"})

    feedback.ingest_digest_ratings(store, tmp_path)
    feedback.ingest_digest_ratings(store, tmp_path)

    assert sum(store.rating_counts().values()) == 1, "one row, not two"


def test_conversational_ratings_are_not_reverted(store, tmp_path):
    """A rating made in chat is newer than any Markdown file."""
    a = make_paper(0).arxiv_id
    store.set_rating(a, 5, source="chat")
    digest_with_ratings(tmp_path, {a: "1"})

    feedback.ingest_digest_ratings(store, tmp_path)
    assert store.get_rating(a) == 5


def test_ratings_for_unknown_papers_do_not_crash_the_run(store, tmp_path):
    md = "## Your ratings\n\n- **9999.99999**: `5`\n"
    (tmp_path / "old.md").write_text(md, encoding="utf-8")
    assert feedback.ingest_digest_ratings(store, tmp_path) == {}


def test_missing_digest_directory_is_fine(store, tmp_path):
    assert feedback.ingest_digest_ratings(store, tmp_path / "nope") == {}


# --- suppression ----------------------------------------------------------


def featured(store, papers, top_k):
    run = store.start_run("baseline", ["cs.LG"], 7, 100, 40, top_k=top_k)
    store.add_assessments(run, "baseline", [
        (p, Assessment(arxiv_id=p.arxiv_id, score=3, method="m", relevance="r"))
        for p in papers
    ])


def test_recently_featured_papers_are_suppressed(store):
    papers = [make_paper(i) for i in range(5)]
    featured(store, papers, top_k=2)

    covered = feedback.recently_covered(store, days=21)
    assert {p.arxiv_id for p in papers[:2]} == covered


def test_papers_below_the_cut_are_not_suppressed(store):
    """Only what actually appeared in the digest; a candidate is not a feature."""
    papers = [make_paper(i) for i in range(5)]
    featured(store, papers, top_k=2)
    assert make_paper(4).arxiv_id not in feedback.recently_covered(store, days=21)


def test_suppression_can_be_switched_off(store):
    featured(store, [make_paper(0)], top_k=1)
    assert feedback.recently_covered(store, days=0) == set()


def test_nothing_is_suppressed_before_any_runs(store):
    assert feedback.recently_covered(store, days=21) == set()
