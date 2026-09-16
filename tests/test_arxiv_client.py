"""Tests for the arXiv client.

This module exists because `fetch_window` was rewritten during an arXiv
rate-limit block and could not be exercised against the real API. The paging
and cutoff logic is the part most likely to be subtly wrong -- an off-by-one in
the stop condition silently truncates the window, and a missing de-duplication
silently inflates it.
"""

from __future__ import annotations

import urllib.error

import pytest

from arxiv_digest import arxiv_client
from conftest import atom_feed, make_paper


def canned(monkeypatch, pages):
    """Serve `pages` (lists of Paper) in order, recording the params asked for."""
    calls = []

    def fake_get(params):
        calls.append(params)
        index = len(calls) - 1
        page = pages[index] if index < len(pages) else []
        return atom_feed(page)

    monkeypatch.setattr(arxiv_client, "_get", fake_get)
    return calls


def test_keeps_only_papers_inside_the_window(monkeypatch, frozen_now):
    page = [make_paper(1, 1), make_paper(2, 3), make_paper(3, 9), make_paper(4, 20)]
    canned(monkeypatch, [page])

    papers = arxiv_client.fetch_window(["cs.LG"], window_days=7)

    assert [p.arxiv_id for p in papers] == ["2609.00001", "2609.00002"]


def test_stops_paging_once_a_page_runs_past_the_cutoff(monkeypatch, frozen_now):
    fresh = [make_paper(i, 1) for i in range(5)]
    straddling = [make_paper(10, 6), make_paper(11, 30)]
    never_requested = [make_paper(99, 0)]
    calls = canned(monkeypatch, [fresh, straddling, never_requested])

    papers = arxiv_client.fetch_window(["cs.LG"], window_days=7, page_size=5)

    assert len(calls) == 2, "should stop after the page containing old papers"
    assert "2609.00099" not in {p.arxiv_id for p in papers}
    assert "2609.00010" in {p.arxiv_id for p in papers}


def test_deduplicates_ids_repeated_across_pages(monkeypatch, frozen_now):
    # arXiv pages can overlap at the boundary; the same id must not count twice.
    page_one = [make_paper(1, 1), make_paper(2, 1)]
    page_two = [make_paper(2, 1), make_paper(3, 1)]
    canned(monkeypatch, [page_one, page_two, []])

    papers = arxiv_client.fetch_window(["cs.LG"], window_days=7, page_size=2)

    ids = [p.arxiv_id for p in papers]
    assert len(ids) == len(set(ids))
    assert sorted(ids) == ["2609.00001", "2609.00002", "2609.00003"]


def test_respects_the_max_papers_ceiling(monkeypatch, frozen_now):
    pages = [[make_paper(i + offset, 1) for i in range(10)] for offset in (0, 100, 200)]
    calls = canned(monkeypatch, pages)

    papers = arxiv_client.fetch_window(
        ["cs.LG"], window_days=7, max_papers=20, page_size=10
    )

    assert len(papers) == 20
    assert sum(c["max_results"] for c in calls) <= 20


def test_stops_on_an_empty_page(monkeypatch, frozen_now):
    canned(monkeypatch, [[make_paper(1, 1)], []])
    papers = arxiv_client.fetch_window(["cs.LG"], window_days=7, page_size=5)
    assert len(papers) == 1


def test_query_joins_categories_with_or(monkeypatch, frozen_now):
    calls = canned(monkeypatch, [[]])
    arxiv_client.fetch_window(["cs.LG", "eess.IV"], window_days=7)
    assert calls[0]["search_query"] == "cat:cs.LG OR cat:eess.IV"


def test_parses_all_paper_fields(monkeypatch, frozen_now):
    canned(monkeypatch, [[make_paper(7, 1, category="eess.IV")], []])
    paper = arxiv_client.fetch_window(["eess.IV"], window_days=7)[0]

    assert paper.arxiv_id == "2609.00007"          # version suffix stripped
    assert paper.title == "Paper 7"
    assert paper.authors == ["A. Author", "B. Author"]
    assert paper.primary_category == "eess.IV"
    assert paper.abs_url == "https://arxiv.org/abs/2609.00007"
    assert paper.pdf_url.endswith("2609.00007v1")
    assert paper.byline() == "A. Author, B. Author"


# --- retry classification -------------------------------------------------
# 406 is how arXiv signals throttling. Treating it as a permanent error (which
# an earlier version of this client did) turns a wait into a hard failure.


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "reason", {}, None)


def raising(monkeypatch, code):
    attempts = []

    def fake_urlopen(request, timeout=None):
        attempts.append(code)
        raise http_error(code)

    monkeypatch.setattr(arxiv_client.urllib.request, "urlopen", fake_urlopen)
    return attempts


@pytest.mark.parametrize("code", [406, 429, 500, 503])
def test_retryable_statuses_are_retried(monkeypatch, no_sleep, code):
    attempts = raising(monkeypatch, code)
    with pytest.raises(urllib.error.HTTPError):
        arxiv_client._get({"search_query": "cat:cs.LG"})
    assert len(attempts) == arxiv_client.MAX_RETRIES


@pytest.mark.parametrize("code", [400, 404])
def test_permanent_statuses_fail_immediately(monkeypatch, no_sleep, code):
    attempts = raising(monkeypatch, code)
    with pytest.raises(RuntimeError, match="not a transient failure"):
        arxiv_client._get({"search_query": "cat:cs.LG"})
    assert len(attempts) == 1


def test_backoff_grows_and_is_minute_scale(monkeypatch, no_sleep):
    raising(monkeypatch, 406)
    with pytest.raises(urllib.error.HTTPError):
        arxiv_client._get({"search_query": "cat:cs.LG"})

    backoffs = [s for s in no_sleep if s >= 30]
    assert backoffs == sorted(backoffs), "backoff must not shrink"
    assert backoffs[0] >= 30, "retrying a throttle in seconds is what caused the block"


def test_partial_results_survive_a_failing_page(monkeypatch, frozen_now, no_sleep):
    """Throttling is per-request; page 3 failing must not discard pages 1-2."""
    good = [make_paper(i, 1) for i in range(5)]
    calls = {"n": 0}

    def flaky_get(params):
        calls["n"] += 1
        if calls["n"] == 1:
            return atom_feed(good)
        raise urllib.error.HTTPError("http://x", 406, "Not Acceptable", {}, None)

    monkeypatch.setattr(arxiv_client, "_get", flaky_get)
    papers = arxiv_client.fetch_window(["cs.LG"], window_days=7, page_size=5)

    assert len(papers) == 5, "first page should be kept"


def test_failure_on_the_very_first_page_still_raises(monkeypatch, frozen_now, no_sleep):
    """Nothing fetched means nothing to salvage -- do not fail silently."""
    def always_fails(params):
        raise urllib.error.HTTPError("http://x", 406, "Not Acceptable", {}, None)

    monkeypatch.setattr(arxiv_client, "_get", always_fails)
    with pytest.raises(urllib.error.HTTPError):
        arxiv_client.fetch_window(["cs.LG"], window_days=7)


def test_partial_fetch_does_not_blame_the_ceiling(monkeypatch, frozen_now, no_sleep, capsys):
    """A truncated fetch explains itself; it must not also claim a ceiling hit."""
    calls = {"n": 0}

    def flaky_get(params):
        calls["n"] += 1
        if calls["n"] == 1:
            return atom_feed([make_paper(i, 1) for i in range(5)])
        raise urllib.error.HTTPError("http://x", 406, "Not Acceptable", {}, None)

    monkeypatch.setattr(arxiv_client, "_get", flaky_get)
    arxiv_client.fetch_window(["cs.LG"], window_days=7, max_papers=100, page_size=5)

    out = capsys.readouterr().out
    assert "failed after retries" in out
    assert "ceiling" not in out


# --- id parsing -----------------------------------------------------------
# People paste whatever their browser gave them. All of these must work.


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2609.17068", "2609.17068"),
        ("2609.17068v2", "2609.17068"),
        ("https://arxiv.org/abs/2609.17068", "2609.17068"),
        ("http://arxiv.org/abs/2609.17068v1", "2609.17068"),
        ("https://arxiv.org/pdf/2609.17068", "2609.17068"),
        ("https://arxiv.org/pdf/2609.17068v3", "2609.17068"),
        ("https://arxiv.org/abs/2609.17068/", "2609.17068"),
        ("arxiv.org/abs/1706.03762", "1706.03762"),
        ("  https://arxiv.org/abs/2301.1234  ", "2301.1234"),
        ("cs.LG/9901001", "cs.LG/9901001"),
    ],
)
def test_arxiv_ids_are_parsed_from_every_common_form(text, expected):
    assert arxiv_client.parse_arxiv_id(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "not a paper", "https://example.com/page"])
def test_non_ids_are_rejected(text):
    assert arxiv_client.parse_arxiv_id(text) is None


def test_fetch_by_ids_batches_into_one_request(monkeypatch, frozen_now):
    calls = []

    def fake_get(params):
        calls.append(params)
        return atom_feed([make_paper(i, 1) for i in range(3)])

    monkeypatch.setattr(arxiv_client, "_get", fake_get)
    papers = arxiv_client.fetch_by_ids(["a", "b", "c"])

    assert len(calls) == 1, "ten papers must not mean ten requests"
    assert calls[0]["id_list"] == "a,b,c"
    assert len(papers) == 3


def test_fetch_by_ids_splits_oversized_batches(monkeypatch, frozen_now):
    calls = []
    monkeypatch.setattr(arxiv_client, "_get",
                        lambda p: calls.append(p) or atom_feed([]))
    arxiv_client.fetch_by_ids([str(i) for i in range(120)], batch_size=50)
    assert [len(c["id_list"].split(",")) for c in calls] == [50, 50, 20]


def test_fetch_by_ids_ignores_the_date_window(monkeypatch, frozen_now):
    """A paper you read last year must still come back."""
    monkeypatch.setattr(arxiv_client, "_get",
                        lambda p: atom_feed([make_paper(1, days_ago=900)]))
    assert len(arxiv_client.fetch_by_ids(["old"])) == 1
