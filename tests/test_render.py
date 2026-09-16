"""Tests for digest rendering and reading ratings back.

The ratings block is the only channel your feedback has, so its format is a
contract. These tests are what stop a cosmetic edit to the digest from silently
breaking label collection weeks later.
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from arxiv_digest import render
from arxiv_digest.models import Assessment
from conftest import make_paper


def ranked(n=5):
    return [
        (make_paper(i), Assessment(arxiv_id=f"2609.{i:05d}", score=5 - (i % 5),
                                   method=f"Method {i}.", relevance=f"Relevance {i}."))
        for i in range(n)
    ]


def fill(markdown, arxiv_id, value):
    """Edit one rating line the way a human would, leaving the rest alone."""
    return re.sub(rf"(\*\*{re.escape(arxiv_id)}\*\*: )`\?`", rf"\g<1>`{value}`", markdown)


def test_unrated_digest_yields_no_labels():
    md = render.render(ranked(), top_k=3, scanned=100, categories=["cs.LG"], window_days=7)
    assert render.parse_ratings(md) == {}


def test_ratings_round_trip():
    md = render.render(ranked(), top_k=3, scanned=100, categories=["cs.LG"], window_days=7)
    md = fill(md, "2609.00000", "5")
    md = fill(md, "2609.00002", "2")
    assert render.parse_ratings(md) == {"2609.00000": 5, "2609.00002": 2}


def test_skip_is_excluded_not_scored_zero():
    """'I did not read it' must not be recorded as 'this was bad'."""
    md = render.render(ranked(), top_k=3, scanned=100, categories=["cs.LG"], window_days=7)
    md = fill(md, "2609.00001", "skip")
    assert "2609.00001" not in render.parse_ratings(md)


@pytest.mark.parametrize("bad", ["0", "6", "yes", "-1", ""])
def test_out_of_range_values_are_ignored(bad):
    md = render.render(ranked(), top_k=2, scanned=10, categories=["cs.LG"], window_days=7)
    md = fill(md, "2609.00000", bad)
    assert "2609.00000" not in render.parse_ratings(md)


def test_instruction_line_is_not_mistaken_for_a_rating():
    """The instructions contain a literal `?`; it must not parse as an entry."""
    md = render.render(ranked(), top_k=2, scanned=10, categories=["cs.LG"], window_days=7)
    assert "Replace each" in md
    assert render.parse_ratings(md.replace("`?`", "`4`", 1)) == {}


def test_only_featured_papers_get_rating_lines():
    md = render.render(ranked(5), top_k=2, scanned=10, categories=["cs.LG"], window_days=7)
    lines = [l for l in md.splitlines() if l.startswith("- **")]
    assert len(lines) == 2


def test_papers_below_the_cut_are_still_listed():
    md = render.render(ranked(5), top_k=2, scanned=10, categories=["cs.LG"], window_days=7)
    assert "Below the cut" in md
    assert "Paper 4" in md


def test_header_reports_the_funnel():
    with_filter = render.render(ranked(), top_k=2, scanned=1394, categories=["cs.LG"],
                                window_days=7, prefiltered=150)
    assert "1394" in with_filter and "150" in with_filter

    without = render.render(ranked(), top_k=2, scanned=300, categories=["cs.LG"], window_days=7)
    assert "All of them were scored" in without


def test_output_is_ascii_only():
    """Non-ASCII separators render as mojibake in a Windows console."""
    md = render.render(ranked(), top_k=3, scanned=100, categories=["cs.LG"],
                       window_days=7, prefiltered=50, run_date=date(2026, 9, 16))
    assert all(ord(c) < 128 for c in md)


def test_empty_ranking_still_renders():
    md = render.render([], top_k=8, scanned=0, categories=["cs.LG"], window_days=7)
    assert "arXiv digest" in md
    assert render.parse_ratings(md) == {}
