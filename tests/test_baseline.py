"""Tests for stage 1 scoring.

The retry path is the point. A model that omits one item from a list of forty
causes a paper to vanish from the digest with only a warning -- the kind of
failure you notice weeks later, if ever.
"""

from __future__ import annotations

import pytest

from arxiv_digest import baseline
from arxiv_digest.models import Assessment, Ranking
from conftest import make_paper


class FakeMessages:
    def __init__(self, script, calls):
        self._script = list(script)
        self._index = 0
        self.calls = calls

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        result = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        return result


class Parsed:
    def __init__(self, assessments):
        self.parsed_output = Ranking(assessments=assessments)


class FakeClient:
    def __init__(self, script):
        self.calls = []
        self.messages = FakeMessages(script, self.calls)


def assessment(arxiv_id, score=3):
    return Assessment(arxiv_id=arxiv_id, score=score, method="m", relevance="r")


PAPERS = [make_paper(i) for i in range(4)]
IDS = [p.arxiv_id for p in PAPERS]


def test_all_papers_scored_in_one_pass():
    client = FakeClient([Parsed([assessment(i) for i in IDS])])
    results = baseline.assess(PAPERS, "profile", client=client)
    assert set(results) == set(IDS)
    assert len(client.calls) == 1, "no retry needed when nothing was dropped"


def test_a_dropped_paper_is_asked_for_again():
    dropped = IDS[2]
    client = FakeClient([
        Parsed([assessment(i) for i in IDS if i != dropped]),
        Parsed([assessment(dropped, 5)]),
    ])
    results = baseline.assess(PAPERS, "profile", client=client)

    assert set(results) == set(IDS), "the dropped paper must come back"
    assert results[dropped].score == 5
    assert len(client.calls) == 2


def test_the_retry_only_asks_about_the_stragglers():
    dropped = IDS[1]
    client = FakeClient([
        Parsed([assessment(i) for i in IDS if i != dropped]),
        Parsed([assessment(dropped)]),
    ])
    baseline.assess(PAPERS, "profile", client=client)

    retry_prompt = client.calls[1]["messages"][0]["content"]
    assert dropped in retry_prompt
    assert sum(i in retry_prompt for i in IDS) == 1


def test_persistent_omission_warns_and_does_not_loop(capsys):
    dropped = IDS[3]
    partial = [assessment(i) for i in IDS if i != dropped]
    client = FakeClient([Parsed(partial), Parsed(partial)])

    results = baseline.assess(PAPERS, "profile", client=client)

    assert dropped not in results
    assert len(client.calls) == 2, "one retry, not an infinite loop"
    assert "still unscored" in capsys.readouterr().out


@pytest.mark.parametrize("given,expected", [(0, 1), (7, 5), (4, 4)])
def test_scores_are_clamped_on_both_paths(given, expected):
    client = FakeClient([Parsed([assessment(i, given) for i in IDS])])
    results = baseline.assess(PAPERS, "profile", client=client)
    assert results[IDS[0]].score == expected


def test_batches_are_bounded():
    many = [make_paper(i) for i in range(baseline.BATCH_SIZE + 5)]
    ids = [p.arxiv_id for p in many]
    client = FakeClient([Parsed([assessment(i) for i in ids])])
    baseline.assess(many, "profile", client=client)
    assert len(client.calls) >= 2, "a long list must be split across calls"
