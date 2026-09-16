"""Shared fixtures.

Everything here is synthetic. No test in this suite touches the network, so the
whole suite runs while arXiv is throttling, offline, or in CI.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from arxiv_digest.models import Paper

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def make_paper(index: int, days_ago: float = 0.0, category: str = "cs.LG") -> Paper:
    published = NOW - timedelta(days=days_ago)
    return Paper(
        arxiv_id=f"2609.{index:05d}",
        title=f"Paper {index}",
        abstract=f"Abstract for paper {index}.",
        authors=["A. Author", "B. Author"],
        primary_category=category,
        categories=[category],
        published=published,
        abs_url=f"https://arxiv.org/abs/2609.{index:05d}",
        pdf_url=f"https://arxiv.org/pdf/2609.{index:05d}",
    )


def atom_entry(paper: Paper) -> str:
    authors = "".join(f"<author><name>{a}</name></author>" for a in paper.authors)
    categories = "".join(f'<category term="{c}"/>' for c in paper.categories)
    return f"""
  <entry>
    <id>https://arxiv.org/abs/{paper.arxiv_id}v1</id>
    <published>{paper.published.strftime('%Y-%m-%dT%H:%M:%SZ')}</published>
    <title>{paper.title}</title>
    <summary>{paper.abstract}</summary>
    {authors}
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="{paper.primary_category}"/>
    {categories}
    <link href="https://arxiv.org/abs/{paper.arxiv_id}v1" rel="alternate"/>
    <link title="pdf" href="https://arxiv.org/pdf/{paper.arxiv_id}v1" rel="related"/>
  </entry>"""


def atom_feed(papers: list[Paper], total: int | None = None) -> bytes:
    total = len(papers) if total is None else total
    entries = "".join(atom_entry(p) for p in papers)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>{total}</opensearch:totalResults>{entries}
</feed>""".encode()


@pytest.fixture
def frozen_now(monkeypatch):
    """Pin datetime.now so window cutoffs are deterministic."""
    import arxiv_digest.arxiv_client as client

    class FrozenDatetime(client.datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(client, "datetime", FrozenDatetime)
    return NOW


@pytest.fixture
def no_sleep(monkeypatch):
    """Make backoff instant so retry tests do not take ten minutes."""
    import arxiv_digest.arxiv_client as client

    slept: list[float] = []
    monkeypatch.setattr(client.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(client, "_last_request_at", 0.0)
    return slept


@pytest.fixture
def profile_text() -> str:
    return """# Research profile

<!-- This comment must never be embedded. -->

## What I work on

Continual learning for medical image classification, specifically
replay-based methods under domain shift.

## Methods I track

- Replay and rehearsal: experience replay, generative replay
- Regularization-based continual learning: EWC, synaptic intelligence

## What I do not need

- Continual learning for language models with no vision component
- Neural architecture search, quantization, inference efficiency
"""
