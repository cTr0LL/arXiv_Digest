"""Stage 1: the plain pipeline.

Fixed steps, no tools, no agency: hand the model a batch of abstracts and sort
by the score that comes back. This is the baseline every later stage gets
measured against, so once stage 2 exists, resist improving this file. A
baseline that keeps moving proves nothing.
"""

from __future__ import annotations

import os

import anthropic

from . import config
from .models import Assessment, Paper, Ranking

# Abstracts per API call. Keeping batches bounded stops a long paper list from
# running past max_tokens and truncating the JSON mid-array.
BATCH_SIZE = 40

SYSTEM_PROMPT = """You triage new arXiv submissions for one researcher.

Their interests, in their own words:

<profile>
{profile}
</profile>

You will get a numbered list of abstracts. Score every one of them from 1 to 5:

5 - directly on their problem; they should read it today
4 - clearly useful; same subfield, method or benchmark they work with
3 - adjacent; worth knowing about, not worth a full read
2 - same broad area, no real connection to their work
1 - unrelated

Be harsh. These abstracts have already passed a similarity filter, so they will
all look superficially plausible. Most still deserve a 1 or 2. Grade inflation
makes the digest useless, which is the only way this task can actually fail.

Write `relevance` for this researcher specifically, naming the part of their
profile it connects to. If a paper is not relevant, say so plainly rather than
inventing a connection. Write `method` as what the paper does, concretely --
not what it claims to achieve.

The abstract is all you get. Do not speculate about content beyond it."""


def load_profile() -> str:
    """Read profile.md, refusing to run on a template nobody filled in."""
    if not config.PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"No profile at {config.PROFILE_PATH}. Copy "
            f"{config.PROFILE_EXAMPLE_PATH.name} to profile.md and fill it in. "
            "profile.md is gitignored, so what you write stays local."
        )

    profile = config.PROFILE_PATH.read_text(encoding="utf-8").strip()

    # Headings and HTML comments are not content. A profile that is all
    # scaffolding produces a digest of uniform 1s, which looks like a model
    # failure and is not one -- so fail here, where the cause is obvious.
    from .prefilter import profile_chunks

    if len(profile_chunks(profile)) < 3:
        raise ValueError(
            f"{config.PROFILE_PATH} has almost no content, just headings or "
            "comments. Fill in a few sentences about what you work on. This "
            "file is the prompt; everything downstream depends on it."
        )
    return profile


def _format_batch(papers: list[Paper]) -> str:
    blocks = []
    for index, paper in enumerate(papers, start=1):
        blocks.append(
            f"[{index}] arxiv_id: {paper.arxiv_id}\n"
            f"categories: {', '.join(paper.categories) or paper.primary_category}\n"
            f"title: {paper.title}\n"
            f"abstract: {paper.abstract}"
        )
    return "\n\n".join(blocks)


def assess(
    papers: list[Paper],
    profile: str,
    client: anthropic.Anthropic | None = None,
) -> dict[str, Assessment]:
    """Score every paper. Returns a map of arxiv_id -> Assessment."""
    client = client or anthropic.Anthropic()
    system = SYSTEM_PROMPT.format(profile=profile)
    results: dict[str, Assessment] = {}

    for start in range(0, len(papers), BATCH_SIZE):
        batch = papers[start : start + BATCH_SIZE]
        print(f"  scoring {start + 1}-{start + len(batch)} of {len(papers)}...")

        response = client.messages.parse(
            model=config.MODEL,
            max_tokens=config.MAX_TOKENS,
            system=system,
            # Opus 5 runs adaptive thinking when `thinking` is omitted. `effort`
            # is the cost knob: "low" is usually enough for abstract triage,
            # "high" is the API default.
            output_config={"effort": "medium"},
            output_format=Ranking,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Score all {len(batch)} of these submissions.\n\n"
                        f"{_format_batch(batch)}"
                    ),
                }
            ],
        )

        for assessment in response.parsed_output.assessments:
            assessment.score = max(1, min(5, assessment.score))
            results[assessment.arxiv_id] = assessment

    # Models occasionally drop an item from a long list. Dropping it here means
    # a paper silently disappears from the digest, so ask again for just the
    # stragglers -- a much shorter list, which is usually enough.
    missing = [p for p in papers if p.arxiv_id not in results]
    if missing:
        print(f"  {len(missing)} paper(s) had no verdict; asking again for those...")
        retry = client.messages.parse(
            model=config.MODEL,
            max_tokens=config.MAX_TOKENS,
            system=system,
            output_config={"effort": "medium"},
            output_format=Ranking,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Score these {len(missing)} submissions. Return one "
                        f"assessment for every one of them.\n\n{_format_batch(missing)}"
                    ),
                }
            ],
        )
        for assessment in retry.parsed_output.assessments:
            assessment.score = max(1, min(5, assessment.score))
            results[assessment.arxiv_id] = assessment

        still_missing = [p.arxiv_id for p in missing if p.arxiv_id not in results]
        if still_missing:
            print(
                f"  warning: {len(still_missing)} paper(s) still unscored after a "
                f"retry: {', '.join(still_missing[:5])}. They will not appear in "
                "the digest."
            )
    return results


def run(papers: list[Paper], profile: str | None = None) -> list[tuple[Paper, Assessment]]:
    """Score and rank. Returns every paper, best first -- the caller cuts to top-K."""
    if not papers:
        return []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or run with --dry-run to test the retrieval half without it."
        )

    profile = profile if profile is not None else load_profile()
    scored = assess(papers, profile)

    ranked = [(p, scored[p.arxiv_id]) for p in papers if p.arxiv_id in scored]
    ranked.sort(key=lambda pair: (-pair[1].score, -pair[0].published.timestamp()))
    return ranked
