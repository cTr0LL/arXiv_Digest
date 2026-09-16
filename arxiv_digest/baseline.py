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

Be harsh. In a typical week of ~60 submissions, one or two earn a 5 and most
land at 1 or 2. Grade inflation makes the digest useless, which is the only
way this task can actually fail.

Write `relevance` for this researcher specifically, naming the part of their
profile it connects to. If a paper is not relevant, say so plainly rather than
inventing a connection. Write `method` as what the paper does, concretely --
not what it claims to achieve.

The abstract is all you get. Do not speculate about content beyond it."""


def load_profile() -> str:
    if not config.PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"No profile at {config.PROFILE_PATH}. Copy profile.md from the repo "
            "and describe what you work on -- the digest is only as good as this file."
        )
    profile = config.PROFILE_PATH.read_text(encoding="utf-8").strip()
    if not profile:
        raise ValueError(f"{config.PROFILE_PATH} is empty.")
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

    missing = [p.arxiv_id for p in papers if p.arxiv_id not in results]
    if missing:
        print(f"  warning: model skipped {len(missing)} paper(s): {', '.join(missing[:5])}")
    return results


def run(papers: list[Paper]) -> list[tuple[Paper, Assessment]]:
    """Score and rank. Returns every paper, best first -- the caller cuts to top-K."""
    if not papers:
        return []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or run with --dry-run to test the arXiv half without it."
        )

    profile = load_profile()
    scored = assess(papers, profile)

    ranked = [(p, scored[p.arxiv_id]) for p in papers if p.arxiv_id in scored]
    ranked.sort(key=lambda pair: (-pair[1].score, -pair[0].published.timestamp()))
    return ranked
