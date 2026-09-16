"""Turn ranked papers into a Markdown digest you can read and rate.

The `Your rating:` field matters more than it looks. It is the only place your
feedback enters the system, and stage 3 reads it back out of these files with
`parse_ratings` below. Keep the format stable.
"""

from __future__ import annotations

import re
from datetime import date

from .models import Assessment, Paper

RATING_LINE = re.compile(
    r"^\s*-\s*\*\*(?P<arxiv_id>[\w.\-/]+)\*\*\s*:\s*`(?P<rating>[^`]*)`", re.MULTILINE
)


def render(
    ranked: list[tuple[Paper, Assessment]],
    top_k: int,
    scanned: int,
    categories: list[str],
    window_days: int,
    prefiltered: int | None = None,
    run_date: date | None = None,
) -> str:
    run_date = run_date or date.today()
    featured = ranked[:top_k]
    remainder = ranked[top_k:]

    out = [
        f"# arXiv digest - {run_date.isoformat()}",
        "",
        f"**{scanned}** submissions in {', '.join(categories)} over the last "
        f"{window_days} days."
        + (
            f" Embedding prefilter kept the top **{prefiltered}**; those were "
            "scored by the model."
            if prefiltered is not None
            else " All of them were scored by the model."
        ),
        "",
        "Ranked by a single pass over abstracts, no tool use (stage 1 baseline).",
        "",
    ]

    for position, (paper, assessment) in enumerate(featured, start=1):
        out += [
            "---",
            "",
            f"## {position}. {paper.title}",
            "",
            f"[{paper.arxiv_id}]({paper.abs_url}) | [pdf]({paper.pdf_url}) | "
            f"{paper.primary_category} | submitted {paper.published.date().isoformat()}",
            "",
            f"*{paper.byline()}*",
            "",
            f"**Method.** {assessment.method}",
            "",
            f"**Why you might care.** {assessment.relevance}",
            "",
            f"**Model score:** {assessment.score}/5",
            "",
        ]

    if remainder:
        out += [
            "---",
            "",
            "## Below the cut",
            "",
            "Scanned and ranked lower. Skim for false negatives - a good paper "
            "sitting down here is the most useful bug report you can give this thing.",
            "",
        ]
        for paper, assessment in remainder:
            out.append(
                f"- `{assessment.score}` [{paper.title}]({paper.abs_url}) "
                f"({paper.primary_category})"
            )
        out.append("")

    out += [
        "---",
        "",
        "## Your ratings",
        "",
        "Replace each `?` with 1-5, or `skip` if you did not look. "
        "Anything left as `?` is treated as unrated.",
        "",
    ]
    for paper, _ in featured:
        out.append(f"- **{paper.arxiv_id}**: `?`  <!-- {paper.title[:70]} -->")
    out.append("")

    return "\n".join(out)


def parse_ratings(markdown: str) -> dict[str, int]:
    """Read the ratings block back out of a digest file.

    Unrated entries (`?`) and `skip` are left out of the result rather than
    recorded as zero -- "I did not look" is not the same as "this was bad",
    and collapsing them would poison the eval later.
    """
    ratings: dict[str, int] = {}
    for match in RATING_LINE.finditer(markdown):
        value = match.group("rating").strip().lower()
        if value.isdigit() and 1 <= int(value) <= 5:
            ratings[match.group("arxiv_id")] = int(value)
    return ratings
