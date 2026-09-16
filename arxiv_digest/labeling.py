"""Stage 0: build a labeled set by hand.

The sampling is the whole design. Two obvious approaches both fail:

- Sample uniformly from the week. At ~1400 submissions and a few percent
  relevance, 100 labels buy you three positives. Nothing can be computed from
  that.
- Sample from the prefilter's top 150. Now every label agrees with the
  prefilter by construction, and its recall -- the number we actually cannot
  see -- stays invisible.

So: sample densely near the top of the ranking and sparsely down the tail,
record which stratum each paper came from, and weight by the inverse sampling
rate when estimating anything about the full pool. That is the standard fix,
and it is what makes `recall@150` a real number instead of a vibe.

Two details that matter as much as the maths. The sample is shuffled and the
prefilter score is hidden while you label, so you cannot anchor on the ranking
you are trying to evaluate. And labels are stored with the rank and stratum
they came from, so the weighting can be recomputed later without re-labeling.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .models import Paper

# Share of the label budget each stratum gets. Front-loaded because positives
# are concentrated there and precision needs them; the thin tail share is still
# enough to detect a miss, because inverse-propensity weighting makes each tail
# label stand in for many papers.
STRATUM_SHARES = [("top", 0.35), ("near", 0.30), ("mid", 0.22), ("tail", 0.13)]


def strata_for(
    pool_size: int, keep: int, total: int = 80
) -> list[tuple[str, int, int, int]]:
    """Stratum boundaries as (name, start, end, sample_size).

    Boundaries are defined relative to `keep` -- the rank where the prefilter
    actually cuts -- rather than as fixed ranks. Two strata sit above the cut
    and two below, so precision and recall each get evidence no matter how big
    the pool is.

    Fixed boundaries break on small pools: with 200 papers and a cut at 150,
    every stratum lands above the line, recall comes out 100% by construction,
    and the number means nothing.
    """
    keep = max(1, min(keep, pool_size))
    bounds = [
        ("top", 0, max(5, round(keep / 3))),
        ("near", max(5, round(keep / 3)), keep),
        ("mid", keep, min(pool_size, round(keep * 2.5))),
        ("tail", min(pool_size, round(keep * 2.5)), pool_size),
    ]

    out = []
    for (name, start, end), (_, share) in zip(bounds, STRATUM_SHARES):
        size = max(0, min(end, pool_size) - start)
        if size <= 0:
            continue
        out.append((name, start, min(end, pool_size), min(size, max(1, round(total * share)))))
    return out


# What a full week (~1400 papers) with the default cut looks like.
DEFAULT_STRATA: list[tuple[str, int, int, int]] = strata_for(1400, 150)

# What the single-keystroke answers mean. Graded rather than binary: "adjacent"
# is a real category and collapsing it into "no" throws away signal.
LABEL_VALUES = {"y": 5, "m": 3, "n": 1}
RELEVANT_AT = 4  # score >= this counts as a positive in the metrics

FIELDNAMES = [
    "arxiv_id",
    "label",
    "score",
    "rank",
    "stratum",
    "stratum_size",
    "stratum_sampled",
    "prefilter_score",
    "title",
    "labeled_at",
]


@dataclass
class LabelTask:
    paper: Paper
    rank: int
    stratum: str
    stratum_size: int
    stratum_sampled: int
    prefilter_score: float


def build_sample(
    ranked: list[tuple[Paper, float]],
    strata: list[tuple[str, int, int, int]] | None = None,
    seed: int = 0,
    exclude: set[str] | None = None,
    keep: int = 150,
    total: int = 80,
) -> list[LabelTask]:
    """Draw a stratified sample from a full prefilter ranking.

    Boundaries scale to the pool and to `keep`, so a 200-paper pool still puts
    strata on both sides of the cut.
    """
    strata = strata or strata_for(len(ranked), keep, total)
    exclude = exclude or set()
    rng = random.Random(seed)

    tasks: list[LabelTask] = []
    for name, start, end, want in strata:
        window = [
            (rank, paper, score)
            for rank, (paper, score) in enumerate(ranked)
            if start <= rank < end and paper.arxiv_id not in exclude
        ]
        if not window:
            continue
        chosen = rng.sample(window, min(want, len(window)))
        for rank, paper, score in chosen:
            tasks.append(
                LabelTask(
                    paper=paper,
                    rank=rank,
                    stratum=name,
                    stratum_size=len(window),
                    stratum_sampled=len(chosen),
                    prefilter_score=score,
                )
            )

    # Shuffle so the labeler cannot infer the stratum from the order. Without
    # this, the first thirty papers are visibly better and the rest feel worse
    # by contrast, which biases exactly the comparison we are trying to make.
    rng.shuffle(tasks)
    return tasks


def load_labels(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def labeled_ids(path: Path) -> set[str]:
    return {row["arxiv_id"] for row in load_labels(path)}


def append_label(path: Path, task: LabelTask, key: str) -> None:
    """Append one label. Written immediately so a quit never loses work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "arxiv_id": task.paper.arxiv_id,
                "label": key,
                "score": LABEL_VALUES[key],
                "rank": task.rank,
                "stratum": task.stratum,
                "stratum_size": task.stratum_size,
                "stratum_sampled": task.stratum_sampled,
                "prefilter_score": f"{task.prefilter_score:.4f}",
                "title": task.paper.title,
                "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )


def paper_to_dict(paper: Paper) -> dict:
    data = paper.__dict__.copy()
    data["published"] = paper.published.isoformat()
    return data


def paper_from_dict(data: dict) -> Paper:
    data = dict(data)
    data["published"] = datetime.fromisoformat(data["published"])
    return Paper(**data)


def save_pool(path: Path, papers: list[Paper], meta: dict) -> None:
    """Cache the fetched window.

    Re-fetching 1400 abstracts on every resume is slow and earns 429s from
    arXiv, and a pool that changes between sessions would mean labels from
    different sessions refer to different rankings.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "papers": [paper_to_dict(p) for p in papers]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_pool(path: Path) -> tuple[list[Paper], dict] | None:
    import json

    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [paper_from_dict(d) for d in payload["papers"]], payload["meta"]
