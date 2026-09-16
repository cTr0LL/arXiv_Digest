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

# (name, start_rank, end_rank_exclusive, how_many_to_sample). Ranks are
# 0-indexed positions in the prefilter ranking.
DEFAULT_STRATA: list[tuple[str, int, int, int]] = [
    ("top", 0, 50, 30),
    ("near", 50, 150, 25),
    ("mid", 150, 400, 25),
    ("tail", 400, 10_000, 20),
]

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
) -> list[LabelTask]:
    """Draw a stratified sample from a full prefilter ranking."""
    strata = strata or DEFAULT_STRATA
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
