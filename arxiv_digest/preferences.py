"""Stage 4: scoring that learns from your ratings.

The scoring rule extends the prefilter rather than replacing it:

    score = max sim(paper, wanted)        # what you said you want
          - max sim(paper, unwanted)      # what you said you do not
          + alpha * max sim(paper, liked) # what you actually rated highly
          - beta  * max sim(paper, disliked)

At `alpha = beta = 0` this is exactly the stage 1.5 prefilter, which is the
point: the baseline is a special case, so "does learning from ratings help?"
becomes the concrete question "is the best alpha non-zero?" -- answerable by a
sweep rather than by argument.

Why nearest-neighbour rather than a trained classifier
------------------------------------------------------
At the time of writing there are 55 ratings and a handful of positives, over
384-dimensional embeddings. Logistic regression on that would memorise the
training set and report a wonderful number. A max-similarity rule has no
parameters to overfit beyond the two weights, degrades gracefully when there
are two positives, and improves smoothly as ratings accumulate -- which is the
actual operating condition.

Note where the signal is. With 48 papers rated 1 and 2 rated 5, the negatives
are the richer half of the data: they say "medical imaging alone is not enough",
which is precisely the intersection problem the profile alone fails to express.

The self-match trap
-------------------
A rated paper is in its own liked pool, so its similarity to itself is 1.0 and
it scores brilliantly. Any evaluation that forgets this reports near-perfect
results. `leave_one_out` exists to prevent exactly that, and there is a test
asserting the naive version really is inflated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Ratings at or above this are "liked"; at or below the other, "disliked".
LIKED_AT = 4
DISLIKED_AT = 2


@dataclass
class Pools:
    """Everything the scorer compares a paper against.

    Ids travel alongside the vectors so `leave_one_out` can drop a paper from
    its own pool. Storing them separately is what makes that check possible.
    """

    wanted: "np.ndarray"
    unwanted: "np.ndarray"
    liked: "np.ndarray"
    liked_ids: list[str] = field(default_factory=list)
    disliked: "np.ndarray" = None
    disliked_ids: list[str] = field(default_factory=list)


def _max_sim(docs: "np.ndarray", pool: "np.ndarray") -> "np.ndarray":
    """Best similarity from each doc to anything in the pool; 0 if empty."""
    import numpy as np

    if pool is None or len(pool) == 0:
        return np.zeros(len(docs), dtype=np.float32)
    return (docs @ pool.T).max(axis=1)


def score(
    docs: "np.ndarray",
    pools: Pools,
    alpha: float = 0.0,
    beta: float = 0.0,
    exclude_ids: set[str] | None = None,
) -> "np.ndarray":
    """Score documents. `alpha = beta = 0` reproduces the plain prefilter.

    `exclude_ids` drops those papers from the liked and disliked pools, which
    is how leave-one-out avoids scoring a paper against itself.
    """
    import numpy as np

    scores = _max_sim(docs, pools.wanted) - _max_sim(docs, pools.unwanted)

    liked, disliked = pools.liked, pools.disliked
    if exclude_ids:
        liked = _drop(pools.liked, pools.liked_ids, exclude_ids)
        disliked = _drop(pools.disliked, pools.disliked_ids, exclude_ids)

    if alpha:
        scores = scores + alpha * _max_sim(docs, liked)
    if beta:
        scores = scores - beta * _max_sim(docs, disliked)
    return scores


def _drop(matrix: "np.ndarray", ids: list[str], exclude: set[str]) -> "np.ndarray":
    import numpy as np

    if matrix is None or len(matrix) == 0:
        return matrix
    keep = [i for i, arxiv_id in enumerate(ids) if arxiv_id not in exclude]
    if len(keep) == len(ids):
        return matrix
    return matrix[keep] if keep else np.zeros((0, matrix.shape[1]), dtype=matrix.dtype)


def leave_one_out(
    doc_ids: list[str],
    docs: "np.ndarray",
    pools: Pools,
    labels: dict[str, int],
    alpha: float,
    beta: float,
    k: int = 10,
    exclude_from_ranking: set[str] | None = None,
) -> dict:
    """Score each labeled paper with itself held out of the pools.

    With 55 labels there is no honest train/test split -- holding out 20% means
    holding out 0.4 positives. Leave-one-out is the standard answer at this
    size: every paper is scored by a model that never saw it.

    `exclude_from_ranking` removes papers from the ranked universe without
    removing them from the pools. Curated training papers need this: they are
    in their own liked pool, so with alpha > 0 they match themselves and
    occupy the top slots, pushing every evaluated paper down by a constant.
    They would not appear in a fresh weekly pool either, so ranking against
    them measures nothing real.

    Returns the mean rank of positives (lower is better), precision@k over
    labeled papers, and the per-paper ranks for inspection.
    """
    import numpy as np

    index = {arxiv_id: i for i, arxiv_id in enumerate(doc_ids)}
    positives = [i for i, r in labels.items() if r >= LIKED_AT and i in index]
    ranks: dict[str, int] = {}

    blocked = exclude_from_ranking or set()
    universe = np.array(
        [i for i, arxiv_id in enumerate(doc_ids) if arxiv_id not in blocked]
    )

    for arxiv_id in labels:
        if arxiv_id not in index or arxiv_id in blocked:
            continue
        held_out = score(docs, pools, alpha, beta, exclude_ids={arxiv_id})
        # Rank among the candidate universe only, 0 = best.
        order = universe[np.argsort(-held_out[universe])]
        position = int(np.where(order == index[arxiv_id])[0][0])
        ranks[arxiv_id] = position

    positive_ranks = [ranks[i] for i in positives if i in ranks]
    top_k = [i for i in labels if i in ranks and ranks[i] < k]
    hits = sum(1 for i in top_k if labels[i] >= LIKED_AT)

    return {
        "alpha": alpha,
        "beta": beta,
        "mean_positive_rank": float(np.mean(positive_ranks)) if positive_ranks else float("nan"),
        "median_positive_rank": float(np.median(positive_ranks)) if positive_ranks else float("nan"),
        "positives": len(positive_ranks),
        "precision_at_k": hits / len(top_k) if top_k else 0.0,
        "labeled_in_top_k": len(top_k),
        "ranks": ranks,
    }


def sweep(
    doc_ids: list[str],
    docs: "np.ndarray",
    pools: Pools,
    labels: dict[str, int],
    alphas: list[float],
    betas: list[float],
    k: int = 10,
    exclude_from_ranking: set[str] | None = None,
) -> list[dict]:
    """Every (alpha, beta) combination, best mean positive rank first.

    The row at alpha=beta=0 is the unlearned baseline. If it wins, ratings did
    not help -- report that rather than picking the second-best row.
    """
    results = [
        leave_one_out(doc_ids, docs, pools, labels, alpha, beta, k,
                      exclude_from_ranking=exclude_from_ranking)
        for alpha in alphas
        for beta in betas
    ]
    return sorted(results, key=lambda r: (r["mean_positive_rank"], r["alpha"], r["beta"]))
