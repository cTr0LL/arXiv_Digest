"""Metrics over the hand-labeled set.

Everything here has to account for stratified sampling. The labeled set is not
a random sample of the week -- the top of the prefilter ranking is sampled at
~60%, the tail at ~2% -- so counting positives directly would say the prefilter
is excellent no matter how it performs.

The fix is inverse-propensity weighting: a labeled paper from a stratum sampled
at rate r stands in for 1/r papers in the pool. A positive found in the tail
therefore counts for far more than one found at the top, which is exactly
right, because finding a good paper at rank 900 is strong evidence the filter
is losing things.
"""

from __future__ import annotations

from collections import defaultdict

from .labeling import RELEVANT_AT


def _weight(row: dict[str, str]) -> float:
    """How many pool papers this labeled paper represents."""
    sampled = int(row["stratum_sampled"])
    return int(row["stratum_size"]) / sampled if sampled else 0.0


def _is_positive(row: dict[str, str]) -> bool:
    return int(row["score"]) >= RELEVANT_AT


def stratum_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["stratum"]].append(row)

    for name, group in grouped.items():
        positives = sum(1 for r in group if _is_positive(r))
        size = int(group[0]["stratum_size"])
        sampled = int(group[0]["stratum_sampled"])
        out[name] = {
            "labeled": len(group),
            "positives": positives,
            "stratum_size": size,
            "sampling_rate": sampled / size if size else 0.0,
            "estimated_positives": positives * _weight(group[0]),
        }
    return out


def recall_at(rows: list[dict[str, str]], k: int) -> dict[str, float]:
    """Estimated fraction of all relevant papers that survive a cut at rank k.

    This is the number that decides whether `--keep` is safe to lower. It is an
    estimate with real uncertainty -- a single positive in the tail stratum
    moves it a lot -- so read it alongside `tail_positives` rather than alone.
    """
    total = 0.0
    kept = 0.0
    tail_positives = 0
    for row in rows:
        if not _is_positive(row):
            continue
        weight = _weight(row)
        total += weight
        if int(row["rank"]) < k:
            kept += weight
        elif row["stratum"] in ("mid", "tail"):
            tail_positives += 1

    return {
        "recall": kept / total if total else 0.0,
        "estimated_relevant_in_pool": total,
        "estimated_relevant_kept": kept,
        "tail_positives": tail_positives,
    }


def precision_at(rows: list[dict[str, str]], k: int) -> dict[str, float]:
    """Precision among labeled papers ranked above k.

    Unweighted on purpose: this only describes the labeled papers inside the
    top k, and the top stratum is densely sampled, so weighting would add
    noise without adding information. It is not an estimate about the pool.
    """
    inside = [r for r in rows if int(r["rank"]) < k]
    if not inside:
        return {"precision": 0.0, "labeled_in_top_k": 0, "positives": 0}
    positives = sum(1 for r in inside if _is_positive(r))
    return {
        "precision": positives / len(inside),
        "labeled_in_top_k": len(inside),
        "positives": positives,
    }


def report(rows: list[dict[str, str]], cuts: tuple[int, ...] = (50, 150, 400)) -> str:
    if not rows:
        return "No labels yet. Run: python label.py"

    lines = [
        f"Labeled: {len(rows)} papers, "
        f"{sum(1 for r in rows if _is_positive(r))} positive "
        f"(score >= {RELEVANT_AT})",
        "",
        f"{'stratum':<8} {'labeled':>7} {'pos':>4} {'pool':>6} {'rate':>6} {'est.pos':>8}",
    ]
    order = {"top": 0, "near": 1, "mid": 2, "tail": 3}
    for name, stats in sorted(stratum_summary(rows).items(), key=lambda x: order.get(x[0], 9)):
        lines.append(
            f"{name:<8} {stats['labeled']:>7.0f} {stats['positives']:>4.0f} "
            f"{stats['stratum_size']:>6.0f} {stats['sampling_rate']:>5.0%} "
            f"{stats['estimated_positives']:>8.1f}"
        )

    lines += ["", "Prefilter, if you cut at:", ""]
    for k in cuts:
        rec = recall_at(rows, k)
        prec = precision_at(rows, k)
        lines.append(
            f"  keep={k:<4} recall {rec['recall']:>5.0%}  "
            f"precision {prec['precision']:>5.0%} "
            f"({prec['positives']:.0f}/{prec['labeled_in_top_k']:.0f} labeled)"
        )

    missed = recall_at(rows, cuts[1] if len(cuts) > 1 else cuts[0])["tail_positives"]
    if missed:
        lines += [
            "",
            f"{missed} paper(s) you marked relevant sat below the default cut. "
            "Raise --keep, or accept that they are lost.",
        ]
    return "\n".join(lines)
