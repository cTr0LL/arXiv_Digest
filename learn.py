"""Stage 4: does learning from your ratings actually help?

    python learn.py              # sweep the weights, leave-one-out
    python learn.py --k 5        # precision at a different cut

Reads everything from the store: papers, cached embeddings, your ratings, and
the current profile. Costs nothing -- no API calls, no arXiv.

What it reports
---------------
A sweep over `alpha` (how much weight to put on papers you rated highly) and
`beta` (how much to subtract for papers you dismissed). The row at
`alpha = beta = 0` is the unlearned prefilter. If that row wins, ratings did
not help, and that is the result -- it is not a bug to tune away.

Every number is leave-one-out: each labeled paper is scored by a model with
that paper removed from the pools. Without that, a rated paper matches itself
at similarity 1.0 and the whole exercise reports a fiction.

The two rating sources are kept apart on purpose
------------------------------------------------
  source='known'  papers you already knew were relevant -> training signal
  source='label'  a stratified sample of a real pool    -> evaluation

Papers you have read are papers you *found*, which is a biased sample. The
stratified labels are not. Learning from the first and grading on the second is
what makes an improvement claim mean something.
"""

from __future__ import annotations

import argparse
import sys

from arxiv_digest import config, preferences
from arxiv_digest.preferences import Pools
from arxiv_digest.store import Store


def build_pools(store: Store, embed_model: str):
    import numpy as np

    from arxiv_digest import prefilter

    doc_ids, docs = store.get_embeddings(embed_model)
    if not doc_ids:
        raise SystemExit("No embeddings cached. Run: python init_store.py")

    profile = store.current_profile()
    if profile is None:
        raise SystemExit("No profile stored. Run: python init_store.py")
    wanted_text, unwanted_text = prefilter.parse_profile(profile["content"])

    wanted = prefilter.embed_texts(wanted_text, embed_model) if wanted_text else np.zeros((0, docs.shape[1]), np.float32)
    unwanted = prefilter.embed_texts(unwanted_text, embed_model) if unwanted_text else np.zeros((0, docs.shape[1]), np.float32)

    rows = store.conn.execute(
        "SELECT arxiv_id, rating, source FROM ratings"
    ).fetchall()
    position = {a: i for i, a in enumerate(doc_ids)}

    liked_ids = [r["arxiv_id"] for r in rows
                 if r["rating"] >= preferences.LIKED_AT and r["arxiv_id"] in position]
    disliked_ids = [r["arxiv_id"] for r in rows
                    if r["rating"] <= preferences.DISLIKED_AT and r["arxiv_id"] in position]

    def stack(ids):
        return (np.stack([docs[position[i]] for i in ids]) if ids
                else np.zeros((0, docs.shape[1]), np.float32))

    pools = Pools(
        wanted=wanted, unwanted=unwanted,
        liked=stack(liked_ids), liked_ids=liked_ids,
        disliked=stack(disliked_ids), disliked_ids=disliked_ids,
    )
    # Evaluation uses only the unbiased stratified labels.
    eval_labels = {r["arxiv_id"]: r["rating"] for r in rows
                   if r["source"] == "label" and r["arxiv_id"] in position}
    # Curated papers are training data, not candidates. They are not in a
    # fresh weekly pool, and with alpha > 0 they match themselves.
    curated = {r["arxiv_id"] for r in rows if r["source"] == "known"}
    return doc_ids, docs, pools, eval_labels, curated


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10, help="Precision cut. Default 10.")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 1.0])
    args = parser.parse_args()

    with Store(config.DB_PATH) as store:
        doc_ids, docs, pools, eval_labels, curated = build_pools(store, config.EMBED_MODEL)

        sources = store.conn.execute(
            "SELECT source, COUNT(*) AS n FROM ratings GROUP BY source"
        ).fetchall()
        print(f"Corpus: {len(doc_ids)} papers with vectors")
        print("Ratings: " + ", ".join(f"{r['source']}={r['n']}" for r in sources))
        print(f"Training pools: {len(pools.liked_ids)} liked, "
              f"{len(pools.disliked_ids)} disliked")
        print(f"Ranking universe: {len(doc_ids) - len(curated)} candidates "
              f"({len(curated)} curated papers held out of the ranking)")
        print(f"Evaluating on {len(eval_labels)} stratified label(s), "
              f"{sum(1 for v in eval_labels.values() if v >= preferences.LIKED_AT)} positive")

        positives = sum(1 for v in eval_labels.values() if v >= preferences.LIKED_AT)
        if positives < 5:
            print(f"\nWARNING: {positives} positive(s) in the eval set. Differences "
                  "below are well inside noise. Treat this as plumbing, not a result.")

        print(f"\nLeave-one-out sweep, precision@{args.k}:\n")
        rows = preferences.sweep(doc_ids, docs, pools, eval_labels,
                                 args.alphas, args.betas, k=args.k,
                                 exclude_from_ranking=curated)

        print(f"  {'alpha':>6} {'beta':>6} {'mean rank':>10} {'median':>8} {'prec@k':>8}")
        baseline = None
        for row in rows:
            marker = ""
            if row["alpha"] == 0.0 and row["beta"] == 0.0:
                baseline = row
                marker = "  <- unlearned baseline"
            print(f"  {row['alpha']:>6.2f} {row['beta']:>6.2f} "
                  f"{row['mean_positive_rank']:>10.1f} "
                  f"{row['median_positive_rank']:>8.1f} "
                  f"{row['precision_at_k']:>7.0%}{marker}")

        best = rows[0]
        print()
        if baseline and best["mean_positive_rank"] >= baseline["mean_positive_rank"]:
            print("Result: ratings did not improve ranking. The unlearned baseline "
                  "is as good as anything. Report that.")
        elif baseline:
            gain = baseline["mean_positive_rank"] - best["mean_positive_rank"]
            print(f"Result: best is alpha={best['alpha']}, beta={best['beta']}, "
                  f"moving positives {gain:.1f} places up on average "
                  f"({baseline['mean_positive_rank']:.1f} -> "
                  f"{best['mean_positive_rank']:.1f}).")
            print("        With this few positives, treat it as a direction to "
                  "retest on more labels, not a tuned setting to ship.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
