"""Score the ranking engines against your hand labels.

    python evaluate.py                 # prefilter only, free
    python evaluate.py --baseline      # + stage 1  (~$0.15)
    python evaluate.py --baseline --agent --max-reads 3   # + stage 2

The headline metric is deliberately not precision@k. With a few dozen labels
and a handful of positives, most of any engine's top 8 is unlabeled, so
precision@8 is computed over whichever two or three papers happen to have
labels -- a number that swings wildly and describes the sample, not the engine.

Instead the primary report is **where each known positive landed** under each
engine. That is robust to unlabeled papers: if you marked a paper `y` and one
engine ranks it 23rd while another ranks it 2nd, that comparison is real
regardless of what the other 198 papers are.

Precision over labeled papers is still printed, with its sample size attached,
because it is the number people expect. Read the `n=` before believing it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from arxiv_digest import agent, baseline, config, labeling, metrics, prefilter

POOL_PATH = config.ROOT / "eval" / "pool.json"
LABELS_PATH = config.ROOT / "eval" / "labels.csv"


def main() -> int:
    load_dotenv()
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=int, default=40)
    parser.add_argument("--baseline", action="store_true", help="Run stage 1 (costs money).")
    parser.add_argument("--agent", action="store_true", help="Run stage 2 (costs more).")
    parser.add_argument("--max-reads", type=int, default=3)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 8, 10])
    args = parser.parse_args()

    cached = labeling.load_pool(POOL_PATH)
    if not cached:
        print("No pool. Run: python label.py")
        return 1
    papers, _ = cached

    rows = labeling.load_labels(LABELS_PATH)
    if not rows:
        print("No labels. Run: python label.py")
        return 1
    labels = {r["arxiv_id"]: int(r["score"]) for r in rows}
    positives = {i for i, s in labels.items() if s >= labeling.RELEVANT_AT}
    maybes = {i for i, s in labels.items() if s == 3}

    print(f"{len(papers)} papers, {len(labels)} labeled, "
          f"{len(positives)} positive, {len(maybes)} maybe\n")
    if len(positives) < 5:
        print(f"NOTE: only {len(positives)} positive label(s). Every number below is "
              "directional at best. Label more before drawing conclusions.\n")

    profile = baseline.load_profile()

    print(f"Prefilter: ranking {len(papers)} papers...")
    full = prefilter.rank(papers, profile, keep=None, model_name=config.EMBED_MODEL)
    rankings: dict[str, list[str]] = {"prefilter": [p.arxiv_id for p, _ in full]}

    candidates = [p for p, _ in full[: args.keep]]

    if args.baseline:
        print(f"\nStage 1: scoring {len(candidates)} candidates...")
        ranked = baseline.run(candidates, profile=profile)
        rankings["baseline"] = [p.arxiv_id for p, _ in ranked]

    if args.agent:
        print(f"\nStage 2: agent over {len(candidates)} candidates "
              f"(read budget {args.max_reads})...")
        context = agent.build_context(candidates, max_reads=args.max_reads)
        ranked = agent.run(candidates, profile, max_reads=args.max_reads, context=context)
        rankings["agent"] = [p.arxiv_id for p, _ in ranked]
        if context.reads:
            print(f"  read in full: {', '.join(context.reads)}")

    print("\n" + "=" * 72)
    print(metrics.compare_rankings(rankings, labels, positives, maybes, tuple(args.ks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
