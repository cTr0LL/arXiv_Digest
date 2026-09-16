"""Hand-label a stratified sample of papers to build the eval set.

    python label.py              # label (resumes where you left off)
    python label.py --report     # just show the metrics
    python label.py --refresh    # re-fetch the paper pool

Costs nothing: no model calls, only arXiv and the local embedding filter.

The sample deliberately includes papers the prefilter ranked badly. Some of
what you see will be obvious junk -- that is the point. If you never see the
rejects, you cannot tell what the filter is throwing away.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import webbrowser
from pathlib import Path

from arxiv_digest import arxiv_client, baseline, config, labeling, metrics, prefilter

POOL_PATH = config.ROOT / "eval" / "pool.json"
LABELS_PATH = config.ROOT / "eval" / "labels.csv"

PROMPT = "[y] would read  [m] maybe  [n] no  [o] open  [s] skip  [q] quit > "


def get_pool(window: int, max_papers: int, categories: list[str], refresh: bool):
    if not refresh:
        cached = labeling.load_pool(POOL_PATH)
        if cached:
            papers, meta = cached
            print(f"Using cached pool: {len(papers)} papers ({meta.get('fetched_at', '?')})")
            print("  --refresh to re-fetch.")
            return papers

    print(f"Fetching {window} days of {', '.join(categories)}...")
    papers = arxiv_client.fetch_window(
        categories=categories, window_days=window, max_papers=max_papers
    )
    from datetime import datetime, timezone

    labeling.save_pool(
        POOL_PATH,
        papers,
        {
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "window_days": window,
            "categories": categories,
        },
    )
    return papers


def show(task: labeling.LabelTask, position: int, total: int) -> None:
    paper = task.paper
    print("\n" + "=" * 78)
    print(f"{position}/{total}   {paper.primary_category}   {paper.published.date()}")
    print("=" * 78)
    for line in textwrap.wrap(paper.title, 76):
        print(line)
    print()
    abstract = paper.abstract
    if len(abstract) > 1400:
        abstract = abstract[:1400].rsplit(" ", 1)[0] + " [...]"
    print(textwrap.fill(abstract, 76))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Show metrics and exit.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch the paper pool.")
    parser.add_argument("--window", type=int, default=14)
    parser.add_argument("--max-papers", type=int, default=config.MAX_PAPERS)
    parser.add_argument("--categories", nargs="+", default=config.CATEGORIES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    if args.report:
        print(metrics.report(labeling.load_labels(LABELS_PATH)))
        return 0

    profile = baseline.load_profile()
    papers = get_pool(args.window, args.max_papers, args.categories, args.refresh)
    if not papers:
        print("Empty pool.")
        return 1

    print(f"Ranking all {len(papers)} papers with the prefilter...")
    ranked = prefilter.rank(papers, profile, keep=None, model_name=config.EMBED_MODEL)

    done = labeling.labeled_ids(LABELS_PATH)
    tasks = labeling.build_sample(ranked, seed=args.seed, exclude=done)
    if not tasks:
        print(f"\nNothing left to label ({len(done)} done).\n")
        print(metrics.report(labeling.load_labels(LABELS_PATH)))
        return 0

    print(f"\n{len(tasks)} to label, {len(done)} already done.")
    print("Scores and ranks are hidden on purpose -- judge the paper, not the filter.")
    print("Every answer is saved immediately, so quitting loses nothing.")

    labeled_now = 0
    for index, task in enumerate(tasks, start=1):
        show(task, index, len(tasks))
        while True:
            try:
                key = input(PROMPT).strip().lower()
            except (EOFError, KeyboardInterrupt):
                key = "q"
                print()
            if key == "q":
                print(f"\nStopped. {labeled_now} labeled this session.\n")
                print(metrics.report(labeling.load_labels(LABELS_PATH)))
                return 0
            if key == "o":
                webbrowser.open(task.paper.abs_url)
                continue
            if key == "s":
                break
            if key in labeling.LABEL_VALUES:
                labeling.append_label(LABELS_PATH, task, key)
                labeled_now += 1
                break
            print(f"  '{key}' is not one of y/m/n/o/s/q")

    print(f"\nDone. {labeled_now} labeled this session.\n")
    print(metrics.report(labeling.load_labels(LABELS_PATH)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
