"""Entry point for the weekly digest.

    python run_digest.py --dry-run     # arXiv only, no API key, no cost
    python run_digest.py               # the real thing

"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from arxiv_digest import arxiv_client, baseline, config, render


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and list papers without calling the model. Costs nothing.",
    )
    parser.add_argument("--limit", type=int, default=config.FETCH_LIMIT)
    parser.add_argument("--window", type=int, default=config.WINDOW_DAYS)
    parser.add_argument("--top", type=int, default=config.TOP_K)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=config.CATEGORIES,
        help="arXiv categories, e.g. --categories cs.LG eess.IV",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path. Defaults to digests/<today>.md",
    )
    args = parser.parse_args()

    print(f"Fetching up to {args.limit} recent submissions in {', '.join(args.categories)}...")
    papers = arxiv_client.fetch_recent(
        categories=args.categories, limit=args.limit, window_days=args.window
    )
    print(f"{len(papers)} submitted in the last {args.window} days.")

    if len(papers) == args.limit:
        print(
            "  note: every fetched paper fell inside the window, so older ones were "
            "likely cut off. Raise --limit or FETCH_LIMIT to see the full week."
        )

    if not papers:
        print("Nothing to digest.")
        return 0

    if args.dry_run:
        for paper in papers:
            print(f"  {paper.published.date()}  {paper.primary_category:10s}  {paper.title[:80]}")
        print(f"\nDry run: {len(papers)} papers, no model calls made.")
        return 0

    print("Scoring against your profile...")
    ranked = baseline.run(papers)

    markdown = render.render(
        ranked=ranked,
        top_k=args.top,
        scanned=len(papers),
        categories=args.categories,
        window_days=args.window,
    )

    config.DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out
        else config.DIGEST_DIR / f"{date.today().isoformat()}.md"
    )
    out_path.write_text(markdown, encoding="utf-8")

    print(f"\nWrote {out_path}")
    top = ranked[: args.top]
    if top:
        best = top[0]
        print(f"Top pick ({best[1].score}/5): {best[0].title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
