"""Entry point for the weekly digest.

    python run_digest.py --dry-run     # retrieval only, no API key, no cost
    python run_digest.py               # the real thing

The funnel is: every submission in the window -> embedding prefilter -> model
scoring -> top K into the digest. Stage 2 replaces `baseline.run` with an agent
loop of the same shape, so nothing else has to change.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from arxiv_digest import agent, arxiv_client, baseline, config, prefilter, render


def main() -> int:
    # Progress prints are the only feedback during the slow fetch and embed
    # steps. Without this they buffer until exit when output is redirected,
    # which is exactly the case under cron.
    sys.stdout.reconfigure(line_buffering=True)
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and prefilter without calling the model. Costs nothing.",
    )
    parser.add_argument("--window", type=int, default=config.WINDOW_DAYS)
    parser.add_argument("--max-papers", type=int, default=config.MAX_PAPERS)
    parser.add_argument(
        "--keep",
        type=int,
        default=config.PREFILTER_KEEP,
        help="How many papers survive the prefilter and get scored by the model.",
    )
    parser.add_argument("--top", type=int, default=config.TOP_K)
    parser.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Score every fetched abstract. Expensive on a full week; use with --max-papers.",
    )
    parser.add_argument("--categories", nargs="+", default=config.CATEGORIES)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Stage 2: let an agent decide which candidates to read in full.",
    )
    parser.add_argument(
        "--max-reads",
        type=int,
        default=8,
        help="Full-text read budget for --agent. Each read is ~12k tokens.",
    )
    args = parser.parse_args()

    profile = baseline.load_profile()

    print(f"Fetching {args.window} days of {', '.join(args.categories)}...")
    papers = arxiv_client.fetch_window(
        categories=args.categories,
        window_days=args.window,
        max_papers=args.max_papers,
    )
    if not papers:
        print("Nothing in the window.")
        return 0
    fetched = len(papers)

    if args.no_prefilter:
        candidates = [(p, None) for p in papers]
        print(f"Prefilter disabled: all {fetched} papers go to the model.")
    else:
        print(f"Prefiltering {fetched} -> {args.keep} by similarity to your profile...")
        candidates = prefilter.rank(
            papers, profile, keep=args.keep, model_name=config.EMBED_MODEL
        )

    if args.dry_run:
        print(f"\nTop {min(25, len(candidates))} by embedding similarity:\n")
        for paper, score in candidates[:25]:
            marker = f"{score:.3f}" if score is not None else "  -  "
            print(f"  {marker}  {paper.primary_category:10s}  {paper.title[:76]}")
        print(f"\nDry run: {fetched} fetched, {len(candidates)} kept, no model calls.")
        return 0

    candidate_papers = [p for p, _ in candidates]
    context = None

    if args.agent:
        print(f"Agent triaging {len(candidate_papers)} candidates "
              f"(read budget {args.max_reads})...")
        context = agent.build_context(candidate_papers, max_reads=args.max_reads)
        ranked = agent.run(
            candidate_papers, profile, max_reads=args.max_reads, context=context
        )
    else:
        print(f"Scoring {len(candidate_papers)} candidates against your profile...")
        ranked = baseline.run(candidate_papers, profile=profile)

    markdown = render.render(
        ranked=ranked,
        top_k=args.top,
        scanned=fetched,
        prefiltered=None if args.no_prefilter else len(candidates),
        categories=args.categories,
        window_days=args.window,
        engine="agent" if args.agent else "baseline",
        reads=context.reads if context else None,
    )

    config.DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else config.DIGEST_DIR / f"{date.today().isoformat()}.md"
    out_path.write_text(markdown, encoding="utf-8")

    print(f"\nWrote {out_path}")
    if ranked:
        best_paper, best = ranked[0]
        print(f"Top pick ({best.score}/5): {best_paper.title}")
        high = sum(1 for _, a in ranked if a.score >= 4)
        print(f"{high} paper(s) scored 4 or better.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
