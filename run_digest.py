"""Entry point for the weekly digest.

    python run_digest.py --dry-run     # retrieval only, no API key, no cost
    python run_digest.py               # the real thing
    python run_digest.py --agent       # stage 2: agent picks what to read

The funnel is: every submission in the window -> embedding prefilter ->
model scoring -> top K into the digest.

What makes it weekly rather than 52 unrelated newsletters
---------------------------------------------------------
Each run, in order:

1. Reads your ratings out of past digests into the store. Without this the
   rating block at the bottom of every digest is decorative.
2. Suppresses papers already featured recently. Fetch windows overlap, so the
   same paper qualifies several weeks running.
3. Scores using the profile *and* your accumulated ratings, weighted by
   PREFERENCE_ALPHA / PREFERENCE_BETA in config.
4. Records the run, every verdict and its rank, so next week can do the same
   and so `evaluate.py` can compare engines after the fact.

`--no-store` skips all of that, for experiments you do not want in the record.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from arxiv_digest import (
    agent, arxiv_client, baseline, config, feedback, prefilter, preferences, render,
)
from arxiv_digest.store import Store


def open_log(enabled: bool):
    """Tee stdout to a dated log file. Scheduled runs have nowhere else to shout."""
    if not enabled:
        return None
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = (config.LOG_DIR / f"{date.today().isoformat()}.log").open("a", encoding="utf-8")
    handle.write(f"\n=== run started {datetime.now(timezone.utc).isoformat()} ===\n")

    class Tee:
        def write(self, text):
            sys.__stdout__.write(text)
            handle.write(text)

        def flush(self):
            sys.__stdout__.flush()
            handle.flush()

    sys.stdout = Tee()
    return handle


def rank_candidates(papers, profile, store, keep, use_store):
    """Prefilter, optionally weighted by ratings already in the store."""
    if not use_store or (config.PREFERENCE_ALPHA == 0 and config.PREFERENCE_BETA == 0):
        return prefilter.rank(papers, profile, keep=keep, model_name=config.EMBED_MODEL)

    import numpy as np

    wanted_text, unwanted_text = prefilter.parse_profile(profile)
    docs = prefilter.embed_texts(
        [prefilter.document_text(p) for p in papers], config.EMBED_MODEL
    )
    empty = np.zeros((0, docs.shape[1]), np.float32)

    rated = store.conn.execute("SELECT arxiv_id, rating FROM ratings").fetchall()
    ids, vectors = store.get_embeddings(config.EMBED_MODEL)
    position = {a: i for i, a in enumerate(ids)}

    def pool(predicate):
        chosen = [r["arxiv_id"] for r in rated
                  if predicate(r["rating"]) and r["arxiv_id"] in position]
        matrix = (np.stack([vectors[position[i]] for i in chosen]) if chosen else empty)
        return chosen, matrix

    liked_ids, liked = pool(lambda r: r >= preferences.LIKED_AT)
    disliked_ids, disliked = pool(lambda r: r <= preferences.DISLIKED_AT)
    print(f"  weighting by {len(liked_ids)} liked and {len(disliked_ids)} dismissed paper(s) "
          f"(alpha={config.PREFERENCE_ALPHA}, beta={config.PREFERENCE_BETA})")

    pools = preferences.Pools(
        wanted=prefilter.embed_texts(wanted_text, config.EMBED_MODEL) if wanted_text else empty,
        unwanted=prefilter.embed_texts(unwanted_text, config.EMBED_MODEL) if unwanted_text else empty,
        liked=liked, liked_ids=liked_ids, disliked=disliked, disliked_ids=disliked_ids,
    )
    scores = preferences.score(docs, pools, config.PREFERENCE_ALPHA, config.PREFERENCE_BETA)
    ranked = sorted(zip(papers, (float(s) for s in scores)), key=lambda x: -x[1])
    return ranked[:keep]


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and prefilter without calling the model. Costs nothing.")
    parser.add_argument("--window", type=int, default=config.WINDOW_DAYS)
    parser.add_argument("--max-papers", type=int, default=config.MAX_PAPERS)
    parser.add_argument("--keep", type=int, default=config.PREFILTER_KEEP)
    parser.add_argument("--top", type=int, default=config.TOP_K)
    parser.add_argument("--no-prefilter", action="store_true",
                        help="Score every fetched abstract. Expensive.")
    parser.add_argument("--categories", nargs="+", default=config.CATEGORIES)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--agent", action="store_true",
                        help="Stage 2: let an agent decide which candidates to read in full.")
    parser.add_argument("--max-reads", type=int, default=8)
    parser.add_argument("--no-store", action="store_true",
                        help="Do not read or write the database. For experiments.")
    parser.add_argument("--log", action="store_true",
                        help="Also write output to logs/. Scheduled runs want this.")
    args = parser.parse_args()

    log = open_log(args.log)
    try:
        return run(args)
    except Exception:
        # A scheduled run has no one watching. Make the failure legible in the
        # log rather than a silent non-zero exit code.
        print("\nRUN FAILED\n" + traceback.format_exc())
        return 1
    finally:
        if log:
            log.close()


def run(args) -> int:
    use_store = not args.no_store
    store = Store(config.DB_PATH) if use_store else None

    try:
        profile = baseline.load_profile()

        if use_store:
            store.save_profile(profile)
            imported = feedback.ingest_digest_ratings(store, config.DIGEST_DIR)
            if imported:
                print(f"Read {len(imported)} rating(s) out of past digests.")
            counts = store.rating_counts()
            print(f"Store: {store.count_papers()} papers, {sum(counts.values())} rated "
                  f"({sum(n for r, n in counts.items() if r >= 4)} positive)")

        print(f"Fetching {args.window} days of {', '.join(args.categories)}...")
        papers = arxiv_client.fetch_window(
            categories=args.categories, window_days=args.window, max_papers=args.max_papers,
        )
        if not papers:
            print("Nothing in the window.")
            return 0
        fetched = len(papers)

        if use_store:
            store.add_papers(papers)
            covered = feedback.recently_covered(store, config.SUPPRESS_SEEN_DAYS)
            before = len(papers)
            papers = [p for p in papers if p.arxiv_id not in covered]
            if before != len(papers):
                print(f"  skipping {before - len(papers)} paper(s) featured in the "
                      f"last {config.SUPPRESS_SEEN_DAYS} days")
            if not papers:
                print("Everything in the window has already been covered.")
                return 0

        if args.no_prefilter:
            candidates = [(p, None) for p in papers]
            print(f"Prefilter disabled: all {len(papers)} papers go to the model.")
        else:
            print(f"Prefiltering {len(papers)} -> {args.keep}...")
            candidates = rank_candidates(papers, profile, store, args.keep, use_store)

        if args.dry_run:
            print(f"\nTop {min(25, len(candidates))} by score:\n")
            for paper, score in candidates[:25]:
                marker = f"{score:.3f}" if score is not None else "  -  "
                print(f"  {marker}  {paper.primary_category:10s}  {paper.title[:74]}")
            print(f"\nDry run: {fetched} fetched, {len(candidates)} kept, no model calls.")
            return 0

        candidate_papers = [p for p, _ in candidates]
        context = None
        engine = "agent" if args.agent else "baseline"

        if args.agent:
            print(f"Agent triaging {len(candidate_papers)} candidates "
                  f"(read budget {args.max_reads})...")
            context = agent.build_context(candidate_papers, max_reads=args.max_reads)
            ranked = agent.run(candidate_papers, profile,
                               max_reads=args.max_reads, context=context)
        else:
            print(f"Scoring {len(candidate_papers)} candidates...")
            ranked = baseline.run(candidate_papers, profile=profile)

        markdown = render.render(
            ranked=ranked, top_k=args.top, scanned=fetched,
            prefiltered=None if args.no_prefilter else len(candidates),
            categories=args.categories, window_days=args.window,
            engine=engine, reads=context.reads if context else None,
        )

        config.DIGEST_DIR.mkdir(parents=True, exist_ok=True)
        out_path = Path(args.out) if args.out else config.DIGEST_DIR / f"{date.today().isoformat()}.md"
        out_path.write_text(markdown, encoding="utf-8")

        if use_store:
            run_id = store.start_run(engine, args.categories, args.window,
                                     fetched, len(candidates), args.top)
            store.add_assessments(run_id, engine, ranked)
            missing = store.papers_missing_embeddings(config.EMBED_MODEL)
            if missing:
                store.set_embeddings(
                    config.EMBED_MODEL,
                    prefilter.embed_papers(missing, config.EMBED_MODEL),
                )
            print(f"  recorded run {run_id} with {len(ranked)} verdict(s)")

        print(f"\nWrote {out_path}")
        if ranked:
            best_paper, best = ranked[0]
            print(f"Top pick ({best.score}/5): {best_paper.title}")
            print(f"{sum(1 for _, a in ranked if a.score >= 4)} paper(s) scored 4 or better.")
        print("Rate them in the file; next week's run reads them back automatically.")
        return 0
    finally:
        if store:
            store.close()


if __name__ == "__main__":
    sys.exit(main())
