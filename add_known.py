"""Add papers you already know are relevant, as curated positive examples.

    python add_known.py https://arxiv.org/abs/2301.12345 2302.11111
    python add_known.py --file my_papers.txt
    cat urls.txt | python add_known.py

Accepts abs links, pdf links, bare ids, versioned ids -- anything with an arXiv
id in it. Fetches them in a single API request using `id_list`, stores them,
embeds them, and records a rating of 5 with source `known`.

Why these are kept separate from `label` ratings
------------------------------------------------
Papers you have read are papers you *found*, which correlates with being
well-publicised. Relevant work you never came across is systematically absent,
so this set is biased in a way a stratified sample is not.

That does not make it less useful -- it is the best available signal about what
you want -- but it does mean the two should not be pooled when measuring. The
intended split is:

  source='known'  ->  training signal: what good looks like
  source='label'  ->  evaluation: an unbiased sample of a real pool

Stage 4 learns from the first and is scored on the second. Mixing them would
mean tuning and grading on the same data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arxiv_digest import arxiv_client, config, prefilter
from arxiv_digest.store import Store


def collect_ids(tokens: list[str], file: str | None) -> tuple[list[str], list[str]]:
    """Return (ids, unrecognised). Reads stdin when given neither."""
    raw: list[str] = list(tokens)
    if file:
        raw += Path(file).read_text(encoding="utf-8").split()
    if not raw and not sys.stdin.isatty():
        raw += sys.stdin.read().split()

    ids, bad = [], []
    for token in raw:
        arxiv_id = arxiv_client.parse_arxiv_id(token)
        if arxiv_id is None:
            bad.append(token)
        elif arxiv_id not in ids:
            ids.append(arxiv_id)
    return ids, bad


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="*", help="arXiv URLs or ids.")
    parser.add_argument("--file", help="File with one URL or id per line.")
    parser.add_argument("--rating", type=int, default=5, help="1-5, default 5.")
    args = parser.parse_args()

    ids, bad = collect_ids(args.urls, args.file)
    if bad:
        print(f"Could not find an arXiv id in: {', '.join(bad[:5])}")
    if not ids:
        print("Nothing to add. Pass URLs, --file, or pipe them in.")
        return 1

    print(f"Fetching {len(ids)} paper(s) in "
          f"{(len(ids) + 49) // 50} request(s)...")
    papers = arxiv_client.fetch_by_ids(ids)

    found = {p.arxiv_id for p in papers}
    missing = [i for i in ids if i not in found]
    if missing:
        print(f"  not found on arXiv: {', '.join(missing)}")
    if not papers:
        return 1

    with Store(config.DB_PATH) as store:
        added = store.add_papers(papers)
        for paper in papers:
            store.set_rating(paper.arxiv_id, args.rating, source="known")

        pending = store.papers_missing_embeddings(config.EMBED_MODEL)
        if pending:
            store.set_embeddings(
                config.EMBED_MODEL, prefilter.embed_papers(pending, config.EMBED_MODEL)
            )

        print(f"\n{len(papers)} paper(s) recorded ({added} new to the store):")
        for paper in papers:
            print(f"  {args.rating}/5  {paper.arxiv_id}  {paper.title[:66]}")

        counts = store.conn.execute(
            "SELECT source, COUNT(*) AS n FROM ratings GROUP BY source"
        ).fetchall()
        print("\nRatings by source: " + ", ".join(f"{r['source']}={r['n']}" for r in counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
