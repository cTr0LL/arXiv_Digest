"""Populate the SQLite store from the files the project already produced.

    python init_store.py

Reads `eval/pool.json`, `eval/labels.csv` and `profile.md` into `eval/digest.db`.
Safe to run repeatedly: papers are inserted only if new, ratings replace rather
than accumulate, and an unchanged profile does not create a new version.

After this the store is the source of truth for ratings. `eval/labels.csv` stays
as the labeling interface and a git-diffable artifact, but if the two disagree,
re-running this makes the database match the CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

from arxiv_digest import config, labeling
from arxiv_digest.store import Store

POOL_PATH = config.ROOT / "eval" / "pool.json"
LABELS_PATH = config.ROOT / "eval" / "labels.csv"


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)

    with Store(config.DB_PATH) as store:
        cached = labeling.load_pool(POOL_PATH)
        if cached:
            papers, meta = cached
            added = store.add_papers(papers)
            print(f"Papers:  {added} new, {store.count_papers()} total "
                  f"(pool fetched {meta.get('fetched_at', '?')})")
        else:
            print(f"Papers:  no pool at {POOL_PATH}, skipping")

        # Ratings made in conversation (via the MCP server's rate_paper) exist
        # only in the database. The CSV knows nothing about them, so a blind
        # import would silently revert them to whatever you first recorded.
        from_chat = {
            r["arxiv_id"] for r in store.conn.execute(
                "SELECT arxiv_id FROM ratings WHERE source = 'chat'"
            ).fetchall()
        }

        rows = labeling.load_labels(LABELS_PATH)
        imported, orphaned, preserved = 0, [], 0
        for row in rows:
            if row["arxiv_id"] in from_chat:
                preserved += 1
                continue
            try:
                store.set_rating(row["arxiv_id"], int(row["score"]), source="label")
                imported += 1
            except ValueError:
                # A label whose paper is not in the pool -- happens if the pool
                # was refreshed after labeling. Report rather than silently drop.
                orphaned.append(row["arxiv_id"])
        print(f"Ratings: {imported} imported")
        if preserved:
            print(f"         {preserved} kept as-is (rated in conversation, "
                  "newer than the CSV)")
        if orphaned:
            print(f"         {len(orphaned)} skipped, paper not in pool: "
                  f"{', '.join(orphaned[:5])}")

        if config.PROFILE_PATH.exists():
            version = store.save_profile(
                config.PROFILE_PATH.read_text(encoding="utf-8").strip()
            )
            print(f"Profile: version {version}")
        else:
            print("Profile: no profile.md, skipping")

        # Embeddings are what make the MCP server's search semantic rather than
        # substring. Incremental: only papers without a vector for this model.
        pending = store.papers_missing_embeddings(config.EMBED_MODEL)
        if pending:
            from arxiv_digest import prefilter

            print(f"Vectors: embedding {len(pending)} paper(s)...")
            store.set_embeddings(
                config.EMBED_MODEL, prefilter.embed_papers(pending, config.EMBED_MODEL)
            )
        print(f"Vectors: {store.count_embeddings(config.EMBED_MODEL)} cached "
              f"({config.EMBED_MODEL})")

        print()
        stats = store.stats()
        print(f"Store at {config.DB_PATH}")
        print(f"  {stats['papers']} papers, {stats['rated']} rated "
              f"({stats['positives']} at 4+), "
              f"{stats['profile_versions']} profile version(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
