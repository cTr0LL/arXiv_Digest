"""Reading your ratings back out of past digests.

Without this the rating block at the bottom of every digest is decorative: you
fill it in, and nothing ever reads it. This closes the loop, and it is the only
difference between a weekly newsletter and something that learns.

Digests are scanned rather than tracked, so it does not matter which ones you
rated or in what order. Re-running is harmless: `set_rating` replaces.
"""

from __future__ import annotations

from pathlib import Path

from . import render
from .store import Store


def ingest_digest_ratings(store: Store, digest_dir: Path) -> dict[str, int]:
    """Import every rating found in `digest_dir`. Returns what was imported.

    Ratings you made in conversation are left alone. Those are newer than any
    Markdown file and re-importing an old digest must not revert them.
    """
    if not digest_dir.exists():
        return {}

    from_chat = {
        row["arxiv_id"]
        for row in store.conn.execute("SELECT arxiv_id FROM ratings WHERE source = 'chat'")
    }

    imported: dict[str, int] = {}
    for path in sorted(digest_dir.glob("*.md")):
        for arxiv_id, rating in render.parse_ratings(path.read_text(encoding="utf-8")).items():
            if arxiv_id in from_chat:
                continue
            try:
                store.set_rating(arxiv_id, rating, source="digest")
                imported[arxiv_id] = rating
            except ValueError:
                # Rated a paper the store never saw. Possible if the database
                # was rebuilt; skip rather than crash the weekly run.
                continue
    return imported


def recently_covered(store: Store, days: int) -> set[str]:
    """Papers already featured in a digest within the last `days`.

    Fetch windows overlap, so the same paper qualifies several weeks running.
    Showing it again wastes the slot and makes the digest feel broken.
    """
    if days <= 0:
        return set()
    rows = store.conn.execute(
        """SELECT DISTINCT a.arxiv_id
           FROM assessments a JOIN runs r USING (run_id)
           WHERE r.created_at >= datetime('now', ?)
             AND a.rank < r.top_k""",
        (f"-{days} days",),
    ).fetchall()
    return {r["arxiv_id"] for r in rows}
