"""SQLite store: the memory the rest of the project has been missing.

Until now every run started from nothing. Papers already covered came back,
ratings lived in Markdown files that only `render.parse_ratings` understood,
and the profile was a file with no history. This module is the single place all
of that lives, and stage 3's MCP server is a thin wrapper over it.

Design notes worth knowing:

- **The store is the source of truth for ratings.** `eval/labels.csv` becomes an
  export, not a second master copy. Two writable copies of the same data
  diverge, and you find out months later when a number looks wrong.
- **Assessments are keyed by engine and run.** The prefilter, stage 1 and stage
  2 all produce a verdict for the same paper; keeping them separate is what
  makes "did the agent beat the baseline" answerable after the fact, without
  re-running anything.
- **The profile is versioned.** Stage 4 is about interests drifting, so the
  question "what did my profile say when I rated this paper?" has to be
  answerable. Overwriting `profile.md` in place destroys exactly that.
- **Every write is idempotent.** Re-running a digest must not duplicate rows or
  double-count anything.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import Assessment, Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id         TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    abstract         TEXT NOT NULL,
    authors          TEXT NOT NULL,   -- JSON array
    primary_category TEXT NOT NULL,
    categories       TEXT NOT NULL,   -- JSON array
    published        TEXT NOT NULL,   -- ISO 8601 UTC
    abs_url          TEXT NOT NULL,
    pdf_url          TEXT NOT NULL,
    first_seen       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    engine      TEXT NOT NULL,
    categories  TEXT NOT NULL,
    window_days INTEGER NOT NULL,
    scanned     INTEGER NOT NULL,
    kept        INTEGER NOT NULL,
    top_k       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS assessments (
    arxiv_id   TEXT NOT NULL REFERENCES papers(arxiv_id),
    run_id     INTEGER REFERENCES runs(run_id),
    engine     TEXT NOT NULL,
    score      INTEGER NOT NULL,
    method     TEXT NOT NULL DEFAULT '',
    relevance  TEXT NOT NULL DEFAULT '',
    rank       INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, run_id, engine)
);

-- Your judgments. One row per paper: re-rating replaces, it does not stack.
CREATE TABLE IF NOT EXISTS ratings (
    arxiv_id   TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
    rating     INTEGER NOT NULL,
    source     TEXT NOT NULL,       -- 'label' | 'digest'
    note       TEXT NOT NULL DEFAULT '',
    rated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profile_versions (
    version    INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Cached document vectors, so semantic search does not re-embed 200 abstracts
-- on every query. Keyed by model: vectors from different models are not
-- comparable, and silently mixing them produces plausible-looking nonsense.
CREATE TABLE IF NOT EXISTS embeddings (
    arxiv_id   TEXT NOT NULL REFERENCES papers(arxiv_id),
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,   -- float32, L2-normalised
    created_at TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, model)
);

CREATE INDEX IF NOT EXISTS idx_papers_published  ON papers(published);
CREATE INDEX IF NOT EXISTS idx_assess_engine     ON assessments(engine, score);
CREATE INDEX IF NOT EXISTS idx_ratings_rating    ON ratings(rating);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """All persistence for the project. Safe to open repeatedly."""

    def __init__(self, path: Path | str = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    # Columns added after the first release. `CREATE TABLE IF NOT EXISTS` does
    # nothing to a table that already exists, so a database created before a
    # column was added never gets it. Each entry is (table, column, definition)
    # and is applied only when missing.
    MIGRATIONS = [
        ("runs", "top_k", "INTEGER NOT NULL DEFAULT 0"),
    ]

    def _migrate(self) -> None:
        for table, column, definition in self.MIGRATIONS:
            existing = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if existing and column not in existing:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- papers -----------------------------------------------------------

    def add_papers(self, papers: list[Paper]) -> int:
        """Insert papers, ignoring ones already stored. Returns the new count.

        `first_seen` is preserved for papers already present: the point of the
        column is when *you* first encountered it, not when it was last seen.
        """
        before = self.count_papers()
        self.conn.executemany(
            """INSERT OR IGNORE INTO papers
               (arxiv_id, title, abstract, authors, primary_category,
                categories, published, abs_url, pdf_url, first_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    p.arxiv_id, p.title, p.abstract, json.dumps(p.authors),
                    p.primary_category, json.dumps(p.categories),
                    p.published.isoformat(), p.abs_url, p.pdf_url, _now(),
                )
                for p in papers
            ],
        )
        self.conn.commit()
        return self.count_papers() - before

    def get_paper(self, arxiv_id: str) -> Paper | None:
        row = self.conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return _row_to_paper(row) if row else None

    def count_papers(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def seen_ids(self, arxiv_ids: list[str]) -> set[str]:
        """Which of these we already have. Used to skip re-covering papers."""
        if not arxiv_ids:
            return set()
        marks = ",".join("?" * len(arxiv_ids))
        rows = self.conn.execute(
            f"SELECT arxiv_id FROM papers WHERE arxiv_id IN ({marks})", arxiv_ids
        ).fetchall()
        return {r["arxiv_id"] for r in rows}

    def search_papers(self, query: str, limit: int = 20) -> list[Paper]:
        """Substring search over title and abstract, newest first."""
        pattern = f"%{query.strip()}%"
        rows = self.conn.execute(
            """SELECT * FROM papers
               WHERE title LIKE ? OR abstract LIKE ?
               ORDER BY published DESC LIMIT ?""",
            (pattern, pattern, limit),
        ).fetchall()
        return [_row_to_paper(r) for r in rows]

    # --- embeddings -------------------------------------------------------

    def set_embeddings(self, model: str, vectors: dict[str, "np.ndarray"]) -> int:
        """Cache document vectors. Expects L2-normalised float32.

        Normalising on the way in means search is a dot product rather than a
        cosine computation, and means a caller cannot accidentally mix
        normalised and raw vectors in one table.
        """
        import numpy as np

        rows = []
        for arxiv_id, vector in vectors.items():
            vector = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            rows.append((arxiv_id, model, len(vector), vector.tobytes(), _now()))

        self.conn.executemany(
            """INSERT OR REPLACE INTO embeddings
               (arxiv_id, model, dim, vector, created_at) VALUES (?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def get_embeddings(self, model: str) -> tuple[list[str], "np.ndarray"]:
        """Every cached vector for `model`, as (ids, matrix)."""
        import numpy as np

        rows = self.conn.execute(
            "SELECT arxiv_id, dim, vector FROM embeddings WHERE model = ? ORDER BY arxiv_id",
            (model,),
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)

        ids = [r["arxiv_id"] for r in rows]
        matrix = np.stack(
            [np.frombuffer(r["vector"], dtype=np.float32, count=r["dim"]) for r in rows]
        )
        return ids, matrix

    def papers_missing_embeddings(self, model: str) -> list[Paper]:
        """Papers with no vector for this model yet, so backfill is incremental."""
        rows = self.conn.execute(
            """SELECT p.* FROM papers p
               LEFT JOIN embeddings e
                 ON e.arxiv_id = p.arxiv_id AND e.model = ?
               WHERE e.arxiv_id IS NULL""",
            (model,),
        ).fetchall()
        return [_row_to_paper(r) for r in rows]

    def semantic_search(
        self, query_vector: "np.ndarray", model: str, limit: int = 10
    ) -> list[tuple[Paper, float]]:
        """Papers most similar to a query vector, best first.

        Takes a vector rather than a string so this module never imports
        sentence-transformers: the store stays a storage layer, and the ~900MB
        of torch only loads in processes that actually embed something.
        """
        import numpy as np

        ids, matrix = self.get_embeddings(model)
        if not ids:
            return []

        query = np.asarray(query_vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm
        # Stored vectors are normalised, so a dot product is cosine similarity.
        scores = matrix @ query

        best = np.argsort(-scores)[:limit]
        out = []
        for index in best:
            paper = self.get_paper(ids[int(index)])
            if paper is not None:
                out.append((paper, float(scores[int(index)])))
        return out

    def count_embeddings(self, model: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)
        ).fetchone()[0]

    # --- runs and assessments --------------------------------------------

    def start_run(
        self, engine: str, categories: list[str], window_days: int,
        scanned: int, kept: int, top_k: int = 0,
    ) -> int:
        cursor = self.conn.execute(
            """INSERT INTO runs
               (created_at, engine, categories, window_days, scanned, kept, top_k)
               VALUES (?,?,?,?,?,?,?)""",
            (_now(), engine, json.dumps(categories), window_days, scanned, kept, top_k),
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_assessments(
        self, run_id: int, engine: str, ranked: list[tuple[Paper, Assessment]]
    ) -> None:
        self.conn.executemany(
            """INSERT OR REPLACE INTO assessments
               (arxiv_id, run_id, engine, score, method, relevance, rank, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                (p.arxiv_id, run_id, engine, a.score, a.method, a.relevance, rank, _now())
                for rank, (p, a) in enumerate(ranked)
            ],
        )
        self.conn.commit()

    def assessments_for(self, arxiv_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT engine, score, method, relevance, rank, run_id, created_at
               FROM assessments WHERE arxiv_id = ? ORDER BY run_id DESC""",
            (arxiv_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- ratings ----------------------------------------------------------

    def set_rating(self, arxiv_id: str, rating: int, source: str = "label", note: str = "") -> None:
        """Record your judgment. Re-rating replaces rather than appends."""
        if self.get_paper(arxiv_id) is None:
            raise ValueError(f"{arxiv_id} is not in the store; add the paper first.")
        self.conn.execute(
            """INSERT INTO ratings (arxiv_id, rating, source, note, rated_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(arxiv_id) DO UPDATE SET
                 rating = excluded.rating, source = excluded.source,
                 note = excluded.note, rated_at = excluded.rated_at""",
            (arxiv_id, max(1, min(5, rating)), source, note, _now()),
        )
        self.conn.commit()

    def get_rating(self, arxiv_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT rating FROM ratings WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return row["rating"] if row else None

    def rated_papers(self, min_rating: int = 1, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            """SELECT p.arxiv_id, p.title, p.abs_url, p.published,
                      r.rating, r.source, r.rated_at
               FROM ratings r JOIN papers p USING (arxiv_id)
               WHERE r.rating >= ?
               ORDER BY r.rating DESC, r.rated_at DESC LIMIT ?""",
            (min_rating, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def rating_counts(self) -> dict[int, int]:
        rows = self.conn.execute(
            "SELECT rating, COUNT(*) AS n FROM ratings GROUP BY rating"
        ).fetchall()
        return {r["rating"]: r["n"] for r in rows}

    # --- profile ----------------------------------------------------------

    def save_profile(self, content: str) -> int:
        """Store a new profile version, unless it is identical to the current one.

        Stage 4 tracks interest drift, which needs a history. Skipping no-op
        saves keeps that history meaningful rather than one row per run.
        """
        current = self.current_profile()
        if current is not None and current["content"] == content:
            return current["version"]
        cursor = self.conn.execute(
            "INSERT INTO profile_versions (content, created_at) VALUES (?,?)",
            (content, _now()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def current_profile(self) -> dict | None:
        row = self.conn.execute(
            "SELECT version, content, created_at FROM profile_versions "
            "ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def profile_history(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT version, created_at, LENGTH(content) AS chars "
            "FROM profile_versions ORDER BY version DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # --- summary ----------------------------------------------------------

    def stats(self) -> dict:
        counts = self.rating_counts()
        runs = self.conn.execute(
            "SELECT engine, COUNT(*) AS n FROM runs GROUP BY engine"
        ).fetchall()
        return {
            "papers": self.count_papers(),
            "rated": sum(counts.values()),
            "rating_counts": counts,
            "positives": sum(n for r, n in counts.items() if r >= 4),
            "runs": {r["engine"]: r["n"] for r in runs},
            "profile_versions": len(self.profile_history()),
        }


def _row_to_paper(row: sqlite3.Row) -> Paper:
    return Paper(
        arxiv_id=row["arxiv_id"],
        title=row["title"],
        abstract=row["abstract"],
        authors=json.loads(row["authors"]),
        primary_category=row["primary_category"],
        categories=json.loads(row["categories"]),
        published=datetime.fromisoformat(row["published"]),
        abs_url=row["abs_url"],
        pdf_url=row["pdf_url"],
    )
