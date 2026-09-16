"""MCP server over the digest store.

This is the piece worth writing yourself. Consuming somebody else's MCP server
is configuration; writing one teaches you what the protocol actually is -- a
tool surface described well enough that a model it has never met can use it
correctly on the first try.

Run it directly for a quick check:

    python mcp_server.py

It speaks JSON-RPC over stdin/stdout, so it will look like it has hung. That is
correct: it is waiting for a client. Ctrl-C to exit.

To use it from Claude Desktop, add this to claude_desktop_config.json:

    {
      "mcpServers": {
        "arxiv-digest": {
          "command": "F:/ML_Projects/arXiv_Digest/.venv/Scripts/python.exe",
          "args": ["F:/ML_Projects/arXiv_Digest/mcp_server.py"]
        }
      }
    }

Forward slashes on purpose: backslashes must be doubled in JSON, and a single
one silently invalidates the whole file. Windows accepts either.

Then the weekly cron run and your ad-hoc questions share one set of tools and
one database, which is the whole argument for MCP here.

A note on the docstrings below: they are not documentation for you, they are
the tool descriptions the model sees and the only thing it has to go on when
deciding which tool to call. Vague ones produce wrong calls. Treat them as
part of the interface, and edit them when the model misuses a tool.
"""

from __future__ import annotations

import contextlib
import sys

from mcp.server.mcpserver import MCPServer

from arxiv_digest import config
from arxiv_digest.store import Store

server = MCPServer(
    name="arxiv-digest",
    instructions=(
        "Personal arXiv research digest. Holds papers already seen, the "
        "verdicts each ranking engine gave them, the user's own ratings, and "
        "the version history of their research profile. Use it to answer "
        "questions about what the user has read, saved or dismissed, and to "
        "record new ratings when they tell you about one."
    ),
)


def _store() -> Store:
    return Store(config.DB_PATH)


def _semantic(store: Store, query: str, limit: int) -> list | None:
    """Semantic hits, or None if semantic search is not available.

    Returning None rather than raising lets `search_papers` fall back to
    substring matching. Three things can make it unavailable: no embeddings
    cached yet (run `python init_store.py`), sentence-transformers not
    installed, or the model failing to load. None of those should turn a
    search into an error the model has to interpret.

    The model is only loaded here, lazily. Importing torch at module scope
    would add ~10s and ~900MB to every start of this server, including the ones
    where the user only wanted `digest_stats`.
    """
    if store.count_embeddings(config.EMBED_MODEL) == 0:
        return None
    try:
        from arxiv_digest import prefilter

        # Belt and braces: this server speaks JSON-RPC over stdout, and a
        # single byte printed by a third-party library corrupts the stream
        # and drops the connection. Anything the model loader emits goes
        # to stderr instead, where it is harmless.
        with contextlib.redirect_stdout(sys.stderr):
            vector = prefilter.embed_texts([query], config.EMBED_MODEL)[0]
    except Exception:
        return None
    return store.semantic_search(vector, config.EMBED_MODEL, limit=limit)


def _format_paper(paper, rating: int | None, assessments: list[dict]) -> str:
    lines = [
        f"{paper.title}",
        f"  {paper.arxiv_id} | {paper.primary_category} | "
        f"submitted {paper.published.date()}",
        f"  {paper.abs_url}",
        f"  {paper.byline()}",
    ]
    if rating is not None:
        lines.append(f"  Your rating: {rating}/5")
    for assessment in assessments:
        lines.append(
            f"  {assessment['engine']}: {assessment['score']}/5 "
            f"(rank {assessment['rank']}) - {assessment['relevance']}"
        )
    return "\n".join(lines)


@server.tool()
def search_papers(query: str, limit: int = 10, exact: bool = False) -> str:
    """Search papers the digest has already seen, by meaning.

    Use this for questions like "what did I see about replay buffers last
    month" or "anything on continual learning under domain shift". The query is
    matched semantically, so it finds papers that are *about* the topic even
    when they never use your words -- describe the subject rather than guessing
    at keywords.

    Set exact=True to fall back to literal substring matching over title and
    abstract. That is the right choice for a distinctive name you expect to
    appear verbatim, such as a dataset ("CheXpert"), a method ("CODA-Prompt")
    or an author.

    Only covers papers a digest or labeling run has already stored. It is not a
    search of arXiv itself.
    """
    with _store() as store:
        if not exact:
            hits = _semantic(store, query, limit)
            if hits is not None:
                if not hits:
                    return f"No stored papers resemble {query!r}."
                out = [f"{len(hits)} paper(s) most similar to {query!r}:", ""]
                for paper, score in hits:
                    out.append(_format_paper(paper, store.get_rating(paper.arxiv_id), []))
                    out.append(f"  similarity: {score:.3f}")
                    out.append("")
                return "\n".join(out)

        papers = store.search_papers(query, limit=limit)
        if not papers:
            return f"No stored papers contain {query!r}."
        out = [f"{len(papers)} literal match(es) for {query!r}:", ""]
        for paper in papers:
            out.append(_format_paper(paper, store.get_rating(paper.arxiv_id), []))
            out.append("")
        return "\n".join(out)


@server.tool()
def get_paper(arxiv_id: str) -> str:
    """Everything stored about one paper: metadata, every engine's verdict, your rating.

    Use the bare arXiv id, for example "2609.17068", without a version suffix.
    """
    with _store() as store:
        paper = store.get_paper(arxiv_id)
        if paper is None:
            return f"{arxiv_id} is not in the store."
        detail = _format_paper(
            paper, store.get_rating(arxiv_id), store.assessments_for(arxiv_id)
        )
        return f"{detail}\n\n  Abstract: {paper.abstract}"


@server.tool()
def list_rated(min_rating: int = 4, limit: int = 20) -> str:
    """Papers the user has rated, highest first.

    Defaults to 4 and above, which is "would read". Pass min_rating=1 to
    include dismissals -- useful when the user asks what they have rejected, or
    when you want to characterise their taste rather than just their favourites.
    """
    with _store() as store:
        rows = store.rated_papers(min_rating=min_rating, limit=limit)
        if not rows:
            return f"No papers rated {min_rating} or above yet."
        out = [f"{len(rows)} paper(s) rated {min_rating}+:", ""]
        for row in rows:
            out.append(f"  {row['rating']}/5  {row['title']}")
            out.append(f"        {row['arxiv_id']} | {row['abs_url']}")
        return "\n".join(out)


@server.tool()
def rate_paper(arxiv_id: str, rating: int, note: str = "") -> str:
    """Record the user's rating of a paper, 1 (unrelated) to 5 (read today).

    Only call this when the user has actually expressed a judgment about a
    specific paper. Re-rating replaces the previous value rather than adding a
    second one. The paper must already be in the store.
    """
    with _store() as store:
        try:
            store.set_rating(arxiv_id, rating, source="chat", note=note)
        except ValueError as exc:
            return f"Error: {exc}"
        paper = store.get_paper(arxiv_id)
        return f"Recorded {max(1, min(5, rating))}/5 for {arxiv_id}: {paper.title}"


@server.tool()
def get_profile() -> str:
    """The user's current research profile: the prose describing what they work on.

    This text drives every ranking decision the digest makes, so read it before
    reasoning about why a paper was or was not surfaced.
    """
    with _store() as store:
        current = store.current_profile()
        if current is None:
            return "No profile stored yet. Run a digest or labeling session first."
        history = store.profile_history()
        return (
            f"Profile version {current['version']} "
            f"(saved {current['created_at']}, {len(history)} version(s) total):\n\n"
            f"{current['content']}"
        )


@server.tool()
def update_profile(content: str) -> str:
    """Replace the research profile with new text, saving it as a new version.

    This changes what every future digest surfaces, so only call it when the
    user has clearly asked to change their stated interests. Previous versions
    are kept and nothing is destroyed, but the next digest will rank
    differently. An identical body is not saved twice.
    """
    with _store() as store:
        version = store.save_profile(content)
        return f"Profile saved as version {version}."


@server.tool()
def digest_stats() -> str:
    """Counts across the whole store: papers seen, ratings given, runs, profile versions.

    A good first call when the user asks something open-ended about their
    reading history, since it tells you how much data actually exists.
    """
    with _store() as store:
        stats = store.stats()
        spread = ", ".join(
            f"{rating}:{count}" for rating, count in sorted(stats["rating_counts"].items())
        )
        runs = ", ".join(f"{k}:{v}" for k, v in stats["runs"].items()) or "none"
        return (
            f"Papers seen: {stats['papers']}\n"
            f"Rated: {stats['rated']} ({stats['positives']} at 4+)\n"
            f"Rating spread: {spread or 'none'}\n"
            f"Runs: {runs}\n"
            f"Profile versions: {stats['profile_versions']}"
        )


@server.resource("profile://current")
def profile_resource() -> str:
    """The current research profile, as a readable resource."""
    with _store() as store:
        current = store.current_profile()
        return current["content"] if current else "No profile stored yet."


if __name__ == "__main__":
    server.run(transport="stdio")
