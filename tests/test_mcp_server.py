"""Tests for the MCP server.

Two layers. The tool functions are tested directly, because that is where the
logic is. Then the registration itself is checked through the server's own
protocol surface, because a tool that works but never reaches `list_tools` is
invisible to every client -- and that failure is silent.

The description assertions are not pedantry. A tool description is the only
thing a model has when deciding whether to call it, so an empty or nameless one
is a real defect, not a style issue.
"""

from __future__ import annotations

import pytest

import mcp_server
from arxiv_digest.models import Assessment
from arxiv_digest.store import Store
from conftest import make_paper


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point the server at a throwaway database holding known data."""
    path = tmp_path / "test.db"
    monkeypatch.setattr(mcp_server.config, "DB_PATH", path)

    with Store(path) as store:
        papers = [make_paper(i, days_ago=i) for i in range(4)]
        store.add_papers(papers)
        store.set_rating(papers[0].arxiv_id, 5)
        store.set_rating(papers[1].arxiv_id, 1)
        store.save_profile("Replay methods for medical imaging.")
        run = store.start_run("digest", ["cs.LG"], 7, 200, 40)
        store.add_assessments(
            run, "baseline",
            [(papers[0], Assessment(arxiv_id=papers[0].arxiv_id, score=4,
                                    method="Does a thing.", relevance="Matches replay."))],
        )
    return path


# --- search_papers --------------------------------------------------------


def test_search_finds_stored_papers(db):
    result = mcp_server.search_papers("Paper 2")
    assert "Paper 2" in result
    assert make_paper(2).arxiv_id in result


def test_search_reports_no_matches_plainly(db):
    # No vectors cached in this fixture, so this exercises the substring path.
    assert "No stored papers contain" in mcp_server.search_papers("quantum chromodynamics")


def test_search_shows_your_rating_when_there_is_one(db):
    assert "Your rating: 5/5" in mcp_server.search_papers("Paper 0")


def test_search_respects_limit(db):
    assert mcp_server.search_papers("Paper", limit=2).startswith("2 literal match")


# --- get_paper ------------------------------------------------------------


def test_get_paper_includes_verdicts_and_abstract(db):
    result = mcp_server.get_paper(make_paper(0).arxiv_id)
    assert "baseline: 4/5" in result
    assert "Matches replay" in result
    assert "Abstract:" in result


def test_get_unknown_paper_says_so(db):
    assert "not in the store" in mcp_server.get_paper("9999.99999")


# --- list_rated -----------------------------------------------------------


def test_list_rated_defaults_to_positives_only(db):
    result = mcp_server.list_rated()
    assert "Paper 0" in result
    assert "Paper 1" not in result, "a 1/5 is not a recommendation"


def test_list_rated_can_include_dismissals(db):
    result = mcp_server.list_rated(min_rating=1)
    assert "Paper 0" in result and "Paper 1" in result


def test_list_rated_handles_an_empty_store(tmp_path, monkeypatch):
    empty = tmp_path / "empty.db"
    monkeypatch.setattr(mcp_server.config, "DB_PATH", empty)
    Store(empty).close()
    assert "No papers rated" in mcp_server.list_rated()


# --- rate_paper -----------------------------------------------------------


def test_rate_paper_records_and_confirms(db):
    result = mcp_server.rate_paper(make_paper(3).arxiv_id, 4)
    assert "Recorded 4/5" in result
    with Store(db) as store:
        assert store.get_rating(make_paper(3).arxiv_id) == 4


def test_rate_paper_replaces_rather_than_stacking(db):
    arxiv_id = make_paper(0).arxiv_id
    mcp_server.rate_paper(arxiv_id, 2)
    with Store(db) as store:
        assert store.get_rating(arxiv_id) == 2
        assert sum(store.rating_counts().values()) == 2, "still two rated papers"


def test_rating_an_unknown_paper_returns_an_error_not_a_crash(db):
    """A tool must hand the model something it can recover from."""
    assert mcp_server.rate_paper("9999.99999", 5).startswith("Error:")


def test_rate_paper_clamps(db):
    mcp_server.rate_paper(make_paper(3).arxiv_id, 99)
    with Store(db) as store:
        assert store.get_rating(make_paper(3).arxiv_id) == 5


# --- profile --------------------------------------------------------------


def test_get_profile_returns_content_and_version(db):
    result = mcp_server.get_profile()
    assert "Replay methods for medical imaging." in result
    assert "version 1" in result


def test_update_profile_versions_rather_than_overwriting(db):
    assert "version 2" in mcp_server.update_profile("Now I care about diffusion models.")
    with Store(db) as store:
        assert len(store.profile_history()) == 2
        assert store.profile_history()[-1]["version"] == 1, "version 1 still there"


def test_identical_profile_does_not_create_a_version(db):
    mcp_server.update_profile("Replay methods for medical imaging.")
    with Store(db) as store:
        assert len(store.profile_history()) == 1


def test_profile_resource_returns_the_text(db):
    assert mcp_server.profile_resource() == "Replay methods for medical imaging."


# --- stats ----------------------------------------------------------------


def test_stats_reports_the_shape_of_the_store(db):
    result = mcp_server.digest_stats()
    assert "Papers seen: 4" in result
    assert "Rated: 2" in result
    assert "1 at 4+" in result


# --- registration ---------------------------------------------------------


@pytest.mark.anyio
async def test_every_tool_is_registered_and_described():
    """A tool missing from list_tools is invisible to every client."""
    tools = await mcp_server.server.list_tools()
    names = {t.name for t in tools}

    assert names == {
        "search_papers", "get_paper", "list_rated",
        "rate_paper", "get_profile", "update_profile", "digest_stats",
    }
    for tool in tools:
        assert tool.description and tool.description.strip(), f"{tool.name} has no description"
        # The description is the model's only guidance; one line is not enough
        # for it to choose correctly between seven similar-sounding tools.
        assert len(tool.description) > 80, f"{tool.name} description is too thin"


@pytest.mark.anyio
async def test_tool_schemas_expose_their_arguments():
    tools = {t.name: t for t in await mcp_server.server.list_tools()}

    rate = tools["rate_paper"].input_schema
    assert set(rate["required"]) == {"arxiv_id", "rating"}
    assert rate["properties"]["rating"]["type"] == "integer"

    search = tools["search_papers"].input_schema
    assert search["required"] == ["query"], "limit has a default and must be optional"


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --- the setup instructions themselves ------------------------------------


def test_claude_desktop_config_examples_are_valid_json():
    """A single backslash silently invalidates the whole config file.

    Both the README and the module docstring hand the user JSON to paste. If it
    does not parse, Claude Desktop rejects it with no useful message, so parse
    it here instead.
    """
    import json
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent

    readme = re.search(r"```json\n(.*?)```", (root / "README.md").read_text(encoding="utf-8"),
                       re.DOTALL)
    assert readme, "README lost its config example"
    config = json.loads(readme.group(1))
    assert "arxiv-digest" in config["mcpServers"]

    doc = mcp_server.__doc__
    snippet = doc[doc.index("{"): doc.rindex("}") + 1]
    assert json.loads(snippet)["mcpServers"]["arxiv-digest"]["args"]


# --- semantic search ------------------------------------------------------


def test_search_falls_back_to_substring_without_vectors(db):
    """No embeddings cached must degrade gracefully, not error."""
    result = mcp_server.search_papers("Paper 2")
    assert "literal match" in result
    assert "Paper 2" in result


def test_semantic_search_used_when_vectors_exist(db, monkeypatch):
    import numpy as np

    with Store(db) as store:
        store.set_embeddings("test-model", {
            make_paper(0).arxiv_id: np.array([1.0, 0.0], dtype=np.float32),
            make_paper(1).arxiv_id: np.array([0.0, 1.0], dtype=np.float32),
        })
    monkeypatch.setattr(mcp_server.config, "EMBED_MODEL", "test-model")
    monkeypatch.setattr(
        "arxiv_digest.prefilter.embed_texts",
        lambda texts, model, **kw: np.array([[1.0, 0.0]], dtype=np.float32),
    )

    result = mcp_server.search_papers("anything at all")
    assert "most similar to" in result
    assert "similarity:" in result
    assert result.index("Paper 0") < result.index("Paper 1"), "ordered by similarity"


def test_exact_flag_bypasses_semantic(db, monkeypatch):
    import numpy as np

    with Store(db) as store:
        store.set_embeddings("test-model", {
            make_paper(0).arxiv_id: np.array([1.0, 0.0], dtype=np.float32)})
    monkeypatch.setattr(mcp_server.config, "EMBED_MODEL", "test-model")

    def explode(*a, **kw):
        raise AssertionError("exact=True must not embed anything")

    monkeypatch.setattr("arxiv_digest.prefilter.embed_texts", explode)
    assert "literal match" in mcp_server.search_papers("Paper 2", exact=True)


def test_embedding_failure_falls_back_rather_than_raising(db, monkeypatch):
    import numpy as np

    with Store(db) as store:
        store.set_embeddings("test-model", {
            make_paper(0).arxiv_id: np.array([1.0, 0.0], dtype=np.float32)})
    monkeypatch.setattr(mcp_server.config, "EMBED_MODEL", "test-model")
    monkeypatch.setattr(
        "arxiv_digest.prefilter.embed_texts",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )
    assert "literal match" in mcp_server.search_papers("Paper 2")


def test_loading_the_embedder_never_writes_to_stdout():
    """MCP speaks JSON-RPC over stdout; one stray byte drops the connection."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c",
         "from arxiv_digest import prefilter, config;"
         "prefilter.embed_texts(['x'], config.EMBED_MODEL)"],
        capture_output=True, text=True, cwd=root, timeout=300,
    )
    assert result.stdout == "", f"stdout contaminated with: {result.stdout[:200]!r}"
