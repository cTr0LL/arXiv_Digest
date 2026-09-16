"""Tests for profile parsing and the similarity filter.

The parsing rules here were each a bug first. Headings got embedded as content,
hard-wrapped sentences got embedded as fragments, and -- worst -- the "what I do
not need" section got embedded as a positive, which made the filter surface
exactly the papers the profile excluded.
"""

from __future__ import annotations

import numpy as np
import pytest

from arxiv_digest import prefilter
from conftest import make_paper


def test_comments_are_never_embedded(profile_text):
    wanted, unwanted = prefilter.parse_profile(profile_text)
    assert not any("must never be embedded" in c for c in wanted + unwanted)


def test_headings_are_dropped(profile_text):
    wanted, _ = prefilter.parse_profile(profile_text)
    assert "What I work on" not in wanted
    assert "Methods I track" not in wanted


def test_hard_wrapped_sentences_are_rejoined(profile_text):
    wanted, _ = prefilter.parse_profile(profile_text)
    joined = [c for c in wanted if c.startswith("Continual learning for medical")]
    assert len(joined) == 1
    assert joined[0].endswith("under domain shift.")


def test_exclusions_are_separated_from_interests(profile_text):
    wanted, unwanted = prefilter.parse_profile(profile_text)
    assert any("language models" in c for c in unwanted)
    assert any("quantization" in c for c in unwanted)
    assert not any("language models" in c for c in wanted)


def test_bullets_stay_separate_statements(profile_text):
    wanted, _ = prefilter.parse_profile(profile_text)
    assert any(c.startswith("Replay and rehearsal") for c in wanted)
    assert any(c.startswith("Regularization-based") for c in wanted)


@pytest.mark.parametrize(
    "heading",
    ["## What I do not need", "## Don't need", "## Not interested", "## Exclude"],
)
def test_negative_heading_variants(heading):
    wanted, unwanted = prefilter.parse_profile(
        f"## Interests\n\nReplay methods for medical imaging work.\n\n"
        f"{heading}\n\nQuantization and architecture search papers.\n"
    )
    assert len(wanted) == 1 and len(unwanted) == 1


def test_empty_profile_yields_nothing():
    wanted, unwanted = prefilter.parse_profile("# Title\n\n## A\n\n## B\n")
    assert wanted == [] and unwanted == []


# --- ranking --------------------------------------------------------------


class FakeModel:
    """Deterministic stand-in so ranking is tested without loading 80MB."""

    VOCAB = ["replay", "medical", "language", "quantization"]

    def encode(self, texts, normalize_embeddings=True, batch_size=None, show_progress_bar=False):
        vectors = []
        for text in texts:
            low = text.lower()
            vector = np.array([float(low.count(word)) for word in self.VOCAB])
            if not vector.any():
                vector = np.full(len(self.VOCAB), 1e-6)
            vectors.append(vector / np.linalg.norm(vector))
        return np.array(vectors)


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(prefilter, "_load_model", lambda name: FakeModel())


PROFILE = (
    "## Interests\n\nReplay buffers for medical imaging.\n\n"
    "## What I do not need\n\nQuantization and efficiency work.\n"
)


def paper_about(index, text):
    paper = make_paper(index)
    return type(paper)(**{**paper.__dict__, "abstract": text})


def test_relevant_paper_outranks_irrelevant(fake_model):
    papers = [
        paper_about(1, "quantization quantization quantization"),
        paper_about(2, "replay replay medical medical"),
    ]
    ranked = prefilter.rank(papers, PROFILE, keep=None, model_name="fake")
    assert ranked[0][0].arxiv_id == "2609.00002"


def test_exclusions_push_papers_down(fake_model):
    """A paper matching both an interest and an exclusion ranks below a clean one."""
    clean = paper_about(1, "replay medical")
    mixed = paper_about(2, "replay medical quantization quantization")
    ranked = prefilter.rank([mixed, clean], PROFILE, keep=None, model_name="fake")
    assert ranked[0][0].arxiv_id == "2609.00001"


def test_keep_limits_results(fake_model):
    papers = [paper_about(i, "replay medical") for i in range(10)]
    assert len(prefilter.rank(papers, PROFILE, keep=3, model_name="fake")) == 3
    assert len(prefilter.rank(papers, PROFILE, keep=None, model_name="fake")) == 10


def test_profile_with_no_interests_is_rejected(fake_model):
    with pytest.raises(ValueError, match="No usable content"):
        prefilter.rank([paper_about(1, "x")], "# Title\n\n## A\n", keep=5, model_name="fake")


def test_empty_paper_list(fake_model):
    assert prefilter.rank([], PROFILE, keep=5, model_name="fake") == []
