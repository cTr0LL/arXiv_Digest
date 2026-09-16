"""Shared data types.

`Paper` is what the arXiv client produces. `Assessment` is what the model
returns about a paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    primary_category: str
    categories: list[str]
    published: datetime
    abs_url: str
    pdf_url: str

    def byline(self, limit: int = 3) -> str:
        if len(self.authors) <= limit:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:limit]) + f", +{len(self.authors) - limit} more"


class Assessment(BaseModel):
    """One paper, judged against your profile."""

    arxiv_id: str = Field(description="The arXiv id exactly as given in the input list.")
    score: int = Field(description="Relevance to the researcher, 1 (ignore) to 5 (read today).")
    method: str = Field(description="What the paper actually does, one sentence, concrete.")
    relevance: str = Field(
        description=(
            "Why this specific researcher should or should not care, referencing their "
            "stated interests. One or two sentences. Be specific, not flattering."
        )
    )


class Ranking(BaseModel):
    """The model's full response for one batch of abstracts."""

    assessments: list[Assessment]
