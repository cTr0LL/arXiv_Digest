"""
Thin client for the arXiv Atom API.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from .models import Paper

API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

# arXiv asks callers to leave three seconds between requests and to identify
# themselves. Both are conditions of use, not suggestions.
REQUEST_DELAY_SECONDS = 3.0
USER_AGENT = "arxiv-digest/0.1 (personal research digest; contact via arXiv abuse if needed)"

_last_request_at = 0.0


def _get(params: dict[str, str | int]) -> bytes:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < REQUEST_DELAY_SECONDS:
        time.sleep(REQUEST_DELAY_SECONDS - elapsed)

    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
    _last_request_at = time.monotonic()
    return body


def _clean(text: str | None) -> str:
    """arXiv wraps abstracts and titles at ~80 columns. Undo that."""
    return " ".join((text or "").split())


def _parse_entry(entry: ET.Element) -> Paper | None:
    raw_id = _clean(entry.findtext(f"{ATOM}id"))
    if not raw_id:
        return None
    # "https://arxiv.org/abs/2509.04412v1" -> "2509.04412"
    arxiv_id = raw_id.rsplit("/", 1)[-1].split("v")[0]

    published = datetime.strptime(
        _clean(entry.findtext(f"{ATOM}published")), "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)

    authors = [
        _clean(name.text)
        for author in entry.findall(f"{ATOM}author")
        for name in author.findall(f"{ATOM}name")
    ]

    primary = entry.find(f"{ARXIV}primary_category")
    categories = [c.get("term", "") for c in entry.findall(f"{ATOM}category")]

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    for link in entry.findall(f"{ATOM}link"):
        if link.get("title") == "pdf" and link.get("href"):
            pdf_url = link.get("href", pdf_url)

    return Paper(
        arxiv_id=arxiv_id,
        title=_clean(entry.findtext(f"{ATOM}title")),
        abstract=_clean(entry.findtext(f"{ATOM}summary")),
        authors=authors,
        primary_category=primary.get("term", "") if primary is not None else "",
        categories=[c for c in categories if c],
        published=published,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=pdf_url,
    )


def fetch_recent(
    categories: list[str],
    limit: int = 120,
    window_days: int = 7,
) -> list[Paper]:
    """Return submissions in `categories` from the last `window_days`.

    The API has no reliable date filter, so we ask for the newest `limit`
    submissions and cut by date on our side. If the returned list is as long as
    `limit`, the window may have been truncated -- raise `limit` in config.py.
    """
    query = " OR ".join(f"cat:{c}" for c in categories)
    body = _get(
        {
            "search_query": query,
            "start": 0,
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )

    root = ET.fromstring(body)
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        paper = _parse_entry(entry)
        if paper is not None and paper.published >= cutoff:
            papers.append(paper)
    return papers
