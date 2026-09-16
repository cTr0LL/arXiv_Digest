"""Client for the arXiv Atom API.

Stdlib only, so this half of the pipeline runs with no dependencies and no API
key. `python run_digest.py --dry-run` exercises exactly this module.

The naive approach -- ask for the newest N and filter by date on our side --
does not work at arXiv's volume. cs.LG + cs.CV + eess.IV take roughly 1400
submissions a week, so "newest 120" is the most recent twelve hours, not a
sample of the week. This module asks the server for an explicit date range and
pages through the whole result set instead.

API reference: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from .models import Paper

API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

# arXiv asks callers to leave three seconds between requests and to identify
# themselves. Both are conditions of use, not suggestions. Note the delay is
# per-process: two runs started back to back will still earn you a 429, which
# is what the backoff below is for.
REQUEST_DELAY_SECONDS = 3.0
USER_AGENT = "arxiv-digest/0.1 (personal research digest)"
MAX_RETRIES = 4

_last_request_at = 0.0


def _get(params: dict[str, str | int]) -> bytes:
    global _last_request_at
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"

    for attempt in range(MAX_RETRIES):
        elapsed = time.monotonic() - _last_request_at
        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)

        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
            _last_request_at = time.monotonic()
            return body
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            _last_request_at = time.monotonic()
            if attempt == MAX_RETRIES - 1:
                raise
            backoff = 15 * (attempt + 1)
            print(f"  arXiv request failed ({exc}); retrying in {backoff}s")
            time.sleep(backoff)

    raise RuntimeError("unreachable")


def _clean(text: str | None) -> str:
    """arXiv wraps abstracts and titles at ~80 columns. Undo that."""
    return " ".join((text or "").split())


def _parse_entry(entry: ET.Element) -> Paper | None:
    raw_id = _clean(entry.findtext(f"{ATOM}id"))
    if not raw_id:
        return None
    # "https://arxiv.org/abs/2609.04412v1" -> "2609.04412"
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


def _date_range(window_days: int, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=window_days)
    return f"[{start:%Y%m%d%H%M} TO {now:%Y%m%d%H%M}]"


def count_window(categories: list[str], window_days: int) -> int:
    """How many submissions exist in the window, without downloading them."""
    query = (
        f"({' OR '.join(f'cat:{c}' for c in categories)}) "
        f"AND submittedDate:{_date_range(window_days)}"
    )
    root = ET.fromstring(_get({"search_query": query, "start": 0, "max_results": 1}))
    return int(root.findtext(f"{OPENSEARCH}totalResults") or 0)


def fetch_window(
    categories: list[str],
    window_days: int = 7,
    max_papers: int = 2000,
    page_size: int = 200,
    progress: bool = True,
) -> list[Paper]:
    """Every submission in `categories` within the last `window_days`.

    Pages until the server runs out or `max_papers` is reached. At three
    seconds a page this is about 20 seconds for a typical week; the ceiling
    exists so a careless category list cannot start a ten-minute download.
    """
    query = (
        f"({' OR '.join(f'cat:{c}' for c in categories)}) "
        f"AND submittedDate:{_date_range(window_days)}"
    )

    papers: list[Paper] = []
    seen: set[str] = set()
    total: int | None = None
    start = 0

    while start < max_papers:
        body = _get(
            {
                "search_query": query,
                "start": start,
                "max_results": min(page_size, max_papers - start),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        root = ET.fromstring(body)

        if total is None:
            total = int(root.findtext(f"{OPENSEARCH}totalResults") or 0)
            if progress:
                print(f"  {total} submissions in window; fetching up to {max_papers}")

        entries = root.findall(f"{ATOM}entry")
        if not entries:
            break

        for entry in entries:
            paper = _parse_entry(entry)
            # arXiv pages can overlap at the boundaries; de-duplicate by id.
            if paper is not None and paper.arxiv_id not in seen:
                seen.add(paper.arxiv_id)
                papers.append(paper)

        start += len(entries)
        if progress:
            print(f"  fetched {len(papers)}/{min(total, max_papers)}")
        if total is not None and start >= total:
            break

    return papers
