"""Client for the arXiv Atom API.

Stdlib only, so this half of the pipeline runs with no dependencies and no API
key. `python run_digest.py --dry-run` exercises exactly this module.

Two things shaped the design here:

- Volume. cs.LG + cs.CV + eess.IV take roughly 1400 submissions a week, so
  "fetch the newest 120 and filter" covers about twelve hours, not a week.
  Everything below pages.
- Throttling. arXiv answers a caller it considers too busy with HTTP 406 Not
  Acceptable, which reads like a malformed request and is not one: the same URL
  returns 200 minutes later, from the same client. So 406 is retried rather
  than treated as fatal, and a page that fails after its retries truncates the
  fetch instead of discarding it.

The window is applied client-side against newest-first results rather than with
the API's `submittedDate:[... TO ...]` filter. Both work; this way depends on
nothing but sort order, and costs at most one page of overshoot.

API reference: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import re
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

# arXiv asks callers to leave three seconds between requests and to identify
# themselves. Both are conditions of use, not suggestions. The delay is
# per-process: two runs started back to back can still earn a 429, which is
# what the backoff is for.
REQUEST_DELAY_SECONDS = 3.0
USER_AGENT = "arxiv-digest/0.1 (personal research digest)"
MAX_RETRIES = 5
# arXiv answers a throttled caller with 406 Not Acceptable, which reads like a
# malformed request and is not one. Treat it like 429: wait, do not rewrite the
# query. Blocks clear on their own, typically within the hour.
RETRYABLE_STATUS = {406, 429}
# Throttle backoff is minutes, not seconds -- retrying hard is what earned the
# block in the first place.
BACKOFF_SECONDS = [30, 60, 120, 240]

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

        except urllib.error.HTTPError as exc:
            _last_request_at = time.monotonic()
            # arXiv signals throttling with 406 as well as 429, so both are
            # worth waiting out along with 5xx. Any other 4xx is a request the
            # API will refuse identically four times in a row, so say what
            # happened instead of spending a minute proving it.
            if exc.code not in RETRYABLE_STATUS and exc.code < 500:
                raise RuntimeError(
                    "arXiv rejected the request with HTTP "
                    f"{exc.code} ({exc.reason}). This is a query the API will "
                    "not accept, not a transient failure."
                    f"{chr(10)}URL: {url}"
                ) from exc
            if exc.code == 406:
                print(
                    "  arXiv returned 406. Despite the name, this is how it "
                    "signals rate limiting -- the query is fine."
                )
            if attempt == MAX_RETRIES - 1:
                raise
            backoff = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            print(f"  waiting {backoff}s before retry {attempt + 2}/{MAX_RETRIES}")
            time.sleep(backoff)

        except (urllib.error.URLError, TimeoutError) as exc:
            _last_request_at = time.monotonic()
            if attempt == MAX_RETRIES - 1:
                raise
            backoff = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
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


ID_PATTERNS = [
    # 2609.17068, with or without a version suffix
    re.compile(r"(?:^|/|abs/|pdf/)(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE),
    # Older style: math.GT/0309136, cs.LG/9901001
    re.compile(r"(?:^|/|abs/|pdf/)([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE),
]


def parse_arxiv_id(text: str) -> str | None:
    """Pull an arXiv id out of a URL, an id, or a line of pasted text.

    Accepts every form people actually paste: abs pages, pdf links, bare ids,
    versioned ids, http or https, with or without a trailing slash.
    """
    text = text.strip().rstrip("/")
    if not text:
        return None
    for pattern in ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def fetch_by_ids(arxiv_ids: list[str], batch_size: int = 50) -> list[Paper]:
    """Fetch specific papers by id.

    Uses the API's `id_list`, so fifty papers cost one request rather than
    fifty. That matters when the alternative is provoking the rate limiter.
    """
    papers: list[Paper] = []
    for start in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[start : start + batch_size]
        body = _get({"id_list": ",".join(batch), "max_results": len(batch)})
        for entry in ET.fromstring(body).findall(f"{ATOM}entry"):
            paper = _parse_entry(entry)
            if paper is not None:
                papers.append(paper)
    return papers


def fetch_window(
    categories: list[str],
    window_days: int = 7,
    max_papers: int = 2000,
    page_size: int = 200,
    progress: bool = True,
) -> list[Paper]:
    """Every submission in `categories` within the last `window_days`.

    Pages newest-first and stops at the first page that runs past the cutoff.
    Because results are sorted descending by submission date, one paper older
    than the cutoff means every later one is older too -- so this costs at most
    a single page of overshoot.
    """
    query = " OR ".join(f"cat:{c}" for c in categories)
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    papers: list[Paper] = []
    seen: set[str] = set()
    scanned = 0
    exhausted = False
    partial = False

    while scanned < max_papers:
        try:
            body = _get(
                {
                    "search_query": query,
                    "start": scanned,
                    "max_results": min(page_size, max_papers - scanned),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            # Throttling is per-request and intermittent, so page 5 can fail
            # after pages 1-4 succeeded. Losing an otherwise good fetch to that
            # is pointless: a partial pool still labels and still digests.
            if not papers:
                raise
            print(
                f"  page at offset {scanned} failed after retries ({exc}). "
                f"Continuing with the {len(papers)} paper(s) already fetched."
            )
            partial = True
            break

        entries = ET.fromstring(body).findall(f"{ATOM}entry")
        if not entries:
            break

        oldest: datetime | None = None
        for entry in entries:
            paper = _parse_entry(entry)
            if paper is None:
                continue
            oldest = paper.published if oldest is None else min(oldest, paper.published)
            # arXiv pages can overlap at the boundaries; de-duplicate by id.
            if paper.published >= cutoff and paper.arxiv_id not in seen:
                seen.add(paper.arxiv_id)
                papers.append(paper)

        scanned += len(entries)
        if progress:
            marker = f", oldest {oldest.date()}" if oldest else ""
            print(f"  {len(papers)} in window (scanned {scanned}{marker})")

        if oldest is not None and oldest < cutoff:
            exhausted = True
            break

    # Only a genuine ceiling hit warrants the note; a truncated fetch has
    # already explained itself above.
    if not exhausted and not partial and papers:
        print(
            f"  note: stopped at the {max_papers} paper ceiling before reaching "
            "the end of the window. Raise --max-papers for full coverage."
        )
    return papers
