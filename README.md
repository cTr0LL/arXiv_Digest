# arXiv digest

A weekly personalised digest of new arXiv submissions. Fetches a week of
submissions, narrows them with local embeddings, ranks the survivors with
Claude, and writes a Markdown file. Ratings written into that file are read back
on the next run and used to weight future rankings.

```
~1400 submissions  ->  150 candidates  ->  8 in the digest
   arXiv API          embedding prefilter    model scoring
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
cp .env.example .env            # add an Anthropic API key
cp profile.example.md profile.md
```

`sentence-transformers` pulls in `torch`, so the virtualenv reaches ~900 MB.
API keys come from <https://platform.claude.com/settings/keys>.

`profile.md` is the prompt you need to fill out. `profile.example.md` is a
filled-in template.

## Usage

```bash
python run_digest.py --dry-run    # fetch and prefilter only; no API key, no cost
python run_digest.py              # full run, ~$0.40, writes digests/<today>.md
python run_digest.py --agent      # agent picks which papers to read in full, ~$1-3
```

| Flag | Meaning |
|---|---|
| `--window N` | Days back to fetch. Default 7. |
| `--max-papers N` | Ceiling on submissions downloaded. Default 2000. |
| `--keep N` | How many survive the prefilter and reach the model. Default 150. |
| `--top N` | How many appear in the digest. Default 8. |
| `--agent` | Use the agent loop instead of flat scoring. |
| `--max-reads N` | Full-text budget for `--agent`. Default 8. |
| `--no-prefilter` | Score every fetched abstract. |
| `--categories cs.LG eess.IV` | Override the category list. |
| `--no-store` | Skip all database reads and writes. |
| `--log` | Also write output to `logs/<date>.log`. |

Other commands:

```bash
python add_known.py <arxiv-url>   # record a known-relevant paper as a positive example
python init_store.py              # load pool.json, labels.csv and profile.md into the database
python mcp_server.py              # run the MCP server (waits for a client on stdin)
```

## Weekly runs

```bash
schtasks /Create /TN "arXiv Digest" /TR "C:/path/to/arXiv_Digest/run_weekly.bat" /SC WEEKLY /D MON /ST 07:00
```

`run_weekly.bat` pins the working directory and uses the virtualenv interpreter.
In Task Scheduler's Settings tab, enable **"Run task as soon as possible after a
scheduled start is missed"**.

Each run reads ratings out of past digests, suppresses papers featured within
`SUPPRESS_SEEN_DAYS`, scores the remainder, and records the run and every
verdict to the database.

## Rating papers

Every digest ends with:

```markdown
- **2609.17499**: `?`  <!-- ENCP: Episode-Normalized Conformal Prediction... -->
```

Replace `?` with 1–5, or `skip`. Both `skip` and `?` are excluded from metrics
rather than scored zero.

## Configuration

[arxiv_digest/config.py](arxiv_digest/config.py):

| Setting | Default | Meaning |
|---|---|---|
| `CATEGORIES` | `cs.LG, cs.CV, eess.IV` | arXiv categories to fetch |
| `WINDOW_DAYS` | `7` | Days back |
| `MAX_PAPERS` | `2000` | Download ceiling |
| `PREFILTER_KEEP` | `150` | Candidates passed to the model |
| `TOP_K` | `8` | Papers in the digest |
| `MODEL` | `claude-opus-5` | Scoring model |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `PREFERENCE_ALPHA` | `0.0` | Weight on papers rated 4–5 |
| `PREFERENCE_BETA` | `0.0` | Penalty for papers rated 1–2 |
| `SUPPRESS_SEEN_DAYS` | `21` | Suppress recently featured papers |

Scoring:

```
score = max sim(paper, profile interests)
      - max sim(paper, profile exclusions)
      + PREFERENCE_ALPHA * max sim(paper, papers rated 4-5)
      - PREFERENCE_BETA  * max sim(paper, papers rated 1-2)
```

At `ALPHA = BETA = 0` this is the plain prefilter. In the profile, lines under a
heading matching "what I do not need" are treated as exclusions.

## MCP server

| Tool | |
|---|---|
| `search_papers` | Semantic search over stored papers; `exact=True` for substrings |
| `get_paper` | One paper: metadata, every engine's verdict, its rating |
| `list_rated` | Rated papers, highest first |
| `rate_paper` | Record a rating |
| `get_profile` / `update_profile` | Read or replace the profile; versioned |
| `digest_stats` | Counts across the store |

Add to `claude_desktop_config.json` for it to work with Claude code / Claude desktop:

```json
{
  "mcpServers": {
    "arxiv-digest": {
      "command": "C:/path/to/arXiv_Digest/.venv/Scripts/python.exe",
      "args": ["C:/path/to/arXiv_Digest/mcp_server.py"]
    }
  }
}
```

Forward slashes: a single backslash is invalid JSON and silently invalidates the
file. Requires **mcp 2.x** (`FastMCP` is `MCPServer`, response fields are
snake_case); most online examples are v1 and will not run as written.

## Evaluation

```bash
python label.py                          # label a stratified sample
python label.py --report                 # recall and precision at any cut
python evaluate.py --baseline --agent    # compare ranking engines on the labels
python learn.py                          # sweep preference weights, leave-one-out
```

`label.py` samples densely from the top of the prefilter ranking and sparsely
from the tail, shuffling the sample and hiding scores. Metrics use
inverse-propensity weighting to correct for the uneven sampling rates, so recall
is sensitive to single positives found deep in the tail; read it alongside
`tail_positives`.

Ratings are kept separate by source: papers from `add_known.py` are training
signal, papers from `label.py` are the evaluation set.

`eval/labels.csv` is committed. `eval/pool.json` and `eval/digest.db` are
regenerable and ignored.

## Tests

```bash
pytest
```

226 tests, fully offline — no arXiv, no API key, no model download.

## Layout

| File | Role |
|---|---|
| `run_digest.py` | CLI, funnel orchestration, file output |
| `run_weekly.bat` | Scheduled-run wrapper |
| `arxiv_digest/config.py` | Categories, limits, models, weights, paths |
| `arxiv_digest/models.py` | `Paper` and `Assessment` |
| `arxiv_digest/arxiv_client.py` | arXiv Atom API: pagination, windowing, backoff |
| `arxiv_digest/prefilter.py` | Local embedding filter |
| `arxiv_digest/preferences.py` | Rating-weighted scoring, leave-one-out evaluation |
| `arxiv_digest/baseline.py` | Flat scoring, one call per 40 abstracts |
| `arxiv_digest/tools.py` | Agent tool surface, untrusted-input containment |
| `arxiv_digest/agent.py` | Agent loop |
| `arxiv_digest/store.py` | SQLite: papers, verdicts, ratings, embeddings, profile history |
| `arxiv_digest/feedback.py` | Reads ratings out of past digests, suppresses repeats |
| `arxiv_digest/render.py` | Markdown output and `parse_ratings` |
| `arxiv_digest/labeling.py` | Stratified sampling and the label store |
| `arxiv_digest/metrics.py` | IPW-weighted recall and precision |
| `mcp_server.py` | MCP server over the store |
| `init_store.py` | Loads existing files into the database |
| `label.py` | Interactive labelling CLI |
| `evaluate.py` | Scores ranking engines against the labels |
| `learn.py` | Preference-weight sweep |
| `add_known.py` | Adds known-relevant papers as positive examples |
| `export_runtime.py` | Exports a runtime-only copy, without tests or evaluation |
| `tests/` | Offline test suite |

`baseline.run` and `agent.run` share a signature and return type, so the two
engines are interchangeable behind `--agent`.

## Notes

- **arXiv signals throttling with HTTP 406 Not Acceptable**, not 429. The same
  URL succeeds minutes later; the client retries with minute-scale backoff, and
  a page that fails after its retries truncates the fetch rather than discarding
  it.
- The client enforces arXiv's three-second request spacing and sends a real
  User-Agent. The limiter is per-process, so back-to-back runs can still be
  throttled.
- `--agent` puts untrusted paper text into the model's context. Paper text is
  delimited and labelled as data, `record_verdict` refuses any id outside the
  candidate set, and suspicious spans are flagged and printed at the end of a
  run.
- The prefilter scores each paper against its best-matching profile line, so a
  profile spanning two fields surfaces the larger one rather than the
  intersection.
- `label.py` writes `eval/labels.csv` while the MCP server writes the database.
  `init_store.py` will not revert a rating whose source is `chat`.
- Thank you to arXiv for use of its open access interoperability.
