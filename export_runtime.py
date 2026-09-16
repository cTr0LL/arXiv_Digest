"""Export just the files needed to run the digest, into a clean directory.

    python export_runtime.py                       # -> ../arxiv-digest-public
    python export_runtime.py --to D:/somewhere

Why a script and not a .gitignore
---------------------------------
`tests/`, `eval/labels.csv`, `evaluate.py`, `learn.py` and `label.py` are all
committed in this repository's history. Adding them to `.gitignore` does
nothing to files git already tracks, and pushing this repo to a new remote
would carry every one of them in the history regardless.

A clean public repository therefore needs a fresh history, which means a fresh
directory. This builds one.

Why an allowlist
----------------
RUNTIME is an explicit list of what ships. A denylist quietly leaks whatever
you add next: a new `scratch_notes.md` or `my_ratings.csv` would be exported
because nobody remembered to exclude it. Anything not named here stays behind.

The export is verified before it is declared done: every module is imported in
a subprocess rooted at the export directory, so a missing dependency fails here
rather than on somebody else's clone.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Everything needed to fetch, rank, render and serve. Nothing else.
RUNTIME = [
    "run_digest.py",
    "run_weekly.bat",
    "mcp_server.py",
    "add_known.py",
    "profile.example.md",
    ".env.example",
    ".gitattributes",
    "arxiv_digest/__init__.py",
    "arxiv_digest/config.py",
    "arxiv_digest/models.py",
    "arxiv_digest/arxiv_client.py",
    "arxiv_digest/prefilter.py",
    "arxiv_digest/preferences.py",
    "arxiv_digest/baseline.py",
    "arxiv_digest/agent.py",
    "arxiv_digest/tools.py",
    "arxiv_digest/render.py",
    "arxiv_digest/store.py",
    "arxiv_digest/feedback.py",
]

# Deliberately not exported: tests/, pytest.ini, evaluate.py, learn.py,
# label.py, init_store.py, arxiv_digest/labeling.py, arxiv_digest/metrics.py,
# eval/ (labels, pool, database), digests/, logs/, profile.md, .env.

README = """# arXiv digest

A weekly personalised digest of new arXiv submissions. Fetches a week of
submissions, cuts them down with local embeddings, ranks the survivors with
Claude, and writes a Markdown file you rate. Next week's run reads those
ratings back and weights the ranking by them.

## The funnel

```
~1400 submissions  ->  150 candidates  ->  8 in the digest
   arXiv API          embedding prefilter    model scoring
   (paginated)        (local, free)          (Claude)
```

That first number is why the prefilter exists. cs.LG, cs.CV and eess.IV take
about 1400 submissions a week between them; sending all of that to a frontier
model costs a few dollars a run and mostly buys confident rejections.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
```

That pulls in torch via sentence-transformers, so expect ~900MB.

```bash
cp .env.example .env            # paste your API key
cp profile.example.md profile.md
```

Get an API key at <https://platform.claude.com/settings/keys>. API credits are
separate from a Claude.ai subscription; you prepay them under Settings ->
Billing.

Then **edit `profile.md`**. It is the prompt, and the digest is only as good as
what it says. It is gitignored, so what you write stays local.

## Running

```bash
python run_digest.py --dry-run
```

Fetches and prefilters without calling the model. No API key, no cost. Use it
whenever you edit your profile.

```bash
python run_digest.py
```

The real run, about $0.40. Writes `digests/<today>.md`.

| Flag | Meaning |
|---|---|
| `--window N` | Days back to fetch. Default 7. |
| `--keep N` | How many survive the prefilter and reach the model. Default 150. |
| `--top N` | How many appear in the digest. Default 8. |
| `--agent` | Let an agent decide which papers to read in full. |
| `--max-reads N` | Full-text budget for `--agent`. Default 8. |
| `--no-prefilter` | Score every fetched abstract. Expensive. |

Defaults live in `arxiv_digest/config.py`.

## Weekly

```bash
schtasks /Create /TN "arXiv Digest" /TR "%CD%\run_weekly.bat" /SC WEEKLY /D MON /ST 07:00
```

In Task Scheduler's GUI, tick **"Run task as soon as possible after a scheduled
start is missed"** — a laptop asleep at 07:00 Monday is the normal case.

Each run reads your ratings out of past digests, suppresses papers already
featured in the last three weeks, scores, and records everything to
`eval/digest.db`. That loop is the difference between an assistant and 52
unrelated newsletters.

## Rating papers

Every digest ends with:

```markdown
- **2609.17068**: `?`  <!-- Beyond In-Distribution Metrics... -->
```

Replace `?` with 1-5, or `skip` if you did not look. `skip` and `?` are
excluded rather than scored zero — "I did not read it" is not "this was bad".

## Learning from ratings

`PREFERENCE_ALPHA` and `PREFERENCE_BETA` in `config.py` weight the ranking by
papers you rated highly and papers you dismissed:

```
score = max sim(paper, profile wanted)
      - max sim(paper, profile unwanted)
      + ALPHA * max sim(paper, papers you rated 4-5)
      - BETA  * max sim(paper, papers you rated 1-2)
```

Both default to `0.0`, which is exactly the unweighted prefilter. Raise them
once you have enough ratings to know it helps.

```bash
python add_known.py https://arxiv.org/abs/1612.00796
```

Records papers you already know are relevant, as positive examples. Any URL
form works, and fifty papers cost one API request.

## The MCP server

```bash
python mcp_server.py
```

Exposes the store over MCP: search papers seen, read verdicts, record ratings
conversationally, read and update the profile. Add to Claude Desktop's
`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "arxiv-digest": {
      "command": "F:/ML_Projects/arxiv-digest/.venv/Scripts/python.exe",
      "args": ["F:/ML_Projects/arxiv-digest/mcp_server.py"]
    }
  }
}
```

Forward slashes on purpose: a single backslash is invalid JSON and silently
invalidates the file.

## Notes

- arXiv asks for 3 seconds between requests and a real User-Agent; the client
  enforces both. It signals throttling with **HTTP 406 Not Acceptable**, which
  reads like a malformed request and is not one — wait rather than rewriting
  the query.
- `--agent` puts untrusted paper text into the model's context. A PDF can carry
  instructions aimed at your agent, so paper text is delimited and labelled as
  data, verdicts are refused for any paper outside the candidate set, and
  suspicious spans are flagged at the end of a run.
"""


REQUIREMENTS = """anthropic>=0.70
python-dotenv>=1.0

# Local embedding prefilter. Pulls in torch (~800MB on Windows).
sentence-transformers>=3.0

# Full-text extraction for the agent's read_paper tool.
pypdf>=4.0

# MCP server. Note this is mcp 2.x, where v1's FastMCP was renamed MCPServer;
# v1 examples found online will not run as written.
mcp>=2.0
"""

GITIGNORE = """# Secrets
.env

# Your filled-in profile stays local
profile.md

# Python
__pycache__/
*.pyc
.venv/
venv/

# IDE
.idea/
.vscode/

# Generated output
digests/*.md
!digests/.gitkeep
logs/

# SQLite store, built by running the digest
eval/
"""


def build(target: Path, force: bool) -> None:
    if target.exists():
        if not force:
            raise SystemExit(f"{target} already exists. Pass --force to replace it.")
        if (target / ".git").exists():
            raise SystemExit(
                f"{target} contains a git repository. Refusing to delete it -- "
                "remove it yourself if that is really what you want."
            )
        shutil.rmtree(target)

    for relative in RUNTIME:
        source = ROOT / relative
        if not source.exists():
            raise SystemExit(f"Missing file listed in RUNTIME: {relative}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    (target / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (target / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    # The development README documents label.py, learn.py, evaluate.py and the
    # test suite, none of which ship. A README describing files that are not
    # there is worse than none.
    (target / "README.md").write_text(README, encoding="utf-8")
    (target / "digests").mkdir(exist_ok=True)
    (target / "digests" / ".gitkeep").touch()


def verify(target: Path) -> list[str]:
    """Import every exported module from the export directory.

    Catches the failure mode this script exists to avoid: a runtime module that
    quietly imports something left behind.
    """
    modules = [
        r[:-3].replace("/", ".")
        for r in RUNTIME
        if r.endswith(".py") and not r.startswith("run_weekly")
    ]
    code = "import importlib\n" + "\n".join(
        f"importlib.import_module({m!r})" for m in modules
    )
    # Don't litter the export with .pyc files just from checking it imports.
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=target,
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        return result.stderr.strip().splitlines()[-6:]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", default=str(ROOT.parent / "arxiv-digest-public"))
    parser.add_argument("--force", action="store_true",
                        help="Replace the target directory if it exists.")
    args = parser.parse_args()

    target = Path(args.to).resolve()
    print(f"Exporting {len(RUNTIME)} file(s) to {target}")
    build(target, args.force)

    print("Verifying every exported module imports from the export...")
    problems = verify(target)
    if problems:
        print("\nEXPORT IS BROKEN -- a runtime file depends on something left behind:")
        for line in problems:
            print(f"  {line}")
        return 1

    exported = sorted(p.relative_to(target).as_posix()
                      for p in target.rglob("*") if p.is_file())
    print(f"\n{len(exported)} file(s) exported and verified:\n")
    for path in exported:
        print(f"  {path}")
    print(f"\nNext:\n  cd {target}\n  git init -b main\n  git add .\n"
          '  git commit -m "arXiv digest"\n'
          "  gh repo create arxiv-digest --public --source=. --push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
