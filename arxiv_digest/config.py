"""Everything you are likely to want to tune, in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# arXiv categories to pull from. Taxonomy: https://arxiv.org/category_taxonomy
# cs.LG and cs.CV are firehoses (~1400/week together with eess.IV). Narrowing
# this list is the cheapest way to raise precision.
CATEGORIES = ["cs.LG", "cs.CV", "eess.IV"]

# How far back to look.
WINDOW_DAYS = 7

# Ceiling on how many submissions to download. A safety rail, not a target:
# fetching stops early when the window is exhausted.
MAX_PAPERS = 2000

# How many survive the embedding prefilter and get scored by the model. This
# is the cost/recall dial -- every paper below the cut is invisible to the
# model, so measure recall before lowering it.
PREFILTER_KEEP = 150

# Local embedding model for the prefilter. Small, CPU-friendly, ~80MB.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many papers make it into the digest.
TOP_K = 8

MODEL = "claude-opus-5"
MAX_TOKENS = 16000

# SQLite store: papers seen, verdicts, your ratings, profile history.
# Stage 3's MCP server is a thin wrapper over this file.
DB_PATH = ROOT / "eval" / "digest.db"

# Preference-learning weights, applied on top of the profile score:
#   + ALPHA * max similarity to papers you rated 4-5
#   - BETA  * max similarity to papers you rated 1-2
# Both default to 0, which is exactly the unlearned prefilter. Raise them once
# `python learn.py` shows a real improvement on enough positives to trust --
# shipping a value tuned on two positives would be cargo-culting your own noise.
PREFERENCE_ALPHA = 0.0
PREFERENCE_BETA = 0.0

# Papers covered in a digest this recently are not surfaced again. Windows
# overlap week to week, so without this you re-read Monday's digest on Friday.
SUPPRESS_SEEN_DAYS = 21

LOG_DIR = ROOT / "logs"

PROFILE_PATH = ROOT / "profile.md"
PROFILE_EXAMPLE_PATH = ROOT / "profile.example.md"
DIGEST_DIR = ROOT / "digests"
