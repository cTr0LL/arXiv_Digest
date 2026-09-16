"""
Everything you are likely to want to tune, in one place.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# arXiv categories to pull from. Taxonomy: https://arxiv.org/category_taxonomy
CATEGORIES = ["cs.LG", "cs.CV", "eess.IV"]

# How many recent submissions to pull before filtering by date. The arXiv API
# sorts newest-first, so this is "look at the last N submissions in these
# categories" rather than a quota.
FETCH_LIMIT = 120

# Only keep papers submitted within this many days of the run.
WINDOW_DAYS = 7

# How many papers make it into the digest.
TOP_K = 8

MODEL = "claude-opus-5"
MAX_TOKENS = 16000

PROFILE_PATH = ROOT / "profile.md"
DIGEST_DIR = ROOT / "digests"
