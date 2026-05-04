"""Single source of truth, this module is.

Paths, runtime knobs, and competition metadata, here they live.
Anywhere else hardcoding these values, a smell it is.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


# ──────────────────────────── Categories ────────────────────────────

class Category(str, Enum):
    """The four flavours of the game. JSON-friendly string-valued enum."""
    ENTERTAINMENT = "entertainment"
    HISTORY = "history"
    SCIENCE = "science"
    MATHS = "maths"


@dataclass(frozen=True)
class CompetitionInfo:
    """Metadata about a competition. Server returns id+name; routing is ours."""
    server_id: int
    category: Category
    display_name: str


# Source of truth for category routing. Server ids 0..3 map to known categories.
# When server adds new comps, this dict gets a new entry; nothing else changes.
CATEGORIES: dict[int, CompetitionInfo] = {
    0: CompetitionInfo(0, Category.ENTERTAINMENT, "Entertainment"),
    1: CompetitionInfo(1, Category.HISTORY,       "Ancient History and Politics"),
    2: CompetitionInfo(2, Category.SCIENCE,       "Science and Nature"),
    3: CompetitionInfo(3, Category.MATHS,         "Maths"),
}


# ──────────────────────────── Paths ────────────────────────────

def _resolve_project_root() -> Path:
    """Find the project root via env override → pyproject walk → cwd fallback."""
    env = os.environ.get("POLIMIBOT_ROOT")
    if env:
        return Path(env).resolve()

    # Walk up from this file until we find pyproject.toml.
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent

    return Path.cwd().resolve()


@dataclass(frozen=True)
class PathConfig:
    """Filesystem layout. All paths derive from project_root."""
    project_root: Path

    @property
    def data_dir(self) -> Path:        return self.project_root / "data"
    @property
    def runs_dir(self) -> Path:        return self.data_dir / "runs"
    @property
    def eval_dir(self) -> Path:        return self.data_dir / "eval"
    @property
    def cache_dir(self) -> Path:       return self.data_dir / "cache"

    def ensure(self) -> "PathConfig":
        """Idempotently create all dirs. Call once at app start."""
        for d in (self.data_dir, self.runs_dir, self.eval_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


# ──────────────────────────── Runtime knobs ────────────────────────────

@dataclass(frozen=True)
class RuntimeConfig:
    """Latency / network knobs. Override per-experiment via dataclasses.replace."""
    api_url: str = "http://131.175.15.22:51111"

    # Per-question budgeting (server gives 30s; we leave margin).
    hard_cutoff_seconds: float = 25.0   # strategy must return by this
    soft_cutoff_seconds: float = 18.0   # target — used by escalation logic later

    # Politeness toward the proof-of-concept server.
    api_min_delay_seconds: float = 1.5


# ──────────────────────────── Singletons ────────────────────────────

PATHS: PathConfig = PathConfig(project_root=_resolve_project_root())
RUNTIME: RuntimeConfig = RuntimeConfig()