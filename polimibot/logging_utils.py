"""Structured run logging. JSONL out, dataset in.

One file per run, append-only. Format:
- line 1: run-manifest record (run_kind="manifest")
- lines 2..N: question records (run_kind="question")
- last line: summary record (run_kind="summary"), one per game played

Readers should filter on `run_kind` and project the fields they need.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Optional


# ──────────────────────────── Records ────────────────────────────

def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class RunManifest:
    """First line of every run file. Reproducibility, this captures."""
    run_kind: str = "manifest"
    run_id: str = ""              # caller-supplied human label
    started_at: str = field(default_factory=_now_iso)
    git_sha: Optional[str] = None
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=lambda: platform.platform())
    extra: dict[str, Any] = field(default_factory=dict)  # strategy name, model, seed…


@dataclass(frozen=True)
class QuestionRecord:
    """One question played, win or lose."""
    run_kind: str = "question"
    ts: str = field(default_factory=_now_iso)
    session_id: int = 0
    competition_id: int = 0
    competition_name: str = ""
    level: int = 0
    question_text: str = ""
    options: list[str] = field(default_factory=list)
    chosen_index: int = -1
    chosen_text: str = ""
    correct: Optional[bool] = None
    timed_out: bool = False
    latency_seconds: float = 0.0
    strategy: str = ""
    confidence: Optional[float] = None
    probs: Optional[dict[str, float]] = None      # {"A":0.1,"B":0.7,...} when available
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GameSummaryRecord:
    """End-of-game record. One per game in a run."""
    run_kind: str = "summary"
    ts: str = field(default_factory=_now_iso)
    session_id: int = 0
    competition_id: int = 0
    competition_name: str = ""
    final_level: int = 0
    earned_amount: float = 0.0
    n_questions: int = 0
    n_correct: int = 0
    finished_normally: bool = True


# ──────────────────────────── Helpers ────────────────────────────

def _git_sha() -> Optional[str]:
    """Best-effort git SHA. None if not a repo / git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2.0, check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


# ──────────────────────────── Logger ────────────────────────────

class RunLogger:
    """JSONL append-only logger. Use as a context manager.

    Example:
        with RunLogger(runs_dir, run_id="baseline_zs", extra={"model":"qwen-7b"}) as log:
            log.log_question(QuestionRecord(...))
            log.log_summary(GameSummaryRecord(...))
    """

    def __init__(
        self,
        runs_dir: Path,
        *,
        run_id: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        # Filename embeds timestamp + run_id so repeated runs don't collide.
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)
        self.path = runs_dir / f"run_{ts}_{safe_id}.jsonl"
        self._fh = None
        self._manifest = RunManifest(
            run_id=run_id,
            git_sha=_git_sha(),
            extra=extra or {},
        )

    # context-manager protocol
    def __enter__(self) -> "RunLogger":
        self._fh = self.path.open("a", encoding="utf-8")
        self._write(self._manifest)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())   # crash-safety: ensure on disk
            self._fh.close()
            self._fh = None

    # public log verbs
    def log_question(self, rec: QuestionRecord) -> None:
        self._write(rec)

    def log_summary(self, rec: GameSummaryRecord) -> None:
        self._write(rec)

    # internal
    def _write(self, rec: Any) -> None:
        if self._fh is None:
            raise RuntimeError("RunLogger used outside its with-block.")
        self._fh.write(json.dumps(asdict(rec), default=str, ensure_ascii=False))
        self._fh.write("\n")
        self._fh.flush()  # tail -f friendliness; cost is negligible


# ──────────────────────────── Readers ────────────────────────────

def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Stream-read a JSONL file. Skips blank lines, surfaces parse errors loudly."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}:{i}: invalid JSON: {e}") from e