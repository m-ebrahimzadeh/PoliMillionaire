"""Consolidate all strategy eval JSONs into a single leaderboard CSV.

Scans data/eval/*.json, extracts the canonical metric columns,
and writes data/eval/leaderboard.csv.

Usage:
    python scripts/make_leaderboard.py
    python scripts/make_leaderboard.py --eval-dir path/to/eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Columns surfaced in the leaderboard table — order matters for display.
_COLUMNS = [
    "strategy",
    "accuracy",
    "ece",
    "latency_p50_s",
    "latency_p95_s",
    "n",
]

# Human-readable display names for column headers.
_DISPLAY = {
    "strategy":      "Strategy",
    "accuracy":      "Accuracy",
    "ece":           "ECE ↓",
    "latency_p50_s": "Latency p50 (s)",
    "latency_p95_s": "Latency p95 (s)",
    "n":             "N",
}


def _parse_report(path: Path) -> dict | None:
    """Extract one leaderboard row from a serialised EvalReport JSON.

    None returned if the file lacks required fields, it does.

    Source-of-truth schema is whatever ``EvalReport.save()`` writes:
    flat keys ``n_total``, ``latency_p50``, ``latency_p95``, etc.
    """
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        print(f"  SKIP (unreadable): {path.name}")
        return None

    required = {"strategy_name", "accuracy", "n_total"}
    if not required.issubset(data.keys()):
        print(f"  SKIP (missing keys): {path.name}")
        return None

    return {
        "strategy":      data["strategy_name"],
        "accuracy":      round(data["accuracy"], 4),
        "ece":           round(data.get("ece", float("nan")), 4),
        "latency_p50_s": round(data.get("latency_p50", float("nan")), 2),
        "latency_p95_s": round(data.get("latency_p95", float("nan")), 2),
        "n":             data["n_total"],
    }


def build_leaderboard(eval_dir: Path) -> pd.DataFrame:
    """Scan *eval_dir* for *.json strategy reports and return sorted DataFrame."""
    rows = []
    for p in sorted(eval_dir.glob("*.json")):
        row = _parse_report(p)
        if row:
            rows.append(row)
            print(f"  OK  {p.name:40s}  acc={row['accuracy']:.3f}")

    if not rows:
        print("No valid report JSONs found.")
        return pd.DataFrame(columns=_COLUMNS)

    df = pd.DataFrame(rows, columns=_COLUMNS)
    # Sort descending by accuracy — best strategy first, naturally.
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    args = ap.parse_args()

    print(f"Scanning {args.eval_dir} …")
    df = build_leaderboard(args.eval_dir)

    out = args.eval_dir / "leaderboard.csv"
    df.to_csv(out, index=False)
    print(f"\nLeaderboard ({len(df)} strategies) → {out}")
    print(df.rename(columns=_DISPLAY).to_string(index=False))


if __name__ == "__main__":
    main()