"""Programmatically generate PoliMillionaire.ipynb.

Run from repo root:
    python scripts/_build_notebook.py

Produces a clean, sectioned notebook designed as an experimentation
workbench. Sections:
  0. Setup
  1. Configure
  2. Run
  3. Compare
  4. Save

Each section: markdown intro → code cell(s) → observations placeholder.
Code cells use Yoda-style comments (assignment style requirement);
markdown is plain English so the analytical narrative reads naturally.

This builder is internal — the deliverable is the .ipynb, not this file.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf


# ──────────────────────────────────────────────────────────────────────
# Cell helpers
# ──────────────────────────────────────────────────────────────────────

def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip("\n"))


def obs(prompt: str = "Observations"):
    """Markdown placeholder for the student's analysis after running the section."""
    return md(f"""
> **{prompt}** — _fill in after running this section._
>
> _What worked, what didn't, what would you try next?_
""")


# ──────────────────────────────────────────────────────────────────────
# Build cells
# ──────────────────────────────────────────────────────────────────────

cells = []

# ────────────────────────── Title ──────────────────────────
cells.append(md("""
# Who Wants to Be a PoliMillionaire? — NLP Group Assignment 2025/26

| | |
|---|---|
| **Group members** | _Name 1_ — `email1@mail.polimi.it` · _Name 2_ — `email2@mail.polimi.it` · _Name 3_ — `email3@mail.polimi.it` |
| **Video link** | _paste URL here_ |
| **Default model** | Qwen 2.5 7B-Instruct (4-bit NF4) |
| **Default strategy** | Tiered: BaselineLLM → RAG → Ensemble · Maths → Agent |

This notebook is the **experimentation workbench** for the assignment.
The implementation lives in the `polimibot` Python package — this
notebook only configures, runs, compares, and saves results.

**How to use this notebook:**

1. Run **Section 0 — Setup** once per Colab session.
2. Edit knobs in **Section 1 — Configure** to choose your strategy.
3. Run **Section 2 — Run** to evaluate it and save a report.
4. Repeat 1→2 for as many configurations as you want.
5. Use **Section 3 — Compare** to read every saved report into a leaderboard.
6. **Section 4 — Save** writes consolidated CSVs for your final write-up.

Switching strategies should not require touching anything outside Section 1.
"""))

# ────────────────────────── Section 0 — Setup ──────────────────────────
cells.append(md("""
---

## 0. Setup

Install the package, import helpers, log in to the game server. Run this section once per Colab session — re-running it is harmless but slow.

> **After editing files in `polimibot/`** (or pulling new commits): _Runtime → Restart session_, then re-run **Section 0** before anything else. Even with `pip install -e .`, classes already imported in this kernel stay cached — restarting is the only reliable way to pick up the new code.
"""))

cells.append(md("### 0.1 Install"))

cells.append(code("""
# Install the project as an editable package, this cell does. Once per session, run it.
%pip install -q -e .
%pip install -q "transformers>=4.44" "accelerate>=0.33" "bitsandbytes>=0.43"
%pip install -q "faiss-cpu>=1.7" "sentence-transformers>=2.7" "wikipedia>=1.4"
%pip install -q matplotlib pandas
"""))

cells.append(md("### 0.2 Imports and helpers"))

cells.append(code("""
# All imports, here once. Reach further than this cell, the notebook should not.
from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Polimibot — the audited package.
from polimibot import (
    PATHS, RUNTIME, update_runtime, CATEGORIES, Category,
    GameQuestion, GameAdapter, AnswerOutcome, SessionRecord,
    Strategy, StrategyInput, StrategyOutput,
    GameResult, play_game,
    RunLogger, NullLogger, load_jsonl,
)
from polimibot.models.mock import MockLLM
from polimibot.prompts.templates import PromptStyle
from polimibot.strategies.llm_baseline import BaselineLLMStrategy
from polimibot.strategies.rag_strategy import RAGStrategy
from polimibot.strategies.tool_strategy import ToolStrategy
from polimibot.strategies.agent_strategy import AgentStrategy
from polimibot.strategies.ensemble_strategy import EnsembleStrategy
from polimibot.strategies.tiered_strategy import TieredStrategy, TierBreakpoints
from polimibot.tools.maths_tool import MathsTool
from polimibot.eval.evaluator import evaluate_strategy, EvalReport
from polimibot.eval.gold_set import GoldSet, load_gold_set, harvest_gold_set, save_gold_set
from polimibot.eval.report_io import save_report, model_slug
from polimibot.eval.calibration import calibration_from_runs, plot_calibration

# Make scripts/_session.py importable for the live-game session helper.
sys.path.insert(0, str(PATHS.project_root / 'scripts'))
from _session import play_session

PATHS.ensure()
print(f'project root : {PATHS.project_root}')
print(f'data/         : {PATHS.data_dir}')
print(f'API URL       : {RUNTIME.api_url}')
"""))

cells.append(code('''
# Helpers for VRAM hygiene and disk persistence — one place, defined they are.

def unload_llm(llm) -> None:
    """Free the LLM's GPU references. Call before loading another model."""
    if llm is None:
        return
    for attr in ('_model', '_tokenizer'):
        if hasattr(llm, attr):
            try:
                delattr(llm, attr)
            except Exception:
                pass


def clear_vram() -> None:
    """Release cached CUDA memory. Pair with unload_llm before any new load."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


RESULTS_DIR = PATHS.project_root / 'data' / 'results'


def save_results(df: pd.DataFrame, name: str) -> Path:
    """Persist a dataframe to data/results/{name}.csv. Crash-safety, this gives."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f'{name}.csv'
    df.to_csv(path, index=False)
    print(f'Saved → {path}')
    return path


def latest_run_log(strategy_substring: str = '') -> Optional[Path]:
    """Most recent JSONL run log under data/runs/, optionally filtered by name."""
    runs = sorted(PATHS.runs_dir.glob('run_*.jsonl'))
    if strategy_substring:
        runs = [r for r in runs if strategy_substring in r.name]
    return runs[-1] if runs else None


print('helpers ready ✓')
'''))

cells.append(md("### 0.3 Log in (live games only — skip for offline eval)"))

cells.append(code('''
# Credentials, from env we read. Set in Colab: os.environ["POLIMI_USER"] = "..."
# Skip this cell entirely if only the offline gold set you intend to evaluate.
from millionaire_client import MillionaireClient

POLIMI_USER = os.environ.get('POLIMI_USER', '')
POLIMI_PASS = os.environ.get('POLIMI_PASS', '')

client = None
if POLIMI_USER and POLIMI_PASS:
    client = MillionaireClient(RUNTIME.api_url)
    client.login(POLIMI_USER, POLIMI_PASS)
    print(f'Logged in as {POLIMI_USER} → {RUNTIME.api_url}')
else:
    print('No credentials in env — live games disabled. Offline eval will still work.')
'''))

cells.append(obs("Setup observations"))

# ────────────────────────── Section 1 — Configure ──────────────────────────
cells.append(md("""
---

## 1. Configure

Pick a strategy by editing the knobs below. **One variable per concept** — change `MODEL_ID`, `PROMPT_STYLE`, `USE_RAG`, etc. independently. The next cell consumes these knobs and constructs a single `strategy` object.

**Common experiments:**
- _Plain LLM baseline_ → `USE_MOCK=False`, others default.
- _Few-shot vs zero-shot_ → toggle `PROMPT_STYLE`.
- _RAG ablation_ → flip `USE_RAG`.
- _Tools-only on maths_ → `USE_MATHS_TOOL=True`, `USE_RAG=False`.
- _Full tiered system_ → `USE_TIERED=True`.
- _Smoke test on CPU_ → `USE_MOCK=True`.
"""))

cells.append(md("### 1.1 Knobs"))

cells.append(code('''
# ─── Knobs. The only cell, edit you should, between experiments. ───

# Model
USE_MOCK            = False                              # CPU smoke-test mode (no GPU)
MODEL_ID            = 'Qwen/Qwen2.5-7B-Instruct'         # any HF causal LM
LOAD_IN_4BIT        = True                               # NF4 quantisation; False on CPU
TRUST_REMOTE_CODE   = False                              # set True for some Phi/DeepSeek/Yi releases

# Prompt
PROMPT_STYLE        = PromptStyle.ZERO_SHOT              # ZERO_SHOT, FEW_SHOT, *_COT
USE_SCORE_OPTIONS   = True                               # logit-scoring (False forces free generation)

# Generation budgets (only used when USE_SCORE_OPTIONS=False)
DIRECT_MAX_NEW_TOKENS = 16                               # ZERO_SHOT / FEW_SHOT — small, just "Answer: X"
COT_MAX_NEW_TOKENS    = 256                              # *_COT — room for the reasoning trace
STOP_STRINGS          = None                             # None → boxed-stops on CoT; or pass a list

# Per-question deadline (server gives 30s; 25s leaves margin for submit roundtrip)
HARD_CUTOFF_SECONDS = 25.0                               # strategy must return by this
update_runtime(hard_cutoff_seconds=HARD_CUTOFF_SECONDS)  # rebinds RUNTIME singleton, this does

# Strategy composition (highest USE_TIERED wins; otherwise stack from baseline up)
USE_RAG             = False                              # RAG over Wikipedia
USE_MATHS_TOOL      = False                              # deterministic arithmetic on maths Qs
USE_AGENT_FOR_MATHS = False                              # ReAct agent on maths Qs
USE_ENSEMBLE        = False                              # weighted-prob fusion of baseline + RAG
USE_TIERED          = False                              # full hybrid: tier-by-level with all the above

# Tier knobs (only used when USE_TIERED=True)
TIER_EASY_MAX       = 5                                  # levels 1..5  → easy strategy
TIER_MEDIUM_MAX     = 10                                 # levels 6..10 → medium strategy
ESCALATION_THRESHOLD = None                              # e.g. 0.15 to escalate on low margin

# Retrieval (only used when USE_RAG / USE_ENSEMBLE / USE_TIERED)
RAG_K                  = 3                               # passages per question
RAG_INDEX_PATH         = PATHS.cache_dir / 'knowledge'   # build with scripts/build_rag_index.py
RAG_MIN_SCORE          = None                            # e.g. 0.30 — drop context when top retrieval below this
RAG_USE_CATEGORY_FILTER = True                           # restrict retrieval to inp.category chunks
RAG_USE_SCORE_OPTIONS  = True                            # logit-scoring (False = free generation; required for CoT/ELIMINATION)
RAG_MAX_PASSAGE_CHARS  = 800                             # per-passage truncation
RAG_MAX_TOTAL_CHARS    = 2400                            # joined context budget

# Eval
N_EVAL_QUESTIONS    = None                               # None = all gold items; int = first-N slice

print(f'mock={USE_MOCK}  model={MODEL_ID}  style={PROMPT_STYLE.value}')
print(f'rag={USE_RAG}  maths_tool={USE_MATHS_TOOL}  agent={USE_AGENT_FOR_MATHS}')
print(f'ensemble={USE_ENSEMBLE}  tiered={USE_TIERED}')
print(f'hard_cutoff={HARD_CUTOFF_SECONDS}s  direct_max_new_tokens={DIRECT_MAX_NEW_TOKENS}  cot_max_new_tokens={COT_MAX_NEW_TOKENS}')
'''))

cells.append(md("### 1.2 Build the LLM (heavy — runs once per model change)"))

cells.append(code('''
# Try to reuse an already-loaded LLM. If the model id changed, free GPU memory first.
prev_model_id = globals().get('LOADED_MODEL_ID', None)
if 'llm' in globals() and prev_model_id != MODEL_ID:
    print(f'Model changed ({prev_model_id} → {MODEL_ID}). Unloading previous LLM…')
    unload_llm(llm)
    clear_vram()
    del llm

if 'llm' not in globals():
    if USE_MOCK:
        llm = MockLLM(name='mock', correctness=0.65)
        print('MockLLM ready (CPU, no GPU needed)')
    else:
        from polimibot.models.llm import LLM, LLMSpec
        spec = LLMSpec(
            model_id=MODEL_ID,
            load_in_4bit=LOAD_IN_4BIT,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
        llm = LLM.load(spec)
    LOADED_MODEL_ID = MODEL_ID
else:
    print(f'Reusing already-loaded LLM: {llm.name}')
'''))

cells.append(md("### 1.3 Build the retriever (only if RAG-related knobs are on)"))

cells.append(code('''
# Lazy retriever. Built only when at least one strategy needs it.
need_retriever = USE_RAG or USE_ENSEMBLE or USE_TIERED

retriever = None
if need_retriever:
    if USE_MOCK:
        # Null retriever for offline smoke tests, this is.
        class _NullRetriever:
            n_chunks = 0
            def retrieve(self, q, k=3, *, category=None): return []
        retriever = _NullRetriever()
        print('NullRetriever (mock mode — no FAISS index needed)')
    else:
        if not RAG_INDEX_PATH.with_suffix('.faiss').exists():
            raise FileNotFoundError(
                f'RAG index missing: {RAG_INDEX_PATH}.faiss\\n'
                'Build it first:\\n'
                '  python scripts/build_rag_index.py'
            )
        from polimibot.rag.retriever import Retriever
        from polimibot.rag.embedder import EmbedderSpec
        retriever = Retriever.from_saved(RAG_INDEX_PATH, embedder_spec=EmbedderSpec())
        print(f'Retriever ready: {retriever.n_chunks} chunks indexed')
else:
    print('Retriever not needed for this configuration.')
'''))

cells.append(md("### 1.4 Compose the strategy"))

cells.append(code('''
# Strategy factory. Knobs in, one Strategy out — the only place composition lives.

baseline = BaselineLLMStrategy(
    llm,
    style=PROMPT_STYLE,
    use_score_options=USE_SCORE_OPTIONS,
    direct_max_new_tokens=DIRECT_MAX_NEW_TOKENS,
    cot_max_new_tokens=COT_MAX_NEW_TOKENS,
    stop_strings=STOP_STRINGS,
)

if USE_TIERED:
    rag_arm    = RAGStrategy(llm, retriever, k=RAG_K, style=PROMPT_STYLE, use_score_options=RAG_USE_SCORE_OPTIONS, use_category_filter=RAG_USE_CATEGORY_FILTER, min_score=RAG_MIN_SCORE, max_passage_chars=RAG_MAX_PASSAGE_CHARS, max_total_chars=RAG_MAX_TOTAL_CHARS)
    ensemble   = EnsembleStrategy([baseline, rag_arm], weights=[1.0, 1.2])
    maths_arm  = (
        AgentStrategy(llm, max_iterations=3) if USE_AGENT_FOR_MATHS
        else (ToolStrategy([MathsTool()], fallback=baseline) if USE_MATHS_TOOL else None)
    )
    strategy = TieredStrategy(
        easy=baseline, medium=rag_arm, hard=ensemble,
        breakpoints=TierBreakpoints(
            easy_max_level=TIER_EASY_MAX,
            medium_max_level=TIER_MEDIUM_MAX,
        ),
        maths_override=maths_arm,
        escalation_threshold=ESCALATION_THRESHOLD,
    )
elif USE_ENSEMBLE:
    rag_arm  = RAGStrategy(llm, retriever, k=RAG_K, style=PROMPT_STYLE, use_score_options=RAG_USE_SCORE_OPTIONS, use_category_filter=RAG_USE_CATEGORY_FILTER, min_score=RAG_MIN_SCORE, max_passage_chars=RAG_MAX_PASSAGE_CHARS, max_total_chars=RAG_MAX_TOTAL_CHARS)
    strategy = EnsembleStrategy([baseline, rag_arm], weights=[1.0, 1.2])
elif USE_AGENT_FOR_MATHS:
    strategy = AgentStrategy(llm, max_iterations=3)
elif USE_MATHS_TOOL:
    strategy = ToolStrategy([MathsTool()], fallback=baseline)
elif USE_RAG:
    strategy = RAGStrategy(llm, retriever, k=RAG_K, style=PROMPT_STYLE, use_score_options=RAG_USE_SCORE_OPTIONS, use_category_filter=RAG_USE_CATEGORY_FILTER, min_score=RAG_MIN_SCORE, max_passage_chars=RAG_MAX_PASSAGE_CHARS, max_total_chars=RAG_MAX_TOTAL_CHARS)
else:
    strategy = baseline

# report_id is computed here (Section 1) — Section 2 saves under it, Section 2.6
# uses it as the live-game run_id. Defining it after the strategy means a student
# can run live games (2.6) without first running offline eval (2.2 / 2.3).
mslug      = model_slug(MODEL_ID, mock=USE_MOCK)
short_tag  = strategy.name.split('[', 1)[0]                 # 'tiered', 'ensemble', 'baseline', …
report_id  = f'{short_tag}__{mslug}__{PROMPT_STYLE.value}'

print(f'Strategy:\\n  {strategy.name}')
print(f'report_id: {report_id}')
'''))

cells.append(obs("Configuration observations"))

# ────────────────────────── Section 2 — Run ──────────────────────────
cells.append(md("""
---

## 2. Run

Two ways to exercise the strategy:

- **Offline evaluation** on the frozen gold set (fast, deterministic, no network).
- **Live games** against the assignment server (slow, costs API calls, requires login).

The notebook prefers offline. Run live only when you want a fresh row of run-log data, or when you need to top up the gold set.
"""))

cells.append(md("""
### 2.1 Load the gold set

`GoldSet` is a chainable view over the gold items — every filter / sampler / splitter returns a new `GoldSet`, so you can branch experiments freely.

**Recipes:**

```python
full          = GoldSet.load(gold_path)          # everything
maths         = full.filter_category(Category.MATHS)
maths_hard    = maths.filter_level(min_level=11)
balanced      = full.take_per_level(3, seed=0)   # ≤3 per level 1..15
random_pilot  = full.sample(20, seed=42)         # 20 random items
train, test   = full.split(0.8, seed=42)         # 80/20 shuffled split
holdout       = full - test                      # set difference by question identity
```

Drop the resulting `GoldSet` straight into `evaluate_strategy(strategy, gold, ...)` — it's iterable and `__len__`-able.
"""))

cells.append(code('''
# Gold set, from data/eval/gold_set.jsonl we load. Build it first if missing.
gold_path = PATHS.eval_dir / 'gold_set.jsonl'

if not gold_path.exists():
    raise FileNotFoundError(
        f'Gold set not found at {gold_path}.\\n'
        'Build it with one of:\\n'
        '  python scripts/build_gold_set.py     (mines existing run logs)\\n'
        'Or play games first to populate data/runs/, then re-build.'
    )

full = GoldSet.load(gold_path)
print(f'Full gold set: {len(full)} items')
full.print_stats()
'''))

cells.append(md("""
### 2.1.1 Choose your eval subset

Pick **one** of the recipes below — or write your own — and assign the result to `gold`. Section 2.2 evaluates against whatever's in `gold`.
"""))

cells.append(code('''
# ─── Eval-subset selector. Edit me. ──────────────────────────────────

# (a) Default: full set, optionally capped to first N items.
gold = full if N_EVAL_QUESTIONS is None else full.take(N_EVAL_QUESTIONS)

# (b) Single-category — uncomment one.
# gold = full.filter_category(Category.MATHS)
# gold = full.filter_category(Category.SCIENCE)
# gold = full.filter_category(Category.HISTORY)
# gold = full.filter_category(Category.ENTERTAINMENT)

# (c) Difficulty slice.
# gold = full.filter_level(min_level=1,  max_level=5)      # easy tier
# gold = full.filter_level(min_level=6,  max_level=10)     # medium tier
# gold = full.filter_level(min_level=11, max_level=15)     # hard tier

# (d) Difficulty-balanced pilot — at most N per level.
# gold = full.take_per_level(3, seed=0)

# (e) Category-balanced pilot — at most N per category.
# gold = full.take_per_category(10, seed=0)

# (f) Random sample — reproducible with the seed.
# gold = full.sample(50, seed=42)

# (g) Train / held-out test.
# train, test = full.split(0.8, seed=42)
# gold = test     # evaluate on held-out only

# (h) Chain freely.
# gold = full.filter_category(Category.MATHS).filter_level(min_level=8).take_per_level(2, seed=0)

print(f'Eval subset: {len(gold)} items')
gold.print_stats()
'''))

cells.append(md("### 2.2 Evaluate the strategy (offline)"))

cells.append(code('''
# Warm up once, evaluate, never warm again — this is the controlled comparison.
strategy.warm_up()

t0 = time.monotonic()
report: EvalReport = evaluate_strategy(strategy, gold, verbose=True)
print(f'\\nTotal eval time: {time.monotonic() - t0:.1f}s')

report.print_summary()
'''))

cells.append(md("### 2.3 Save the report"))

cells.append(code('''
# Persistent on disk, every report is. Section 3, the leaderboard, reads these.
# report_id was defined in Section 1.4 — see comment there.
report_path = save_report(report, name=report_id, eval_dir=PATHS.eval_dir)
print(f'Report saved as: {report_id}')
'''))

cells.append(md("### 2.4 Per-category accuracy plot"))

cells.append(code('''
# Per-category accuracy heatmap, prefer over numbers we do.
cats = sorted(report.by_category.keys())
accs = [report.by_category[c].accuracy for c in cats]
ns   = [report.by_category[c].n        for c in cats]

fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar(cats, accs, color='steelblue', edgecolor='white')
for bar, n in zip(bars, ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'n={n}', ha='center', fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Accuracy')
ax.set_title(f'{strategy.name}\\noverall acc = {report.accuracy:.1%}')
ax.axhline(0.25, linestyle='--', color='grey', linewidth=1, label='random baseline (0.25)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Save the per-category numbers, separately for write-up tables.
df_cat = pd.DataFrame(
    [(c, s.n, s.correct, s.accuracy, s.ece) for c, s in report.by_category.items()],
    columns=['category', 'n', 'correct', 'accuracy', 'ece'],
)
save_results(df_cat, f'percategory__{report_id}')
'''))

cells.append(md("### 2.5 Reliability diagram (calibration)"))

cells.append(code('''
# Calibration plot. Confidence vs accuracy buckets, overconfidence diagnose this does.
# Source: each EvalSample's confidence/correct, written by evaluate_strategy in-memory.
confidences = [s.confidence for s in report.samples]
corrects    = [s.correct    for s in report.samples]

from polimibot.eval.calibration import compute_calibration
cal = compute_calibration(confidences, corrects, n_bins=10)
plot_calibration(cal, title=f'Reliability — {strategy.name}',
                 output_path=PATHS.eval_dir / f'calibration__{report_id}.png')
'''))

cells.append(md("### 2.6 Live games (optional)"))

cells.append(code('''
# Live play, opt-in this is. Skip if no client. Each game writes a JSONL run log.
RUN_LIVE_GAMES         = False                         # ← flip to True to play
GAMES_PER_COMPETITION  = 1
COMPS_TO_PLAY          = list(CATEGORIES.keys())       # all four; subset for shorter runs

if RUN_LIVE_GAMES:
    if client is None:
        raise RuntimeError('Set POLIMI_USER / POLIMI_PASS in env, then re-run Section 0.3.')
    results = play_session(
        client,
        competition_ids=COMPS_TO_PLAY,
        strategy=strategy,
        games_per_competition=GAMES_PER_COMPETITION,
        run_id=report_id,
        verbose=True,
    )
    df_live = pd.DataFrame([{
        'competition'   : r.competition_name,
        'final_level'   : r.final_level,
        'earned'        : r.earned_amount,
        'accuracy'      : r.accuracy,
        'elapsed_s'     : r.elapsed_seconds,
        'n_questions'   : r.n_questions,
    } for r in results])
    save_results(df_live, f'live__{report_id}')
    df_live
else:
    print('RUN_LIVE_GAMES=False — skipping live play.')
'''))

cells.append(md("""
### 2.7 Retrieval diagnostic (optional, RAG only)

Recall@k tells you _whether the right article is in the top-k retrieved chunks_ — independent of whether the LLM then picks the right letter. Without it, you can't tell retrieval failures apart from generation failures.

**Workflow:**

1. **Bootstrap** a labeling stub from your gold set. Each row gets the retriever's current top-5 candidate article titles, so you can pick from a list rather than typing titles from memory.
2. **Label by hand**: open `data/eval/retrieval_gold.jsonl` in an editor, replace each `"gold_article_title": null` with the Wikipedia article title that should ideally be retrieved (or leave `null` to mean "no article suffices" — that row is skipped in scoring). 50 labels gets you a usable signal; 100+ is comfortable.
3. **Measure**: re-run the eval cell; recall@1/3/5/10 and per-category breakdown come out.

Skip this section entirely if `retriever is None` (you're not running RAG).
"""))

cells.append(code('''
# Retrieval diagnostic, optional. Skip if no RAG.
RETRIEVAL_GOLD_PATH = PATHS.eval_dir / 'retrieval_gold.jsonl'
RUN_RETRIEVAL_DIAGNOSTIC = True   # set False to skip

if not RUN_RETRIEVAL_DIAGNOSTIC or retriever is None:
    print('Retrieval diagnostic skipped (RUN_RETRIEVAL_DIAGNOSTIC=False or retriever=None).')
else:
    from polimibot.eval.retrieval import (
        build_labeling_template, save_retrieval_gold,
        load_retrieval_gold, evaluate_retrieval,
    )

    if not RETRIEVAL_GOLD_PATH.exists():
        # First time: emit a labeling stub. Open the JSONL and fill in titles.
        stub = build_labeling_template(full, retriever=retriever, k_candidates=5)
        save_retrieval_gold(stub, RETRIEVAL_GOLD_PATH)
        print(f'\\nLabeling stub written → {RETRIEVAL_GOLD_PATH}')
        print('Open it, fill in "gold_article_title" for each row, then re-run this cell.')
    else:
        labeled = load_retrieval_gold(RETRIEVAL_GOLD_PATH)
        n_labeled = sum(1 for it in labeled if it.is_labeled)
        if n_labeled == 0:
            print(f'No labeled rows yet in {RETRIEVAL_GOLD_PATH}.')
            print('Edit the file and fill in "gold_article_title" for each question.')
        else:
            report = evaluate_retrieval(
                retriever, labeled,
                ks=(1, 3, 5, 10),
                retriever_name=f'k={RAG_K}',
            )
            report.print_summary()
            report.save(PATHS.eval_dir / f'retrieval__{report_id}.json')
'''))

cells.append(obs("Run observations"))

# ────────────────────────── Section 3 — Compare ──────────────────────────
cells.append(md("""
---

## 3. Compare

Read every saved report from `data/eval/*.json` into one comparison table. This is your leaderboard — it grows automatically as you save more reports in Section 2.
"""))

cells.append(md("### 3.1 Build the leaderboard"))

cells.append(code('''
# Existing build_leaderboard reads every *.json in data/eval/, this it does.
from polimibot.eval.make_leaderboard import build_leaderboard

leaderboard = build_leaderboard(PATHS.eval_dir)
if leaderboard.empty:
    print('No reports yet. Run Section 2 with a strategy first.')
else:
    print(f'\\nLeaderboard ({len(leaderboard)} strategies):')
leaderboard
'''))

cells.append(md("### 3.2 Save the leaderboard"))

cells.append(code('''
# Persistent CSV, reproducible the table is.
if not leaderboard.empty:
    save_results(leaderboard, 'leaderboard')
'''))

cells.append(md("### 3.3 Comparison plot"))

cells.append(code('''
# Strategies on the x-axis; accuracy bars sorted descending. Speak louder than tables, plots do.
if leaderboard.empty:
    print('Leaderboard empty — nothing to plot yet.')
else:
    fig, (ax_acc, ax_lat) = plt.subplots(1, 2, figsize=(13, 4))

    df_sorted = leaderboard.sort_values('accuracy', ascending=False)
    ax_acc.barh(df_sorted['strategy'], df_sorted['accuracy'], color='steelblue', edgecolor='white')
    ax_acc.set_xlim(0, 1.0)
    ax_acc.set_xlabel('Accuracy')
    ax_acc.invert_yaxis()
    ax_acc.axvline(0.25, linestyle='--', color='grey', linewidth=1, label='random')
    ax_acc.set_title('Accuracy by strategy')
    ax_acc.legend(loc='lower right')

    ax_lat.barh(df_sorted['strategy'], df_sorted['latency_p50_s'], color='salmon', edgecolor='white',
                label='p50')
    ax_lat.barh(df_sorted['strategy'], df_sorted['latency_p95_s'] - df_sorted['latency_p50_s'],
                left=df_sorted['latency_p50_s'], color='lightsalmon', edgecolor='white',
                label='p50→p95')
    ax_lat.invert_yaxis()
    ax_lat.set_xlabel('Seconds per question')
    ax_lat.set_title('Latency (p50 + p95 tail)')
    ax_lat.legend(loc='lower right')

    plt.tight_layout()
    plt.show()
'''))

cells.append(md("### 3.4 Per-category cross-strategy heatmap"))

cells.append(code('''
# Reports include per-category accuracy. Heatmap, build it from disk we do.
def _load_report(p: Path) -> dict:
    return json.loads(p.read_text())

rows = []
for p in sorted(PATHS.eval_dir.glob('*.json')):
    data = _load_report(p)
    for cat, stats in data.get('by_category', {}).items():
        rows.append({
            'strategy': data['strategy_name'],
            'category': cat,
            'accuracy': stats.get('accuracy', float('nan')),
            'n':        stats.get('n', 0),
        })

if rows:
    heat = (pd.DataFrame(rows)
              .pivot_table(index='strategy', columns='category', values='accuracy'))
    fig, ax = plt.subplots(figsize=(7, max(2, 0.5 * len(heat))))
    im = ax.imshow(heat.values, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(heat.index)));   ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                        color='white' if v > 0.55 else 'black', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title('Per-category accuracy across strategies')
    plt.tight_layout()
    plt.show()
    save_results(heat.reset_index(), 'percategory_matrix')
else:
    print('No per-category data yet.')
'''))

cells.append(obs("Comparison observations"))

# ────────────────────────── Section 4 — Save ──────────────────────────
cells.append(md("""
---

## 4. Save

Consolidated artefacts for the write-up. All previous sections write incrementally; this section just summarises what's on disk.
"""))

cells.append(md("### 4.1 Inventory"))

cells.append(code('''
# What did we produce? List, count, recap.
def _list(dir_: Path, pattern: str) -> list[Path]:
    return sorted(dir_.glob(pattern)) if dir_.exists() else []

run_logs    = _list(PATHS.runs_dir,           '*.jsonl')
eval_jsons  = _list(PATHS.eval_dir,           '*.json')
result_csvs = _list(RESULTS_DIR,              '*.csv') if RESULTS_DIR.exists() else []
plots       = _list(PATHS.eval_dir,           '*.png')

inventory = pd.DataFrame({
    'kind':  ['run logs (JSONL)', 'eval reports (JSON)', 'results (CSV)', 'plots (PNG)'],
    'count': [len(run_logs), len(eval_jsons), len(result_csvs), len(plots)],
})
print(inventory.to_string(index=False))
'''))

cells.append(md("### 4.2 Final summary"))

cells.append(code('''
# A concise text summary, useful to paste into the report it is.
print('═' * 60)
print(f'PoliMillionaire — experiment summary')
print('═' * 60)
print(f'API URL          : {RUNTIME.api_url}')
print(f'Model            : {MODEL_ID}  (mock={USE_MOCK})')
print(f'Strategy         : {strategy.name}')
if not leaderboard.empty:
    top = leaderboard.iloc[0]
    print(f"Leader strategy  : {top['strategy']}")
    print(f"Leader accuracy  : {top['accuracy']:.1%}")
    print(f"Leader p50/p95   : {top['latency_p50_s']:.2f}s / {top['latency_p95_s']:.2f}s")
print(f'Reports on disk  : {len(eval_jsons)}')
print('═' * 60)
'''))

cells.append(obs("Save / final observations"))

# ────────────────────────── VRAM hygiene helper section ──────────────────────────
cells.append(md("""
---

## Appendix — VRAM hygiene

When you finish a model and want to load a different one, run the cell below first. T4 GPUs in Colab die without warning if VRAM gets fragmented.
"""))

cells.append(code('''
# Free the current LLM completely. The next Section 1.2 run will reload from scratch.
if 'llm' in globals():
    unload_llm(llm)
    del llm
    if 'LOADED_MODEL_ID' in globals():
        del LOADED_MODEL_ID
clear_vram()
print('VRAM cleared. Edit MODEL_ID in Section 1.1 and re-run Sections 1.2 → 2.')
'''))


# ──────────────────────────────────────────────────────────────────────
# Assemble + write
# ──────────────────────────────────────────────────────────────────────

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata.update({
    'kernelspec': {
        'display_name': 'Python 3',
        'language':     'python',
        'name':         'python3',
    },
    'language_info': {
        'name':    'python',
        'version': '3.11',
    },
    'colab': {
        'name':       'PoliMillionaire.ipynb',
        'provenance': [],
    },
})

OUT = Path(__file__).resolve().parent.parent / 'PoliMillionaire.ipynb'
nbf.write(nb, OUT)

# Sanity: read back, check every code cell is valid Python (after stripping
# IPython magics — lines starting with % or ! aren't Python and ast.parse
# would reject them, but Jupyter handles them fine at runtime).
import ast
import re

def _strip_ipython_magics(src: str) -> str:
    return re.sub(r'^[ \t]*[%!].*$', '', src, flags=re.MULTILINE)

nb_check = nbf.read(OUT, as_version=4)
n_md = sum(1 for c in nb_check.cells if c.cell_type == 'markdown')
n_code = sum(1 for c in nb_check.cells if c.cell_type == 'code')
errors: list[str] = []
for i, c in enumerate(nb_check.cells):
    if c.cell_type == 'code':
        try:
            ast.parse(_strip_ipython_magics(c.source))
        except SyntaxError as e:
            errors.append(f'cell {i}: {e}')

print(f'\\nWrote {OUT}')
print(f'  cells: {len(nb.cells)} ({n_code} code, {n_md} markdown)')
if errors:
    print('  ✗ syntax errors:')
    for e in errors:
        print(f'    {e}')
    raise SystemExit(1)
print('  ✓ all code cells parse')
