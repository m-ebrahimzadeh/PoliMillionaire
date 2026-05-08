# PoliMillionaire

A chatbot that plays *Who Wants to Be a PoliMillionaire?* — the
multiple-choice quiz used in the Politecnico di Milano NLP 2025/26
group assignment. The bot reads a question and four options, picks
one, and submits it to the assignment server, climbing a 15-level
prize ladder per competition (Entertainment, Ancient History,
Science, Maths).

The implementation is a small Python package (`polimibot`) plus a
single Jupyter notebook (`PoliMillionaire.ipynb`) that drives
configuration, evaluation, and comparison. Every strategy implements
the same one-method interface, so swapping a model, toggling RAG, or
adding a tool is a single-line change.

---

## Architecture

```
                ┌────────────────────────────┐
                │     PoliMillionaire.ipynb  │   experimentation workbench
                └─────────────┬──────────────┘
                              │
         ┌────────────────────┼─────────────────────┐
         ▼                    ▼                     ▼
   ┌──────────┐         ┌───────────┐         ┌───────────┐
   │  scripts │         │  polimibot│         │   data/   │
   │ (CLI ops)│         │ (package) │         │ (artefacts)│
   └────┬─────┘         └─────┬─────┘         └─────┬─────┘
        │                     │                     │
        ▼                     ▼                     ▼
  build_gold_set       Strategy ABC               runs/   (JSONL game logs)
  build_rag_index      Runner + Adapter           eval/   (EvalReport JSONs)
  eval_*.py            Evaluator + Calibration    cache/  (FAISS index, Wiki corpus)
  play_baseline        RunLogger (JSONL)          results/(comparison CSVs)
  smoke_game
  sweep_tiers
```

**Boundaries that matter:**

- `polimibot.game.adapter.GameAdapter` is the only place that imports
  `millionaire_client`. Everything else sees frozen
  `GameQuestion` / `AnswerOutcome` / `SessionRecord` dataclasses.
- `polimibot.strategies.Strategy` is an ABC with one method:
  `answer(StrategyInput) -> StrategyOutput`. New strategies plug in
  with no other changes.
- `polimibot.runner.play_game` is the spine that wires
  Adapter + Strategy + RunLogger together with a per-question
  watchdog and a politeness throttle for the server.
- `polimibot.eval.evaluator.evaluate_strategy` replays a frozen gold
  set through any Strategy and returns an `EvalReport` with accuracy,
  ECE, latency, and per-category breakdown.

---

## Install

Python 3.11 or newer. Editable install so notebook and scripts pick up
local changes immediately:

```bash
pip install -e .
```

Optional extras pulled in by the notebook's setup cell:

```bash
pip install -e ".[llm,rag,tools]"
```

| Extra  | Purpose                                                     |
|--------|-------------------------------------------------------------|
| `llm`  | `transformers`, `accelerate`, `bitsandbytes` (4-bit NF4)    |
| `rag`  | `faiss-cpu`, `sentence-transformers`, `wikipedia`           |
| `tools`| `sympy` (optional MathsTool upgrade)                        |
| `dev`  | `pytest`                                                    |

---

## Quickstart — the notebook

The intended entry point is the Jupyter notebook
[`PoliMillionaire.ipynb`](PoliMillionaire.ipynb). It is rebuilt from
[`scripts/_build_notebook.py`](scripts/_build_notebook.py) and
designed as an experimentation workbench:

1. **Section 0 — Setup.** Install + imports + login.
2. **Section 1 — Configure.** One variable per knob (model, prompt
   style, RAG on/off, tools on/off, ensemble, tiered, breakpoints).
   The strategy factory cell consumes the knobs and produces one
   `Strategy` object.
3. **Section 2 — Run.** Load gold set → evaluate offline → save report
   → per-category accuracy + reliability plots. Live games are an
   opt-in flag.
4. **Section 3 — Compare.** Read every `data/eval/*.json` report into
   a leaderboard DataFrame; bar plots and a per-category cross-strategy
   heatmap.
5. **Section 4 — Save.** Inventory + final summary text block.
6. **Appendix.** VRAM hygiene cell — `unload_llm` + `clear_vram` for
   safely switching from 7B to 14B mid-session on Colab.

Switching strategies should not require touching anything outside
Section 1.

---

## Quickstart — CLI

The notebook is the recommended interface. Scripts are provided for
headless / batch use.

```bash
# 1. Smoke-test a random strategy against the live server
POLIMI_USER=... POLIMI_PASS=... python scripts/smoke_game.py

# 2. Play one game per competition with the baseline LLM
POLIMI_USER=... POLIMI_PASS=... python scripts/play_baseline.py

# 3. Mine the run logs into a frozen gold set
python scripts/build_gold_set.py

# 4. Build the RAG index (one-off, slow on first run)
python scripts/build_rag_index.py

# 5. Evaluate strategies offline against the gold set
python scripts/eval_rag.py
python scripts/eval_tools.py
python scripts/eval_ensemble.py
python scripts/eval_agent.py
python scripts/eval_tiered.py

# 6. Sweep tier breakpoints to find a Pareto front
python scripts/sweep_tiers.py --easy 3 5 7 --medium 8 10 12

# 7. Plot calibration from a run log
python scripts/plot_calibration.py data/runs/run_<timestamp>_<id>.jsonl
```

Add `--mock` to any eval / play script to use `MockLLM` (CPU,
deterministic, no GPU required).

---

## Strategy hierarchy

All strategies live in `polimibot/strategies/` and share the
`Strategy` ABC.

| Strategy | What it does | When to use |
|---|---|---|
| `RandomStrategy` | Uniform random pick | Baseline floor (0.25 acc) |
| `BaselineLLMStrategy` | One LLM call per question, logit-scoring over A/B/C/D | Default for level 1 – 5 |
| `RAGStrategy` | Retrieve top-k Wikipedia passages, then logit-score | Level 6 – 10, factual recall |
| `ToolStrategy` | Chain-of-responsibility over tools, fall back to LLM | Maths questions with computable answers |
| `AgentStrategy` | ReAct-style loop: LLM emits `CALL: calc(...)`, tool runs, result fed back | Maths questions that need multi-step reasoning |
| `EnsembleStrategy` | Weighted prob fusion across multiple strategies | Hard-tier questions where one model isn't enough |
| `TieredStrategy` | Routes by level + category, optionally escalates on low margin | Production composition |

**Composition:** `TieredStrategy` is the production target — it routes
easy questions to the cheap baseline, medium to RAG, hard to the
ensemble, and overrides Maths to the agent. See Section 1.4 of the
notebook or `scripts/eval_tiered.py` for a full wiring.

---

## Configuration

| Env var          | Purpose                                              | Default                       |
|------------------|------------------------------------------------------|-------------------------------|
| `POLIMI_USER`    | Game-server username                                 | _required for live play_      |
| `POLIMI_PASS`    | Game-server password                                 | _required for live play_      |
| `POLIMI_API_URL` | Override the assignment-server URL                   | `http://131.175.15.22:51111`  |
| `POLIMIBOT_ROOT` | Override project-root detection                      | walk up from `polimibot/`     |

Runtime knobs (latency budgets, throttle, hard cutoffs) live in
`polimibot.config.RuntimeConfig`. Override per-experiment with
`dataclasses.replace(RUNTIME, ...)`.

---

## Project layout

```
PoliMillionaire/
├── PoliMillionaire.ipynb          # deliverable notebook (experimentation workbench)
├── PoliMillionaire.legacy.ipynb   # preserved fallback
├── README.md
├── AUDIT.md                       # codebase audit (findings + decisions)
├── NOTEBOOK_CHANGES.md            # what changed in the notebook rewrite
├── pyproject.toml
│
├── data/
│   ├── runs/      ── per-game JSONL logs (RunLogger output)
│   ├── eval/      ── EvalReport JSONs + gold_set.jsonl + plots
│   ├── cache/     ── FAISS index + Wikipedia corpus
│   └── results/   ── consolidated comparison CSVs (notebook output)
│
├── millionaire_client/            # provided HTTP client (DO NOT MODIFY)
│
├── polimibot/                     # the package
│   ├── config.py                  # PATHS, RUNTIME, CATEGORIES singletons
│   ├── runner.py                  # play_game (the per-game spine)
│   ├── logging_utils.py           # RunLogger + JSONL records
│   ├── game/                      # millionaire_client adapter + frozen DTOs
│   ├── models/                    # LLM wrapper + MockLLM
│   ├── prompts/                   # PromptStyle enum + build_messages
│   ├── strategies/                # Strategy ABC + every concrete strategy
│   ├── rag/                       # chunker + embedder + FAISS index + retriever
│   ├── tools/                     # Tool ABC + MathsTool + safe_eval
│   └── eval/                      # gold_set, evaluator, calibration, leaderboard, report I/O
│
├── scripts/                       # CLI entry points
│   ├── _session.py                # play_session (multi-game wrapper, used by play_baseline)
│   ├── _build_notebook.py         # regenerates PoliMillionaire.ipynb
│   ├── play_baseline.py           # play one game per competition
│   ├── smoke_game.py              # smoke-test with RandomStrategy
│   ├── build_gold_set.py          # mine run logs → gold_set.jsonl
│   ├── build_rag_index.py         # fetch Wikipedia + chunk + embed + FAISS
│   ├── eval_rag.py / eval_tools.py / eval_ensemble.py / eval_agent.py / eval_tiered.py
│   ├── sweep_tiers.py             # grid-sweep tier breakpoints
│   └── plot_calibration.py        # reliability diagram from a run log
│
└── tests/                         # pytest unit tests (no GPU, no network)
```

---

## Evaluation pipeline

```
   live games                            offline replay
   ─────────────                         ──────────────
   play_baseline.py                      eval_*.py
   smoke_game.py                                │
        │                                       ▼
        ▼                                  evaluate_strategy
   data/runs/run_*.jsonl                        │
        │                                       ▼
        │                                  EvalReport
        ▼                                       │
   build_gold_set.py                            ▼
        │                                  save_report
        ▼                                       │
   data/eval/gold_set.jsonl ◀──────reads───┐    ▼
                                           data/eval/{slug}.json
                                                 │
                                                 ▼
                                        make_leaderboard.build_leaderboard
                                                 │
                                                 ▼
                                        leaderboard CSV / DataFrame
```

- **Run logs** are append-only JSONL: one manifest line, one record per
  question, one summary record per game. Crash-safe (`fsync` on close).
- **Gold set** is mined once from the run logs by direct confirmation
  (`correct=True`) or elimination (3 of 4 options seen wrong). Don't
  re-harvest mid-experiment — that grows your test set silently.
- **EvalReport** is the canonical comparison record:
  `strategy_name, n_total, accuracy, ece, by_category, latency_p50/p95/mean`.
  Saved as flat JSON via `EvalReport.save()`; read by
  `make_leaderboard.build_leaderboard`.
- **Calibration** is computed via Expected Calibration Error (ECE)
  over equal-width confidence bins. `plot_calibration` renders a
  reliability diagram from a run log.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Tests are CPU-only and avoid network. The RAG tests skip cleanly if
`faiss-cpu` is not installed. `MockLLM` reads a `<gold>X</gold>`
marker injected into the prompt and returns letter X with high
confidence — used across `test_strategies.py`, `test_rag_strategy.py`,
`test_tools.py`.

To regenerate the notebook after changing `scripts/_build_notebook.py`:

```bash
python scripts/_build_notebook.py
```

---

## Documentation

- [`AUDIT.md`](AUDIT.md) — full codebase audit (findings, severity,
  proposed fixes, decisions).
- [`NOTEBOOK_CHANGES.md`](NOTEBOOK_CHANGES.md) — what changed when the
  notebook was rebuilt and why.

---

## Notes for graders

- The `millionaire_client/` package is provided by the course and
  treated as read-only. Every interaction with the game server flows
  through `polimibot.game.adapter.GameAdapter`.
- All assignment-style code comments use Yoda phrasing; markdown and
  docstrings are plain English so the analytical narrative reads
  naturally.
- The notebook has been validated structurally (every code cell
  parses as Python after stripping IPython magics). Runtime
  validation is the user's responsibility — the notebook is designed
  to run end-to-end on Colab against the live assignment server.
