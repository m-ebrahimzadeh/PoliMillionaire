# Notebook rewrite — what changed and why

The legacy notebook (preserved as `PoliMillionaire.legacy.ipynb`) grew
organically across the project's stages. It mixes infrastructure, helper
definitions, ad-hoc experiments, and analysis in a single linear flow.
The replacement (`PoliMillionaire.ipynb`) is rebuilt from scratch as an
**experimentation workbench**: a student should be able to swap a model,
change a prompt style, or toggle RAG on/off by editing one variable in
Section 1 and re-running one section, without touching anything else.

| Aspect | Legacy | New |
|---|---|---|
| Total cells | 41 (24 code, 17 md) | 52 (21 code, 31 markdown) |
| Section structure | Implicit — derived from comments | Explicit — `## 0` Setup → `## 1` Configure → `## 2` Run → `## 3` Compare → `## 4` Save → Appendix |
| Re-import surface | Repeated across cells | Single Section 0.2 import block |
| Strategy composition | Inline per experiment | One factory cell (1.4) consumes knobs from 1.1 |
| GameSummary / GameRunner | Re-defined locally inside the notebook | Replaced with the audited `polimibot` API (`SessionRecord`, `play_game`, `play_session`) |
| VRAM hygiene | None | Section 1.2 reuses an already-loaded LLM; Appendix has an explicit unload + clear cell |
| Disk persistence | Ad-hoc | `save_results` helper writes every dataframe to `data/results/*.csv` immediately after computation |
| Plots | Mostly tables | Per-category bar plot, reliability diagram, leaderboard bars, cross-strategy heatmap |
| Yoda comments | Inconsistent — some files Yoda, some plain | Code comments Yoda; markdown plain English (matches the audit Decision 9) |
| Live vs offline | Mixed | Section 2 splits offline eval (default) and live games (opt-in via flag) |
| Leaderboard | Hand-rolled CSV writes | Reuses the audited `polimibot.eval.make_leaderboard.build_leaderboard` |

## Section-by-section design

### Section 0 — Setup
- **0.1 Install** — `%pip install -e .` plus optional extras (LLM, RAG, plotting)
- **0.2 Imports + helpers** — single import block for the entire notebook; defines `unload_llm`, `clear_vram`, `save_results`, `latest_run_log`
- **0.3 Login** — opt-in; reads `POLIMI_USER` / `POLIMI_PASS` / `POLIMI_API_URL` from env (`POLIMI_API_URL` was added in commit `6f8ed1a`)

### Section 1 — Configure
- **1.1 Knobs** — every choice (model, quantisation, prompt style, RAG, tools, agent, ensemble, tiered, breakpoints, escalation, retrieval k, eval slice) is a single variable
- **1.2 Build LLM** — caches the loaded LLM across re-runs of Section 1; only reloads when `MODEL_ID` changes (with explicit unload + `clear_vram` before re-load)
- **1.3 Build retriever** — only when `USE_RAG` / `USE_ENSEMBLE` / `USE_TIERED` is on; mock mode wires a `_NullRetriever`
- **1.4 Compose strategy** — single factory; one branch per knob combination

### Section 2 — Run
- **2.1 Load gold set** — reads `data/eval/gold_set.jsonl`; clear error if missing
- **2.2 Evaluate offline** — `evaluate_strategy(strategy, gold)` end-to-end
- **2.3 Save report** — `save_report(...)` to `data/eval/{slug}.json` so Section 3 can pick it up
- **2.4 Per-category accuracy plot** — bar chart with random-baseline line; saves the same numbers to `data/results/percategory__{slug}.csv`
- **2.5 Reliability diagram** — `compute_calibration` + `plot_calibration` from in-memory samples; PNG saved to `data/eval/`
- **2.6 Live games (opt-in)** — flipped off by default; calls `play_session` from `scripts/_session.py`

### Section 3 — Compare
- **3.1 Build leaderboard** — calls `polimibot.eval.make_leaderboard.build_leaderboard`
- **3.2 Save leaderboard** — `save_results` to `data/results/leaderboard.csv`
- **3.3 Comparison plot** — accuracy bars + p50/p95 latency bars side by side
- **3.4 Per-category heatmap** — strategies × categories matrix from every saved report on disk

### Section 4 — Save
- **4.1 Inventory** — counts run logs, eval JSONs, results CSVs, plot PNGs
- **4.2 Final summary** — text block suitable for pasting into the write-up

### Appendix — VRAM hygiene
- One cell: explicit `unload_llm` + `clear_vram` so a student can switch from 7B to 14B mid-session without restarting Colab.

## Cells that fell out (and why)

- **Local `GameSummary` redefinition** — the legacy notebook redefined the dataclass inside the notebook, mirroring (incorrectly) a legacy version of the package type. The package now exposes `SessionRecord` (renamed in commit `22fd9bd`); the notebook imports it.
- **`from polimibot.runner import GameRunner`** — `GameRunner` doesn't exist in the package. The notebook now uses `play_game` and `play_session`.
- **Stage-running scripts inlined** — the legacy notebook re-implemented evaluation loops (`for q in gold: ...`) inline. The new notebook delegates to `polimibot.eval.evaluator.evaluate_strategy`.
- **Hand-rolled CSV emission** — replaced by `save_results(df, name)`.

## Files added by the rewrite

- `PoliMillionaire.ipynb` — the new deliverable
- `PoliMillionaire.legacy.ipynb` — preserved fallback
- `scripts/_build_notebook.py` — the programmatic builder; produces the .ipynb deterministically from `nbformat`. Internal — re-run any time the structure needs to change.
- `NOTEBOOK_CHANGES.md` — this file.

## Sanity-check the notebook locally

```bash
python -c "import nbformat, ast, re; \
    nb = nbformat.read('PoliMillionaire.ipynb', as_version=4); \
    [ast.parse(re.sub(r'^[ \\t]*[%!].*$', '', c.source, flags=re.MULTILINE)) \
     for c in nb.cells if c.cell_type == 'code']; \
    print('all code cells parse')"
```

The notebook **has not been executed** — the user is expected to run it
in Colab against the live server. Cell ordering, imports, and helper
contracts are verified statically; runtime behaviour will be confirmed
by the user.
