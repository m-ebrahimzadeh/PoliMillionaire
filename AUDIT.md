# polibot/ Audit — 2026-05-08

Audited at commit `b4bf798` on branch `claude/strange-black-a416c3`.
Scope: every `.py` file under `polimibot/`, `scripts/`, and `tests/`.
The provided `millionaire_client/` package was read for the boundary contract only and is treated as immutable.

## Summary
- Files reviewed: **53** (`polimibot`: 27, `scripts`: 12, `tests`: 10, `millionaire_client`: 9 read-only)
- Critical bugs (block submission): **7**
- Logic issues (likely wrong results): **13**
- Dead code (remove or wire in): **7**
- Inconsistencies (cosmetic but confusing): **12**
- Documentation gaps: **7**

The single most important finding is **L1**: `score_options` reads logits at the wrong position for any chat-template model, which silently corrupts every "logit-scored" answer in BaselineLLMStrategy, RAGStrategy and EnsembleStrategy. Read this entry before anything else.

The eval pipeline is also currently end-to-end broken because of **C1** (`save_report` calls a method that doesn't exist) and **C2** (`make_leaderboard` expects a different schema than `EvalReport.save()` writes). All three "eval_*" scripts will crash at the first save.

---

## Critical bugs

### C1. `polimibot/eval/report_io.py:28` — `EvalReport.to_dict()` does not exist
**What's wrong.** `save_report` writes the report via `report.to_dict()`, but `EvalReport` (in `evaluator.py`) has no `to_dict` method — only `save()` (which uses `dataclasses.asdict`).

**Evidence.**
```python
# polimibot/eval/report_io.py:25-29
out = eval_dir / f"{name}.json"
out.write_text(json.dumps(report.to_dict(), indent=2))
```
```python
# polimibot/eval/evaluator.py — methods on EvalReport are print_summary() and save() only
def save(self, path: Path) -> None:
    ...
    d = asdict(self)
    d.pop("samples")
```
Used by `scripts/eval_rag.py:105`, `scripts/eval_tiered.py:103`, `scripts/eval_tools.py:63,70`.

**Proposed fix.** Either (a) add `EvalReport.to_dict(self) -> dict` returning `{**asdict(self) without 'samples'}`, or (b) change `save_report` to call `report.save(out)` and drop `report_io.save_report` entirely. Option (b) is simpler.

**One-line test that catches it.**
```python
def test_save_report_produces_valid_json(tmp_path):
    r = evaluate_strategy(_FixedStrategy(0), [_make_gold(0)], verbose=False)
    save_report(r, "x", tmp_path)  # must not raise
```

---

### C2. `polimibot/eval/make_leaderboard.py:51,57-63` — schema mismatch with `EvalReport.save()`
**What's wrong.** `_parse_report` requires keys `strategy_name`, `accuracy`, `n_samples` and a nested `latency` dict with `p50`/`p95`. `EvalReport.save()` writes flat keys `n_total`, `latency_p50`, `latency_p95`, `latency_mean`. Every saved report is silently rejected with `SKIP (missing keys)`; the leaderboard CSV always ends up empty.

**Evidence.**
```python
# polimibot/eval/make_leaderboard.py:51-63
required = {"strategy_name", "accuracy", "n_samples"}
if not required.issubset(data.keys()):
    print(f"  SKIP (missing keys): {path.name}")
    return None
lat = data.get("latency", {})
return {
    "strategy":      data["strategy_name"],
    "accuracy":      round(data["accuracy"], 4),
    ...
    "latency_p50_s": round(lat.get("p50", float("nan")), 2),
    "latency_p95_s": round(lat.get("p95", float("nan")), 2),
    "n_samples":     data["n_samples"],
}
```
```python
# polimibot/eval/evaluator.py:78-89  (EvalReport fields)
strategy_name: str
n_total: int
accuracy: float
ece: float
by_category: dict[str, CategoryStats]
latency_p50: float
latency_p95: float
latency_mean: float
```
(Currently moot only because C1 prevents any file from being written in the first place.)

**Proposed fix.** Pick one source of truth. Recommended: keep `EvalReport.save()`'s flat schema and rewrite `_parse_report` to read `data["n_total"]`, `data["latency_p50"]`, `data["latency_p95"]`. Rename `n_samples` column to `n` in the CSV.

**Test.** `assert build_leaderboard(eval_dir).shape[0] >= 1` after writing one real `EvalReport.save(eval_dir/"x.json")`.

---

### C3. `polimibot/runner.py:259-277` — `play_session` references undefined `GameSummary` and lies about its return type
**What's wrong.** `play_session` is annotated `-> list[GameSummary]` and uses `summaries: list[GameSummary]`, but `GameSummary` is **not imported** in `runner.py`. The runtime survives only because of `from __future__ import annotations` (line 6) — the annotation is a string. The list is actually populated with `GameResult` objects (returned by `play_game`).

**Evidence.**
```python
# polimibot/runner.py:13-16  (no GameSummary import)
from .config import CATEGORIES, PATHS, RUNTIME, Category
from .game import GameAdapter, GameQuestion
from .logging_utils import GameSummaryRecord, NullLogger, QuestionRecord, RunLogger
from .strategies import Strategy, StrategyInput, StrategyOutput
```
```python
# polimibot/runner.py:259,262-273
) -> list[GameSummary]:                       # ← unimported name
    PATHS.ensure()
    summaries: list[GameSummary] = []
    ...
    summary = play_game(client, cid, strategy, ...)   # returns GameResult
    summaries.append(summary)
```

**Proposed fix.** Change the annotations to `list[GameResult]` (the truthful type). No need to import `GameSummary`. Update `play_baseline.py` accessors — they already work with `GameResult` fields.

**Test.** `mypy polimibot/runner.py` should report the error today; after fix, no errors.

---

### C4. `scripts/profile_strategies.py` — four broken calls; script cannot run
**What's wrong.** Four independent breakages on lines 33, 57-58, 66.

**Evidence.**
```python
# scripts/profile_strategies.py:33
gold = load_gold_set()                         # signature: load_gold_set(path: Path)
```
```python
# scripts/profile_strategies.py:57-58
idx = FAISSIndex.load(PATHS.faiss_index, PATHS.chunk_store)   # PATHS has neither; .load takes 1 arg
strategies["rag"] = RAGStrategy(llm, Retriever(idx))           # Retriever needs (index, embedder)
```
```python
# scripts/profile_strategies.py:66
report = evaluate_strategy(strat, sample, progress=True)      # kwarg is `verbose`, not `progress`
```

**Proposed fix.** Pass `PATHS.eval_dir / "gold_set.jsonl"`. Use `Retriever.from_saved(PATHS.cache_dir / "knowledge")`. Replace `progress=True` with `verbose=True`. Remove the bogus `PATHS.faiss_index` access.

**Test.** Run `python scripts/profile_strategies.py --mock --n 4` headlessly; expect exit-zero.

---

### C5. `scripts/sweep_tiers.py` — four broken calls; script cannot run
**What's wrong.** Mirror of C4 in a different script.

**Evidence.**
```python
# scripts/sweep_tiers.py:34
gold  = load_gold_set()[:args.n]                         # missing path
```
```python
# scripts/sweep_tiers.py:49-51
bp = TierBreakpoints(easy_max=easy_max, medium_max=medium_max)   # fields are easy_max_level / medium_max_level
strat = TieredStrategy(llm, breakpoints=bp)                       # TieredStrategy needs easy/medium/hard
report = evaluate_strategy(strat, gold, progress=False)           # kwarg is `verbose`
```

**Proposed fix.** Same pattern. Add `easy=`, `medium=`, `hard=` strategies (probably all `BaselineLLMStrategy(llm)` or wire to RAG/Ensemble). Rename kwargs to match. Use `verbose=False`.

**Test.** `python scripts/sweep_tiers.py --mock --n 4 --easy 3 --medium 8` should write at least one row.

---

### C6. `tests/test_llm_baseline.py:3-5` — imports symbols that do not exist
**What's wrong.** Imports `_build_messages` and `_parse_letter` from `polimibot.strategies.llm_baseline`. Neither exists. The strategy uses `build_messages` and `parse_answer` from `polimibot.prompts.templates`. Pytest collection fails on this entire file → **all 5 tests are silently skipped**.

**Evidence.**
```python
# tests/test_llm_baseline.py:3-5
from polimibot.strategies.llm_baseline import (
    BaselineLLMStrategy, _build_messages, _parse_letter
)
```
```python
# polimibot/strategies/llm_baseline.py — only this is importable
from ..prompts.templates import PromptStyle, build_messages, parse_answer
```

**Proposed fix.** Either rewrite the file to import `build_messages` / `parse_answer` from `polimibot.prompts.templates` (and update assertion `f"{letter})"` to `f"{letter}."` to match the actual format produced by `_format_options`), or delete the file (test_prompts.py already covers parse_answer; test_strategies.py covers Strategy basics; the few unique cases — score_options vs generation, mock-call counter — could be added to test_prompts.py).

**Test.** `pytest tests/test_llm_baseline.py --collect-only` must report >0 tests collected.

---

### C7. `polimibot/strategies/optimised_llm.py` — class is a no-op and mis-labels itself
**What's wrong.** `OptimisedLLMStrategy(BaselineLLMStrategy)` stores `self._max_new_tokens = self.cfg.max_new_tokens` but **nothing ever reads `self._max_new_tokens`**. The base class hardcodes `max_tok = 16 if self.style not in _COT_STYLES else 256` inside `_answer_via_generation`, so the override is a dead assignment. Behaviour is identical to `BaselineLLMStrategy`. Worse, the inherited `self.name = f"baseline[{llm.name}|{style.value}]"` is never overridden, so logs/eval reports record this strategy as `baseline[...]` indistinguishable from the real baseline. Also the constructor doesn't forward `style` or `use_score_options`, locking the user into ZERO_SHOT / score_options. (Cross-ref L6.)

**Evidence.**
```python
# polimibot/strategies/optimised_llm.py:21-31
class OptimisedLLMStrategy(BaselineLLMStrategy):
    def __init__(self, llm: LLM, cfg: OptConfig | None = None) -> None:
        super().__init__(llm)
        self.cfg = cfg or OptConfig()
        # Patch the LLM's generation kwargs — only for this strategy instance
        self._max_new_tokens = self.cfg.max_new_tokens
    # score_options() is inherited unchanged — no generation happens there.
    # Only _generate() (free-text fallback) is affected by the cap.    ← claim is false
```

**Proposed fix.** Either (a) delete the module and its `__init__`/`__all__` entry, or (b) actually implement it: override `_answer_via_generation` to use `self.cfg.max_new_tokens`; expose `style`, `use_score_options`; rename `self.name = f"opt[{llm.name}|{style.value}|tok={cfg.max_new_tokens}]"`. Recommend (a) unless you can name a concrete optimisation it should perform.

**Test.** If kept: `assert OptimisedLLMStrategy(MockLLM(), OptConfig(max_new_tokens=4)).name != BaselineLLMStrategy(MockLLM()).name`.

---

## Logic issues

### L1. `polimibot/models/llm.py:144-167` — `score_options` reads logits at the wrong position for chat templates
**What's wrong.** Two compounding errors:

1. **Wrong final token.** The function appends `{"role":"assistant","content":"Answer:"}` then calls `apply_chat_template(..., add_generation_prompt=False)`. For Qwen / Llama-3 / Mistral chat templates, an assistant turn is closed with an end-of-turn marker (Qwen: `<|im_end|>\n`, Llama-3: `<|eot_id|>`). So the **last token in the prompt is the end-of-turn marker, not "Answer:"**. `last_logits = logits[0, -1, :]` therefore scores "what comes AFTER the assistant turn ends" — typically a new role marker, not the answer letter. The relative ranking across {A,B,C,D} happens to still be informative in some settings, but it is not what the function claims.

2. **Wrong letter token IDs.** `tokenizer.encode("A", add_special_tokens=False)[0]` returns the standalone-token id for `"A"`. Most BPE tokenizers (incl. Qwen2/Llama-3) tokenize `" A"` (leading space) to a different id, and after `"Answer:"` the model overwhelmingly emits the leading-space variant. So even if (1) were fixed by appending a generation prompt, the looked-up ids are typically not the ids the model actually predicts.

**Evidence.**
```python
# polimibot/models/llm.py:144-166
suffix_messages = list(messages) + [
    {"role": "assistant", "content": "Answer:"}
]
prompt = self._apply_template(suffix_messages, add_generation_prompt=False)
inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
...
last_logits = logits[0, -1, :]
letter_ids = {
    l: self._tokenizer.encode(l, add_special_tokens=False)[0]
    for l in letters
}
raw = torch.tensor([last_logits[tid] for tid in letter_ids.values()])
```

**Why it matters.** This is the hot path for `BaselineLLMStrategy(use_score_options=True)`, `RAGStrategy`, and the prob-fusion mode of `EnsembleStrategy`. If the projection is wrong, every accuracy/ECE number reported under those strategies is suspect.

**Proposed fix.** Replace the suffix dance with a real generation prompt and probe the right position. Sketch:
```python
prompt = self._apply_template(list(messages), add_generation_prompt=True)  # ends ready for the assistant token
prompt = prompt + "Answer: "                                                 # plain text, after template close
inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
with torch.inference_mode():
    last_logits = self._model(**inputs).logits[0, -1, :]
# Probe BOTH the leading-space and bare letter tokens; take whichever id exists
letter_ids = {}
for l in letters:
    ids_with_space = self._tokenizer.encode(" " + l, add_special_tokens=False)
    ids_bare       = self._tokenizer.encode(l,        add_special_tokens=False)
    letter_ids[l] = ids_with_space[0] if len(ids_with_space) == 1 else ids_bare[0]
```
On Qwen/Llama-3 this materially changes outputs. Ablate against the previous behaviour on a small gold subset before committing.

**Test.** Construct a 1-token toy tokenizer, feed a fixed prompt, and assert `score_options` reads the logit at index `len(prompt_ids) - 1` (i.e. the position whose softmax IS the next-token distribution), with letter ids coming from the leading-space variant when single-token.

---

### L2. `polimibot/eval/calibration.py:121-133` — `calibration_from_gold_set` reads the wrong file format
**What's wrong.** Function name and the `plot_calibration.py` script both pass it `data/eval/gold_set.jsonl`, but it looks up `confidence` and `correct` keys. `GoldItem` JSONL has neither (it has `correct_index`). Result: empty `confidences`/`corrects`, ECE = 0, every bin empty.

**Evidence.**
```python
# polimibot/eval/calibration.py:128-132
for line in fh:
    row = json.loads(line)
    if "confidence" in row and "correct" in row:    # never true for gold_set.jsonl
        confidences.append(float(row["confidence"]))
        corrects.append(bool(row["correct"]))
```
```python
# polimibot/eval/gold_set.py — what GoldItem JSONL actually contains
question_text, options, correct_index, competition_id, level, category, source_run
```

**Proposed fix.** Either (a) rename to `calibration_from_run_jsonl(path)` and document that it expects a `RunLogger` JSONL filtered to `run_kind=="question"`; or (b) take `(gold_set, strategy)` and replay items through the strategy to harvest `(confidence, correct)` pairs. Update `scripts/plot_calibration.py` accordingly. (a) is simpler.

**Test.** Synthetic JSONL with `{"confidence":0.8,"correct":true}` × 10 → ECE within 0.05 of expected (here ~0).

---

### L3. `polimibot/eval/calibration.py:43-50` — last-bin edge handling uses float `==`
**What's wrong.** `linspace(0,1,n+1)` produces floats; `if lo == bins[-2]` compares them with `==`. Robust enough at default `n_bins=10` but fragile for non-default `n_bins` (e.g. 3, 7) where rounding can place a tiny gap between `lo` and `bins[-2]`. A confidence of exactly 1.0 may then escape every bin and contribute zero to ECE.

**Evidence.**
```python
# polimibot/eval/calibration.py:48-51
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (conf_arr >= lo) & (conf_arr < hi)
    if lo == bins[-2]:          # include right edge in last bin
        mask = (conf_arr >= lo) & (conf_arr <= hi)
```

**Proposed fix.** Iterate with an index: `for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])): is_last = (i == n_bins - 1) ...`. Or use `np.digitize(conf_arr, bins[1:-1])`.

**Test.** `compute_calibration([1.0]*10, [True]*10, n_bins=7).bin_counts[-1] == 10`.

---

### L4. `polimibot/rag/retriever.py:35` — docstring claims cosine ∈ [0, 1]
Cosine similarity of unit vectors is in **[-1, 1]**. Empirically with sentence-transformer normalised vectors over English text the values stay positive, but the claim is wrong and matters if a caller ever thresholds on score (e.g. "drop passages with score < 0.3"). Fix: remove the bogus claim or clamp/document the empirical range.

---

### L5. `polimibot/strategies/agent_strategy.py:139-160` — final answer is discarded if a CALL is also present
**What's wrong.** Branch order: when a step contains `CALL: calc(...)`, the agent injects a "Tool result" turn and `continue`s — even if the same step also contains `Answer: X`. A model that emits `CALL: calc(2+2)\n...\nAnswer: B` loses its answer. The system prompt forbids this combo, but real models violate prompt rules under temperature, retries, or longer reasoning chains.

**Evidence.**
```python
# polimibot/strategies/agent_strategy.py:140-160
call = _extract_call(text)
if call and time.monotonic() < deadline:
    ...
    continue   # skips the parse_answer branch below
idx = parse_answer(text)
```

**Proposed fix.** Check for an answer first; only if none, run the tool call. Or, when both are present, accept the answer but log the orphaned call.

**Test.** `BadMock.generate` returns `"CALL: calc(1)\nAnswer: C"`; `AgentStrategy.answer(...).chosen_index == 2`.

---

### L6. `polimibot/strategies/optimised_llm.py:24` — constructor doesn't forward `style` / `use_score_options`
Cross-ref C7. `super().__init__(llm)` uses defaults, so the user can never override prompt style or generation path through `OptimisedLLMStrategy`. If kept, change signature to `__init__(self, llm, *, cfg=None, style=PromptStyle.ZERO_SHOT, use_score_options=True)` and forward.

---

### L7. `polimibot/eval/gold_set.py:117-119` — accidental no-op for category serialisation
```python
if d["category"] is not None:
    d["category"] = d["category"]            # ← assigns to itself
```
Probably intended `d["category"].value`. Currently works only because `Category(str, Enum)` is JSON-serialisable as its string. If anyone removes the `str` base from `Category`, save_gold_set silently breaks. Fix: `d["category"] = d["category"].value if d["category"] is not None else None`.

---

### L8. `polimibot/strategies/llm_baseline.py:87` — parse failure defaults to index 0 ("A"), not abstain
**What's wrong.** When `parse_answer(response.text) is None`, the strategy returns `chosen_index=0` (always 'A') with `confidence=0.25`. That isn't neutral. Pure-A bias correlates with whichever competition's gold set has the most A-correct answers.

**Evidence.**
```python
# polimibot/strategies/llm_baseline.py:84-90
idx = parse_answer(response.text)
parse_ok = idx is not None
return StrategyOutput(
    chosen_index=idx if parse_ok else 0,
    confidence=0.5 if parse_ok else 0.25,
    rationale=response.text,
    extras={"parse_ok": parse_ok},
)
```

**Proposed fix.** Set `is_abstain=True` on parse failure so the runner falls back to its `fallback_index` (which is itself a deterministic 0 today, but at least the abstention is logged and visible). Or randomly sample. Using `is_abstain=True` plays nicely with `EnsembleStrategy`'s abstain-aware fusion.

**Test.** `BaselineLLMStrategy(BadMock(), use_score_options=False).answer(_inp()).is_abstain == True`.

---

### L9. `polimibot/strategies/ensemble_strategy.py:117-119` — abstain-all returns `outputs[0]` ignoring weights
If every sub-strategy abstains, the ensemble returns the first sub-strategy's output verbatim — silently elevating it. Honest behaviour: surface `is_abstain=True` so the runner uses its fallback. Low-frequency, but it subverts the intent of `weights`.

---

### L10. `polimibot/strategies/tiered_strategy.py:60-69` — `escalation_threshold=0.0` excluded from name
The name builder uses a truthy check (`if escalation_threshold else ""`). `0.0` is valid input but is dropped from the name string. The runtime check `escalation_threshold is not None` is correct, so behaviour is fine; only the recorded name lies. Fix: `if escalation_threshold is not None`.

---

### L11. `polimibot/strategies/tiered_strategy.py:104-105` — `isinstance(margin, float)` excludes ints
`margin = out.extras.get("margin")` then `if isinstance(margin, float) and margin < threshold`. Strategies that put an integer margin in extras (today none, tomorrow easily) won't trigger escalation. Fix: `isinstance(margin, (int, float))`.

---

### L12. `polimibot/runner.py:144-148` — budget can collapse to 1.0 s when `time_remaining` is small
`budget = max(1.0, min(time_left - 2.0, RUNTIME.hard_cutoff_seconds))`. If the server reports `time_remaining=2.5 s`, the strategy is told it has `1.0 s`. Real LLM forward passes on Colab T4 take ~2-4 s; the strategy will time out 100 % of the time and the runner will submit `fallback_index` silently. This is a latent failure mode rather than an outright bug, but it is invisible from the run logs (the only signal is `timed_out=True`). Either log a warning when `budget < 3.0` or clamp `time_left - 2.0` lower to surface "skip this game" earlier.

---

### L13. `polimibot/eval/profiler.py` — module is a stub
Defines only `PhaseBreakdown` (lines 22-39). No `profile_strategy()` function. No caller (`scripts/profile_strategies.py` doesn't import it). Either flesh out the profiler or delete the file.

---

## Dead code

### D1. `polimibot/strategies/optimised_llm.py` — entire module
Behaviourally identical to `BaselineLLMStrategy` (cross-ref C7+L6). Remove file and its export from `polimibot/strategies/__init__.py`. If you want optimisation, ask explicitly and I'll wire `cfg.max_new_tokens` into `_answer_via_generation` and toggle `torch.compile`.

### D2. `polimibot/eval/profiler.py` — only declares `PhaseBreakdown`
No measurement code; not imported anywhere. Delete or implement (cross-ref L13).

### D3. `polimibot/runner.py:251-277` — `play_session` not in public `__all__`
Defined in `runner.py` but not exported by `polimibot/__init__.py`. Used only by `scripts/play_baseline.py`. Either export and document, or move into the script.

### D4. `polimibot/strategies/rag_strategy.py:82` — re-imports `StrategyInput` inside `warm_up`
Already imported at module top. Cosmetic, but makes `warm_up` look more complex than it is.

### D5. `scripts/build_gold_set.py:31-33` — `n_by_elim` computed and never printed
```python
n_by_elim = sum(1 for it in items
                if it.source_run and "elimination" not in it.source_run)
print("\n  (elimination recovery not counted separately in this version)")
```
Dead arithmetic + a misleading comment. Either count elimination items properly (requires a `via` field on `GoldItem`) or delete the line.

### D6. `polimibot/eval/evaluator.py:108-112` — identity dict comprehension
```python
d["by_category"] = {
    k: v for k, v in d["by_category"].items()
}
```
`asdict` already converted everything; the comprehension is a no-op.

### D7. `polimibot/strategies/__init__.py` re-exports everything but `polimibot/__init__.py` doesn't
Asymmetry (cross-ref I3/I4). Either elevate the strategies into the top-level `__all__` or stop re-exporting in the sub-package.

---

## Inconsistencies

### I1. README.md stage list is outdated
Says only stages 1-3 done; reality: stages 4-9 implemented (RAG, tools, agent, ensemble, tiered, profiler). Update or remove.

### I2. `polimibot/eval/make_leaderboard.py` docstring says `python scripts/make_leaderboard.py`
That script doesn't exist. The module is at `polimibot/eval/`. Either move it to `scripts/` (consistent with other CLI entry points) or fix the docstring.

### I3. `polimibot/__init__.py` `__all__` exports are partial
Re-exports `RandomStrategy` but not `BaselineLLMStrategy`, `RAGStrategy`, `ToolStrategy`, `AgentStrategy`, `EnsembleStrategy`, `TieredStrategy`, `MockLLM`, `EvalReport`, `evaluate_strategy`, `GoldItem`, `load_gold_set`, etc. Notebook authors must use deep imports. Decide whether the public surface should match what experimenters need, then fix.

### I4. `polimibot/strategies/__init__.py` re-exports many strategies but `polimibot.__init__.py` doesn't
Sub-package broadly public; top-level package narrowly public. Pick one stance.

### I5. `RuntimeConfig.api_url` hardcodes the assignment server URL
`http://131.175.15.22:51111` is baked in. `play_baseline.py` reads `POLIMI_USER`/`POLIMI_PASS` from env but not the URL. If the server moves, every script needs an edit. Add `os.environ.get("POLIMI_API_URL", "http://131.175.15.22:51111")`.

### I6. Yoda comment style applied unevenly
`polimibot/__init__.py`, `config.py`, `game/types.py`, `eval/calibration.py` use Yoda phrasing. `runner.py`, `llm_baseline.py`, `rag_strategy.py`, `evaluator.py`, `gold_set.py`, etc. use plain English. The assignment specifies Yoda comments; the notebook prompt also says "Yoda comments in code, plain English in markdown". Decide a single rule and harmonise. (Recommend: every code comment Yoda-flavoured; every markdown/docstring plain.)

### I7. `tests/test_llm_baseline.py` masquerades as live tests but is uncollectable (cross-ref C6)

### I8. `polimibot/eval/profiler.py:19` references `polimibot.game.types.Question`
That symbol doesn't exist (correct name: `GameQuestion`). Inside `if TYPE_CHECKING:`, so silent today, but breaks under any type-checker.

### I9. `scripts/eval_tools.py:75-80` — assigns `tool_answered` twice; first assignment touches non-existent attribute
```python
tool_answered = sum(
    1 for s in report_tool.samples
    if s.extras.get("tool") == "maths_tool"   # EvalSample has no .extras
)
tool_answered = sum(1 for s in report_tool.samples if s.confidence > 0.95)
```
First line raises `AttributeError` if reached; second line shadows it. Today the error is reachable because `report_tool.samples` is non-empty. Either add `extras` to `EvalSample` (and have `evaluate_strategy` propagate it) or delete the first assignment.

### I10. `OptConfig.use_compile` field defined but never wired
Cross-ref D1.

### I11. `polimibot/runner.py:130` uses ASCII `===` while other modules print Unicode `─` / `═`
Cosmetic.

### I12. `data/cache/.gitkeep` referenced by `.gitignore` but missing on disk
`.gitignore` whitelists `!data/cache/.gitkeep`, but only `data/runs/.gitkeep` and `data/eval/.gitkeep` actually exist. `polimibot/config.py:75` will create `data/cache/` at first run, which is fine; just an empty-directory tracking inconsistency.

---

## Documentation gaps

### Doc1. No top-level architecture overview in README.md
Strategy hierarchy (Random → Baseline → RAG → Tool → Agent → Ensemble → Tiered) and data flow (Adapter → Runner → Strategy → Logger → Gold-set → Evaluator) are not described anywhere in markdown. Newcomers must read 50 files.

### Doc2. No notebook quickstart
`PoliMillionaire.ipynb` opens with section 0 expecting a logged-in client; nothing tells the user where to set credentials, where to put the RAG index, or what `--mock` means.

### Doc3. `RUNTIME.api_url` is undocumented
Not mentioned in README. Easy fix once I5 is decided.

### Doc4. `calibration_from_gold_set` (cross-ref L2) silently expects run-log JSONL
Function name and parameter name are misleading.

### Doc5. `MathsTool` coverage is undocumented
The `_NORMALIZATIONS` table tells you it handles `%`, `to the power of`, `square root`, `times/divided by/plus/minus`, `×/÷`. Students should know it does NOT handle word-number conversions ("ten percent of two hundred"), inequalities, fractions ("two thirds of"), or geometry ("area of a circle of radius 5"). One paragraph in the docstring would prevent surprise.

### Doc6. No CHANGELOG / no record of what each "Stage" delivered
Recent commits hint at stages 7, 8, 9 but the README says "fill in as we go". One line per stage in README.md would close the gap.

### Doc7. Eval scripts have inconsistent CLI conventions
`eval_rag.py` has `--style`, `eval_tools.py` doesn't. `eval_ensemble.py` has `--rag-weight`, `eval_tiered.py` doesn't. Add a small "evaluation scripts" table to README.

---

## Strengths (preserve)

- **`Strategy` ABC + `StrategyInput` / `StrategyOutput` frozen dataclasses** (`polimibot/strategies/base.py`). Clean contract; trivial to add new strategies.
- **`RunLogger` JSONL design** (`polimibot/logging_utils.py`). Manifest / question / summary records, append-only with `os.fsync` on close, `NullLogger` no-op for tests, `load_jsonl` streams. Survives crashes; greppable.
- **`GameAdapter` boundary** (`polimibot/game/adapter.py`). Only place that imports `millionaire_client.models`. The rest of the package sees frozen `GameQuestion`/`AnswerOutcome`. Solid separation; the audit only had to read the adapter to understand the API surface.
- **AST-based `safe_eval`** (`polimibot/tools/calculator.py`). Whitelisted nodes, well-tested; precisely the right approach for executing model-emitted expressions safely.
- **`MathsTool`'s precision-over-recall stance**. Comment explicitly says "abstain rather than guess". Returns `None` on any uncertainty; the chain-of-responsibility hands control to the LLM. Good calibration hygiene.
- **`MockLLM` `<gold>` markers**. Lets every test assert deterministic correctness without GPU; reused cleanly across multiple test files.
- **Watchdog with daemon thread + non-killable runaway** (`polimibot/runner.py:46-70`). Acknowledges the impossibility of safely killing a CUDA call and chooses the right tradeoff (let it die, discard).
- **`warm_up` / `shutdown` lifecycle on `Strategy`**. Gives every strategy a chance to compile CUDA kernels before the per-question budget starts ticking; ensemble/tiered correctly deduplicate by object identity.
- **`EnsembleStrategy` soft-one-hot fallback**. When a sub-strategy doesn't return probs, it manufactures an honest distribution from `(chosen_index, confidence)` rather than fabricating uniformity.
- **`TieredStrategy` confidence-based escalation**. Optional, opt-in via `escalation_threshold`. Tags escalated answers in `extras["escalated_from"]` for measurement. Good experiment ergonomics.
- **`PromptStyle` enum + `build_messages`**. One enum value per experimental condition; swapping styles is a one-line change. Few-shot examples curated per category, with `rationale` field used only in CoT modes.
- **Calibration code (`_ece` + `compute_calibration`)** is correct in spirit and small enough to audit. The surrounding glue (L2/L3) is the issue, not the math.

---

## Decisions I need from you (the user) before fixing

1. **`score_options` (L1) — the most important question.** Two paths:
   - (a) **Keep the fast path**: rewrite `score_options` to use `add_generation_prompt=True` (no fake assistant turn), append `"Answer: "` as raw text, and probe the leading-space variant of letter token-ids when single-token (with bare-letter fallback). This preserves the speed advantage and the ECE story.
   - (b) **Drop the fast path**: have BaselineLLMStrategy and RAGStrategy go through `generate(...)` + `parse_answer`. Slower (one short generation per question) but eliminates an entire class of bugs.
   I lean (a). Confirm.

2. **`OptimisedLLMStrategy` (C7+L6+D1).** Delete? Or implement properly? My strong recommendation: **delete** unless you can name a concrete optimisation it should perform.

3. **`profiler.py` + `scripts/profile_strategies.py` (D2+L13+C4).** Two options:
   - Implement a real `profile_strategy(strategy, gold)` that fills `PhaseBreakdown` (instrument the LLM forward + decode + retrieval).
   - Delete both files.
   Implementing requires hooks inside `LLM.generate` / `LLM.score_options`. Confirm scope.

4. **`play_session` (C3+D3).** Fix the typing and export it properly, or fold it into `scripts/play_baseline.py` and remove from `runner.py`?

5. **`scripts/sweep_tiers.py` (C5).** Do you actually use this, or was it earlier exploration to be deleted? If kept, it needs concrete `easy/medium/hard` strategy choices wired in.

6. **`make_leaderboard` schema (C2).** Pick the canonical schema:
   - (a) match `EvalReport.save()`'s flat keys (`n_total`, `latency_p50`, `latency_p95`) — change `make_leaderboard`.
   - (b) introduce `EvalReport.to_dict()` matching `make_leaderboard`'s expected nested form (`n_samples`, `latency: {p50, p95}`) — change `EvalReport`.
   I prefer (a); minimal change, no behavioural drift.

7. **`calibration_from_gold_set` (L2).** Rename to `calibration_from_runs(path)` and document run-log expectation, or take `(gold_set, strategy)` and replay? (a) is simpler; (b) lets you draw reliability diagrams at eval time directly from a strategy.

8. **`parse_answer` fallback (L8).** When the LLM emits unparseable text, should the strategy:
   - (a) `is_abstain=True` and let the runner's `fallback_index` apply,
   - (b) random pick (adds run-to-run noise),
   - (c) keep current "default to A" (silently inflates accuracy on A-heavy gold sets).
   Recommend (a). Confirm.

9. **Yoda comment policy (I6).** Enforce throughout `polimibot/`, or relax to "code comments may be Yoda; markdown plain English"? The assignment's stylistic requirement is non-trivial — losing marks for inconsistency would be a shame.

10. **`tests/test_llm_baseline.py` (C6).** Rewrite against the current API or delete? `test_prompts.py` and `test_strategies.py` already cover much of the same ground.

11. **Hardcoded server URL (I5).** Add `POLIMI_API_URL` env override?

12. **README rewrite scope.** Should I rewrite README.md as part of the fix pass (covering install, env vars, strategy hierarchy, gold-set / RAG-index build, notebook quickstart, evaluation scripts)? If yes, I'll do it after the code fixes land. If no, scope it out.

Once you've answered, I'll apply the approved fixes one focused commit per file and then move on to Job 2 (notebook rebuild).
