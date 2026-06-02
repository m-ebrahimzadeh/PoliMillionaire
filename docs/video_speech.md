# Video speech — *Who Wants to Be a PoliMillionaire?*

**Target length: 5:00 (strict).** Read at a steady pace and pause on the cells you point at.
Section time budgets are in each heading. Section 1 (Configure) and Section 4 (Compare)
carry the argument — keep them strong; trim Sections 0/2/3 first if you run long.

---

### INTRO (~20s)
Welcome — we're Future Millionaires. This is our notebook for an AI bot that plays
*Who Wants to Be a PoliMillionaire?* The core challenge: no single strategy works across
all six categories — recall, computation, and current events each need something different
— so we built a tailored architecture for each.

### SECTION 0 — Setup (~30s)
Section 0 is the one-time session setup. Three cells get us running: we mount Google Drive
for persistence, clone our GitHub repository, and install our `polimibot` package in
editable mode. The implementation lives in that package; this notebook is just the control
panel that configures, runs, and measures. Section 0.4 builds the RAG index — a FAISS dense
index plus a BM25 sparse sidecar, from Wikipedia. It runs once.

### SECTION 1 — Configure (~2:00)  ⟵ core
Section 1 is where the work lives. Our process had three steps. First, baseline evaluation
with no knobs enabled — **Phi-4** and **Qwen3-8B** were consistently the strongest
backbones. Second, prompt style — few-shot and zero-shot won; chain-of-thought hurt
accuracy, speed, and calibration at once, so we dropped it. Third, strategy composition,
which the rest of this section documents.

*[1.1]* loads the chosen model. *[1.2]* sets game parameters, including a speech mode: when
enabled, a Whisper transcriber turns spoken questions and options into text before the same
pipeline — the model underneath doesn't change, it's just an input adapter.

*[1.3]* is the only cell you edit between experiments. The key knobs: scoring — we score the
four option tokens directly in one forward pass instead of generating free text, which is
fast and well-calibrated; the composition flags, each turning on an architectural layer;
and the retrieval and news knobs.

*[1.3.1]* is our final per-category configuration, and the reasoning. **Entertainment and
Science** use Qwen3-8B with hybrid retrieval — dense plus BM25 — because these are broad
factual pools where semantic search and exact entity matching both help; Science drops
multi-query since the question is already specific. **History and Philosophy** use Phi-4,
few-shot, and a confidence gate: always-on RAG distracted the model, so we commit its
answer when it's confident and only call live Wikipedia when it's genuinely unsure.
**Maths** takes a different route — no article computes an answer, so we run three
deterministic tools first; if none fires, the model answers, and when it's unsure we have
it rewrite the question as a SymPy expression, which fixes its arithmetic slips. **Current
news** bypasses the confidence check entirely — the model can't know post-cutoff facts —
so every news question goes straight to the Guardian API.

*[1.4]* builds the retriever lazily, only if a RAG knob is on. *[1.5]* composes the final
strategy — the one place everything is wired together.

### SECTION 2 — Offline Evaluation (~40s)
Section 2 is where we measure. The gold set is our foundation: questions with confirmed
answers, harvested from live game logs. We also build a wrong set for error analysis. The
subset selector filters by category, difficulty, or a random sample. *[2.2]* runs the
evaluation through the full pipeline into an EvalReport; *[2.3]* saves it. *[2.4]* plots
per-category accuracy, and *[2.5]* the reliability diagram — confidence versus actual
accuracy.

### SECTION 3 — Online Evaluation (~20s)
Section 3 documents the live-game pipeline, kept here for reference. It plays sessions
against the live server and writes a log per game; those logs feed Section 2 to grow the
gold set. This is our data flywheel — every live game makes offline evaluation more reliable.

### SECTION 4 — Compare (~30s)  ⟵ punchline
Section 4 is the synthesis. *[4.1]* aggregates every saved report into one leaderboard;
*[4.3]* plots accuracy and latency across all strategies. And *[4.4]*, the per-category
heatmap, is the punchline: there is no single best strategy. The confidence gate dominates
History and Philosophy, the tool chain dominates Maths, and hybrid RAG serves Entertainment
and Science. Each category needed a different solution.

### CONCLUSION + FUTURE WORK (~25s)
In short: per-category specialisation clearly beats any universal strategy — each category
needed a different model, prompt style, and retrieval architecture. For future work, the
ensemble and tiered-routing knobs are implemented but were superseded by these targeted
approaches, and chain-of-thought showed promise but exceeded the time budget. A full
write-up of our architecture and experiments is on our documentation site, linked in the
notebook header — *m-ebrahimzadeh.github.io/PoliMillionaire*. Thank you.
