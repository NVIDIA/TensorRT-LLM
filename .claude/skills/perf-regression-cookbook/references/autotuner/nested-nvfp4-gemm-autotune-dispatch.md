---
id: case-nested-nvfp4-gemm-autotune-dispatch
type: regression-case
family: execution-and-graph
module: autotuner
maturity: full
regression_class: [host-work-added]
signals: [host-time-increase, gpu-idle-between-steps, throughput-drop]
subsystems: [autotuner, gemm-kernel, runtime-python]
introduced_via: [refactor]
phase: [any-phase]
patterns: [pattern-host-work-on-hot-path]
nvbugs: ["5758265"]
commits: ["15281de799b9"]
success_prs: [10503]
failed_prs: []
---

# Unified NVFP4 GEMM: two-level nested autotuning paid a double cache lookup per call

> Part of the [Autotuner regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5758265` · commit `15281de799b9` · PR #10503 —
  "[None][fix] Reduce host overhead for unified nvfp4 gemm tuning path." **The PR
  names no bug and states no symptom**; the link and all numbers come from the
  NVBug, which names the PR explicitly and records the merge.
- **Symptom:** MLPerf Inference Llama3.1-405B TP2PP2 throughput regressed
  between Nov 30 and Dec 17 on GB200 and B200 — **−8 %**, 182 → 172 tps/gpu,
  aarch64 GB200/B200. The bisect table from the NVBug is the useful
  artifact: `b024040df0` 186 tps/gpu (Nov 30) → `1a2118b8fe` 154 (0.83×) →
  `0a7a88e74e` 162 (0.87×) → `d600b9f851` 172 (0.92×) → `0dbf3948cc` 172 (Dec 17).
  Note the shape: a *staircase* of partial recoveries, not one step — several
  changes moved the number in the same window, which is why suspicion first
  fell on the async-PP-send PR (#9952) whose effect was being measured.
- **Root cause:** per the NVBug, **CPU bubbles** from a two-level nested tuning
  design: before the NVFP4 GEMM was unified, tuning only had to pick the
  fastest kernel of the CUTLASS backend; after it, an outer level selects
  among backends and an inner level picks the fastest kernel within each one.
  Tuning itself runs only at warmup, but **inference still walks the complete call
  stack**: even with the fastest backend and tactic cached, every call still
  performs two layers of cache lookups and `choose_one` calls. In the diff
  three costs are visible: bare backend strings were dispatched to ops that each
  ran their own `AutoTuner.get()`; `get_valid_tactics` was re-run on the dispatch
  path; and a **function-local CuteDSL import** executed on a cache miss. The
  tuning strategy was also forced to `INDEPENDENT`.
- **How introduced:** `refactor` — unifying the NVFP4 dense GEMM behind a
  configurable-backend interface. The design is right (it does pick the global
  optimum); the cost is that the abstraction's dispatch stayed on the per-call
  path. This is the classic shape of a host-overhead regression from a
  *correctness-neutral* restructuring, and it is why the bisect landed on a commit
  whose subject looks like a pure interface change.
- **Fix mechanism:** flatten the two levels into one. Tactics become
  `(backend, sub_tactic)` pairs so a single `choose_one` covers both decisions;
  runners are called directly instead of through the string-dispatched ops; the
  CuteDSL import is hoisted out of the cache-miss path; and the strategy flips
  from `INDEPENDENT` to `PARALLEL`. One file, +56/−69 — the fix is *smaller* than
  what it replaces. It also adds `assert len(self.allowed_backends) > 0` and, as a
  cost, removes the richer shape-naming error message.
- **Detection signal:** `python -c "…; print(R.tuning_config.distributed_tuning_strategy)"`
  on the runner — `INDEPENDENT` is the pre-fix tree. At runtime the signature is
  GPU idle between steps with flat kernel times and a host span that grows with
  the *number of GEMM call sites* rather than with token count. Because the cost
  is per call and not per token, low-concurrency / small-batch regimes show it
  worst; the NVBug's workaround was piecewise CUDA graph, which hides host
  overhead rather than removing it — so a "piecewise makes it go away" result is
  evidence *for* this class, not against it.
- **Prevention/guard:** the added `assert` guards configuration, not overhead.
  There is no test asserting that a cached-tactic dispatch performs one lookup.
  Rule worth carrying: when an autotuner gains a level, measure the **cached**
  path, not the tuning path — the tuning cost is at warmup and visible, while the
  residual lookup cost is per call and invisible until a host-bound workload
  reports it.
- **Generalizes to:** `pattern-host-work-on-hot-path`; carries to every
  backend-selection abstraction (MoE backends, attention backends, quant GEMM
  variants) where a warmup-time decision is re-resolved per invocation, to
  function-local imports on any path a cache miss can reach, and to
  `INDEPENDENT`-vs-`PARALLEL` tuning strategy choices under TP/PP.
