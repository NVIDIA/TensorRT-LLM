---
id: case-autodeploy-gc-gen0-threshold-unforwarded
type: instability-case
family: runtime-determinism
module: scheduler-and-executor
maturity: full
instability_class: [gc-pause-on-decode-path]
signals: [midrun-stall, rep-to-rep-variance]
subsystems: [runtime-python, scheduler-executor]
introduced_via: [incomplete-coverage]
phase: [decode]
patterns: [pattern-gc-pause-on-decode-path]
nvbugs: []
commits: ["1c2c5c3fffa4"]
success_prs: [14218]
failed_prs: []
---

# AutoDeploy never forwarded `garbage_collection_gen0_threshold`, so Python GC stalled decode

> Part of the [Scheduler & executor instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `1c2c5c3fffa4` · PR #14218 — AutoDeploy: forward
  `garbage_collection_gen0_threshold` to PyExecutor. No NVBug; the PR is filed
  against GitHub issue **#13561** ("[AutoDeploy] BTK findings: Nano doesn't
  scale well for tp>1", still open — this PR is one contributing fix, not its
  closure), whose own hypothesis was host-boundness.
- **Symptom (variance signature):** on heavy models, **gen-2 collections
  periodically stalled decode for 0.5–1 s**, producing large ITL spikes and
  substantial run-to-run throughput variance — e.g. ~286 vs ~342 tok/s/user on
  Nemotron-3-Nano-30B-A3B-FP8 TP=4, i.e. a ~20 % spread on identical config.
  Surfaced from a scaling investigation (tok/s flat across TP=1/2/4/8), not from
  a perf CI bar: the variance was large enough that the mean read as
  "host-bound" rather than as noise.
- **Root cause:** the PyTorch backend passes
  `llm_args.garbage_collection_gen0_threshold` (`TorchLlmArgs` default 20000)
  into `PyExecutor`, which wraps the executor loop in `customized_gc_thresholds`
  and calls `gc.set_threshold(gen0_threshold)`. AutoDeploy's
  `create_autodeploy_executor()` **did not forward the field**, so `PyExecutor`
  saw `None` and the context manager became a **no-op** — not an error, not a
  warning, just the tuning silently absent. AutoDeploy therefore ran at
  CPython's default gen-0 threshold of **700 (~28× more frequent)**, and each
  gen-0 sweep is a chance to promote and eventually run a gen-2 collection over
  an object graph that is enormous on these models (FX graph + CUDA-graph
  wrappers + mamba/SSM caches + MoE state).
- **How introduced:** `incomplete-coverage` — AutoDeploy is a second executor
  construction path alongside the PyTorch backend's, and it enumerates the
  `PyExecutor` kwargs by hand. The field was added to one path and not the
  other. Nothing about the omission is visible at the call site: the parameter
  simply defaults.
- **Fix mechanism:** one line — forward `ad_config.garbage_collection_gen0_threshold`
  into the `PyExecutor(...)` construction. The existing `TorchLlmArgs` default is
  honored and users can still override from YAML. Measured on Nano V3 1k/1k
  concurrency 1 TP4: t/s/u 295 → 346, output tok/s 282 → 336.
- **Detection signal:** periodic sub-second decode stalls with **no GPU work in
  the gap** — the profile shows the device idle while the host is inside
  `gc.collect`; `py-spy dump` during a stall lands in
  `gc`/`collect` rather than in a kernel launch. The config-level check is
  cheaper and is the one to run first: `gc.get_threshold()` inside the executor
  process should report the configured value, not `(700, 10, 10)`. Statically,
  `grep -n 'garbage_collection_gen0_threshold' tensorrt_llm/_torch/pyexecutor/py_executor.py
  tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py` — every construction site
  of `PyExecutor` must appear.
- **Prevention/guard:** **the gap is still open.** The PR's test change adds
  `garbage_collection_gen0_threshold: Optional[int] = None` to the
  `MockPyExecutor` dataclass in
  `tests/unittest/auto_deploy/singlegpu/shim/test_create_ad_executor.py` — which
  only makes the mock *accept* the kwarg. Verified at the merge commit: there is
  **no assertion that the value is forwarded** (the file asserts
  `resource_governor_queue`, `guided_decoder`, `max_num_sequences`,
  `vocab_size_padded` — not this field), so re-dropping the forward would not
  fail this test. That is the guard to add: assert the mock received the
  configured threshold, and better, assert it for *every* field the PyTorch path
  forwards, so a hand-enumerated kwarg list cannot drift again. The general form
  of the defect is worth stating as a review rule: **a performance tuning knob
  that defaults to "off" cannot report its own absence.** Prefer a required
  parameter, a shared construction helper, or a startup log line stating the
  effective GC thresholds.
- **Generalizes to:** `pattern-gc-pause-on-decode-path`; carries to any second
  executor / runtime construction path that enumerates kwargs by hand
  (AutoDeploy, disagg workers, benchmark harnesses, spec-decode drafters), to
  any `Optional[...] = None` tuning field whose `None` branch is a silent no-op,
  and to every host-side pause mechanism on the decode path — GC, allocator
  trims, `cudaFree` under an expanding pool, Python finalizers — where the cost
  is invisible in a GPU-only profile.
