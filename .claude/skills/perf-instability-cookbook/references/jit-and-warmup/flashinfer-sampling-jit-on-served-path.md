---
id: case-flashinfer-sampling-jit-on-served-path
type: instability-case
family: warmup-and-jit
module: jit-and-warmup
maturity: full
instability_class: [warmup-gap]
signals: [midrun-stall, cross-rank-hang, rep-to-rep-variance]
subsystems: [sampler, model-engine]
introduced_via: [incomplete-coverage]
phase: [decode]
patterns: [pattern-jit-on-hot-path]
nvbugs: []
commits: ["b40c29bb333c"]
success_prs: [17286]
failed_prs: []
---

# flashinfer sampling module JIT-built on the served path when `cuda_graph_config: null`

> Part of the [JIT & warmup instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `b40c29bb333c` · PR #17286 — Warm up flashinfer
  sampling module during engine warmup. No NVBug (`[None]` PR title); diagnosed
  from a production 16-node GB300 disaggregated DeepSeek-V4 run.
- **Symptom (variance signature):** mid-serving stalls of **98.2 s and 58.1 s**
  in a single unpacked run, escalating to `Hang detected` → `MPI_Abort`: the
  unpatched 16-node run **died at iteration 9**, while two patched 4-node runs
  reached 2255 and 2287 iterations with zero hang events. The failure is
  outcome-variable, not a fixed cost: whether the job survives depends on
  whether the build crosses the 300 s hang-detector threshold, and the patched
  runs still show 19–20 s stalls, so the cost is paid **more than once** rather
  than once per process. Generation workers in the same runs were unaffected.
- **Root cause:** flashinfer ships its sampling kernels as source and JIT-builds
  them with `nvcc` on first use, taking **~80–105 s with a cold cache** (the
  cache lives at `/root/.cache/flashinfer/...` inside the container, so it is
  cold on every job). The only thing that exercised the non-greedy sampler
  during warmup was the advanced-sampling CUDA-graph capture pass
  (`_run_capture_pass(force_non_greedy=True)`), so with **`cuda_graph_config:
  null`** nothing triggered the build until the first non-greedy request was
  actually served — inside the executor loop. While one rank compiles it never
  launches its MoE all-to-all dispatch, so peer ranks GPU-spin on its completion
  flag and block in `tp_gather`; the arming of the hang detector then converts a
  stall into a whole-job abort. The captured stack is unambiguous:
  `sample_from_logits_op` → `top_k_mask_logits_op` →
  `flashinfer/sampling.py:get_sampling_module` → `jit/core.py:build_and_load` →
  `cpp_ext.py:run_ninja` blocked in `nvcc` >300 s.
- **How introduced:** `incomplete-coverage` — warmup for these kernels existed,
  but only as a **side effect** of a different feature's warmup step. Gen
  workers set `cuda_graph_config.batch_sizes` and so built the module during
  warmup for free; ctx workers running `cuda_graph_config: null` had no coverage
  at all. Nothing declared the dependency, so disabling CUDA graphs silently
  removed a sampler warmup.
- **Fix mechanism:** call `get_sampling_module()` during engine warmup, where
  the hang detector is not armed and no peer is waiting on this rank. It covers
  every sampling kernel the sampler uses and is `@functools.cache`d upstream, so
  it is a no-op once built. Deliberately **not** gated on spec-decoding: the
  plain `TorchSampler` non-greedy path reaches the same kernels, so gating would
  leave that case exposed.
- **Detection signal:** the `ninja: Entering .../cached_ops/sampling` block
  appearing **after** the first warmup forward instead of before it — that
  ordering is the whole signal, and it is visible in any run's stdout:
  `grep -n 'ninja: Entering' <log>` and compare the line number against the
  first warmup-forward marker. When a hang detector fires, read the stack for a
  `jit`/`build_and_load`/`run_ninja` frame before concluding "collective hang";
  the peers blocked in `tp_gather` are victims, not the cause (the same
  entry-vs-exit misread as the a2a barrier-spin trap). Config-level precursor:
  any worker with `cuda_graph_config: null` plus non-greedy sampling.
- **Prevention/guard:** the PR verifies idempotency and the no-flashinfer
  branch, but there is **no test that asserts the build happens during warmup**
  — the guard is behavioral, validated on cluster runs only. The rule this case
  argues for: **a warmup step must be requested explicitly, never inherited
  from another feature's warmup.** When any config flag can disable a warmup
  path (`cuda_graph_config: null`, `enable_autotuner: false`, eager mode),
  enumerate what that flag silently stops warming. A cheap structural guard is
  to assert, at the end of engine warmup, that every `@functools.cache`d JIT
  entry point the sampler can reach is already populated. Note also the
  validation's own caveat, stated in the PR: the A/B is **not controlled**
  (4-node ctx1+gen1 @ concurrency 224 vs 16-node ctx4+gen1 @ 1418), so the
  iteration counts are directional, not a measured speedup.
- **Generalizes to:** `pattern-jit-on-hot-path`; carries to every JIT-on-first-use
  dependency reached only from a conditional warmup step (flashinfer sampling,
  DeepGEMM, trtllm-gen FMHA cubins, `torch.compile` branches behind a config
  flag), to any container whose JIT cache is under `$HOME` and therefore cold
  every job, and to any hang-detector threshold that a legitimate one-time build
  can exceed — where the detector's abort turns a slow start into a lost run.
