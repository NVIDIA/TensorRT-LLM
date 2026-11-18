# Regression Cookbook — JIT & warmup

This module is the startup work that is supposed to pay for the run: the
ModelEngine warmup orchestration, the trtllm-gen FMHA JIT warmup grids, and the
`maybe_compile` / `maybe_compiled_*` helpers that wrap `torch.compile` around
in-tree ops. Its defining property is that nothing here is wrong *at rest* —
warmup either covered the shape the run requests or it did not, and the
uncovered case pays a compile inside a served or measured iteration. Regressions
therefore look like a mean shift, not like a slow kernel: on a deterministic
perf case every rep hits the same uncovered shape, so a one-time cost reads as a
permanent level change. First thing to check: which shapes the run actually
requests versus which the warmup grid enumerates — and whether warmup ran at all
(it has been gated behind conditions the workload did not satisfy).

## Recurring patterns in this module

- **Warmup coverage gap** — JIT/compile/capture cost lands in measured or served
  iterations because the warmed set does not cover live traffic. Four distinct
  shapes appear here: no warmup at all; a grid too *sparse* for a legitimate
  runtime band; a grid *shrunk* by a protective cap a prior fix added, which
  pulled the warmed set below what real traffic requests (so the regression hits
  a broad row set — itself the signal); and a compiled op reachable only from a
  branch (chunked prefill) that warmup never enters. Enumerate the shape space
  the workload can reach, not the one the author expected.
  _(Instances: trtllm-gen FMHA JIT warmup round 1 and the round-2 densified
  grid — read them as a pair, the second is the first fix's residual hole; the
  warmup token-cap revert; the MLA chunked-prefill `maybe_compiled_cat`; and the
  ModelEngine warmup orchestration gaps. The Helix context-parallel warmup
  sibling was removed on 2026-08-12: its fix PR #11460 describes a reporting
  discrepancy (the first iteration's median TPOT includes cudagraph warmup
  time), not a perf regression.)_
- **Host work on the hot path** — a compile *helper* can be the defect rather
  than a missing warmup entry: calling `torch.compile` inside the wrapper means
  every invocation re-enters the compiler, so no amount of warmup helps. Kernel
  names and durations are unchanged; compare host spans.
  _(Instance: `maybe_compile` calling torch.compile inside the wrapper.)_
- **Unaccounted startup residency** — warmup is also a *memory* phase. Buffers
  held resident across it (MoE workspaces kept into autotuning) shrink the free
  pool the KV estimate is computed from, so the symptom is capacity rather than
  time and is invisible in any steady-state profile.
  _(Instance: the ModelEngine warmup orchestration gaps.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [`maybe_compile` called torch.compile inside the wrapper](maybe-compile-recompiles-every-call.md) | flat per-call host overhead across MLA attention, layer norm and DSA; the PR states no number | host-work-added |
| [`maybe_compiled_cat` torch.compile stall on the chunked-prefill MLA path](mla-chunked-prefill-maybe-compiled-cat-warmup.md) | first-use recompile on the chunked-prefill MLA context path; a mean shift, not variance; the PR states no number | warmup-jit-gap |
| [ModelEngine warmup gated on torch.compile, missing the (1,0) shape, holding MoE workspaces](model-engine-warmup-orchestration-gaps.md) | filed as a refactor, so the PR states no symptom and no number | warmup-jit-gap, memory-footprint-regression |
| [trtllm-gen FMHA warmup grid too sparse for a narrow seqlen band (round 2)](trtllmgen-fmha-densify-grid.md) | DSR1 `output_token_throughput` 992 → 615.3 (#15279); TTFT P99 882 ms → 25 s (#15472) | warmup-jit-gap |
| [trtllm-gen FMHA kernels JIT-compile mid-serving (round 1: no warmup)](trtllmgen-fmha-jit-warmup.md) | 7–9 s per kernel compile, up to 8 runtime compilations (#14851) | warmup-jit-gap |
| [Warmup token cap left large-token shapes untuned at runtime — revert](warmup-token-cap-revert.md) | shapes between 8192 and `max_num_tokens` untuned; the revert PR states no number | warmup-jit-gap |
