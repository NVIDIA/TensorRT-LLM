---
id: case-fmha-dispatcher-sm-count-query
type: regression-case
family: execution-and-graph
module: attention-fmha
maturity: full
regression_class: [host-work-added]
signals: [host-time-increase, perf-ci-bar-failure]
subsystems: [attention-kernel]
introduced_via: [pre-existing-gap, dep-bump]
phase: [any-phase]
patterns: [pattern-host-work-on-hot-path]
nvbugs: ["6368480"]
commits: ["2ee936ecb102"]
success_prs: [15611]
failed_prs: []
---

# FMHA dispatcher queries SM count from the CUDA driver every step

> Part of the [Attention & FMHA regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6368480` · commit `2ee936ecb102` · PR #15611 —
  "Cache the SM count once in FmhaDispatcher's constructor and reuse the
  cached…".
- **Symptom:** host-side perf regression on llama-3.3-70B FP4 TP4; surfaced
  on the perf test `llama70b_fp4_tp4_512_32-con512_iter10_512_32`, where the
  repeated driver round-trips show up as host overhead (per the fix PR).
- **Root cause:** `FmhaDispatcher::isSupported()` and `FmhaDispatcher::run()`
  sit on the per-iteration FMHA dispatch hot path, and each invocation called
  `tensorrt_llm::common::getMultiProcessorCount()` — a
  `cudaDeviceGetAttribute()` round-trip into the CUDA driver — to fill
  `tllmRunnerParams.mMultiProcessorCount`.
- **How introduced:** the per-call driver queries pre-date the fix (both call
  sites are visible in the diff being replaced); the fix PR states the
  round-trips "caused the perf regression observed in PR12643"
  ("[TRTLLM-11715][infra] Upgrade dependencies for dlfw 26.04 stack", merged
  2026-06-28 as `b6d186af57`). Per the NVBug, the cost was hidden on `main`
  but exposed on that PR's branch — i.e. the dependency-stack upgrade raised
  the unit cost of a pre-existing per-step host call until it tripped the bar
  in PR12643's pre-merge job, and the fix (2026-06-26) actually landed on
  `main` two days *before* the dep bump did.
- **Fix mechanism:** cache the SM count once: new private member
  `mMultiProcessorCount` in `cpp/tensorrt_llm/kernels/fmhaDispatcher.h`,
  initialized from `getMultiProcessorCount()` in the constructor initializer
  list; both call sites now read the cached value. SM count is fixed per
  process, so caching is safe.
- **Detection signal:** an nsys CUDA API trace shows repeated
  `cudaDeviceGetAttribute` calls inside every forward step instead of only at
  startup; audit hot dispatch paths with
  `grep -rn "getMultiProcessorCount()" cpp/tensorrt_llm/kernels/` — any hit
  inside a per-call `run()`/`isSupported()` body is suspect.
- **Prevention/guard:** none added by the fix PR (an automated repair-bot
  fix). Gap: no assert/lint that flags CUDA driver/attribute queries on
  per-iteration paths; review checklist item — device properties are
  process-constant, query them at construction and cache.
- **Generalizes to:** `pattern-host-work-on-hot-path` — hidden host work
  (here a driver round-trip) executed once per dispatch instead of once per
  process. Carries to: other device-property queries
  (`cudaGetDeviceProperties`, `cudaGetDevice`) inside kernel-runner
  `run()`/`isSupported()` bodies; per-step env-var reads or config parsing in
  the executor loop; Python-side per-request attribute recomputation of
  process-constant values; dependency bumps that raise the unit cost of a
  pre-existing per-step host call until it trips the perf bar.
