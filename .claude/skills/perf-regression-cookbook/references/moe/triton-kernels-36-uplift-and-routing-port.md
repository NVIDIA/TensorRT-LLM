---
id: case-triton-kernels-36-uplift-and-routing-port
type: regression-case
family: kernel-and-fusion
module: moe
maturity: full
regression_class: [dependency-regression]
signals: [slower-kernel-in-trace, throughput-drop]
subsystems: [moe, build-dependency, gemm-kernel]
introduced_via: [pre-existing-gap]
phase: [any-phase]
patterns: [pattern-pinned-dep-holds-kernel-perf]
nvbugs: ["5877121"]
commits: ["e44df9e21a5f"]
success_prs: [12102]
failed_prs: []
---

# GPT-OSS Triton MoE slower than vLLM Marlin — the pinned triton_kernels carried the slow kernel

> Part of the [MoE regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5877121` · commit `e44df9e21a5f` · PR #12102 —
  "[TRTLLM-10820][infra] Update dependencies to align with NGC PyTorch 26.02
  stack". **The PR names no NVBug**; the link is established from the bug side:
  per the NVBug, the slow kernel was removed upstream in
  https://github.com/triton-lang/triton/pull/8483, so Triton 3.6 is fine, and
  the Triton 3.6 upgrade landed in PR #12102, so 1.3.0rc10 has the new kernels.
  A PR→bug lookup finds nothing here; only the bug records the resolution.
- **Symptom:** the TRT-LLM TRITON MoE backend runs 10% slower than vLLM Marlin
  for gpt-oss-120b on H100. This is a **gap against a competitor at a fixed
  version**, not a drop against a previous TRT-LLM build — no bisect, no
  culprit commit in TRT-LLM.
- **Root cause:** the slowness lived in the **pinned third-party kernel
  library**, not in TRT-LLM. `triton_kernels` at the pinned 3.5.1 contained a slow
  routing kernel that upstream Triton deleted in triton-lang/triton#8483; TRT-LLM's
  Triton MoE backend called it through `routing()` /
  `routing_from_bitmatrix()`. So no amount of TRT-LLM-side profiling of TRT-LLM
  code explains the deficit, and the fix is a version move.
- **How introduced:** `pre-existing-gap` — the backend was never faster; it
  inherited whatever the pinned kernel library shipped. Worth stating plainly
  because framing the bug as a percentage gap against vLLM invites a bisect that
  cannot succeed.
- **Fix mechanism:** the dependency uplift to the NGC PyTorch 26.02 stack —
  torch 2.9.1 → 2.10.0, **triton 3.5.1 → 3.6.0**, TensorRT 10.14.1 → 10.15.1,
  CUDA 13.1.0 → 13.1.1 — plus the API port the uplift forces. triton_kernels 3.6.0
  **deletes** `routing()`, `routing_from_bitmatrix()` and `_routing_clear_bitmatrix`,
  makes `topk()` return a bitmatrix, and moves `RoutingData` / `GatherIndx` /
  `ScatterIndx` into `matmul_ogs`. PR #12102 therefore rewrites
  `TritonEPRouter.__call__` — keeping a **local copy of the deleted kernel** and
  recomputing `mask_metadata` *after* EP pruning — and updates `mxfp4_moe.py`,
  plus a `scales.value().clone()` in `fp8Op.cpp`. Two things to carry forward: a
  dep uplift that fixes perf is still a port, and the port has perf-relevant
  semantics of its own (the recomputed `mask_metadata` is not cosmetic — stale
  metadata after pruning changes which experts are gathered).
- **Detection signal:** version-first, before any profiling. In the container:
  `cat $(python -c "import triton_kernels,os;print(os.path.dirname(triton_kernels.__file__))")/VERSION`
  and `python -c "import triton_kernels.routing"` — the import **succeeds only on
  a pre-fix (3.5.1) tree**, because 3.6.0 removed the module's entry points. For
  the general case: when a backend is "N % slower than <other framework>" and
  both frameworks call the same third-party kernel library, diff the *pinned
  versions* of that library first.
- **Prevention/guard:** the PR adds **7 SKIP waives** (nvbugs 5996776, 5983320,
  5983283) — the opposite of a guard. That is the honest lesson here: a
  stack-wide uplift lands with known collateral, and the waives are the record of
  what the uplift broke. There is no test pinning the routing port's behaviour, so
  a future triton uplift can re-break `mask_metadata` recomputation silently.
- **Generalizes to:** `pattern-pinned-dep-holds-kernel-perf`; carries to every
  backend whose kernels come from a pinned wheel (triton_kernels, flashinfer,
  DeepGEMM, cutlass python packages) — check the pin before the profile — and to
  the reverse direction, `pattern-fusion-pattern-drift`, where the *same* uplift
  breaks a fusion pattern and costs perf. Both directions of a dep bump are live
  at once; the uplift that closes one bug's gap is another bug's culprit commit.
