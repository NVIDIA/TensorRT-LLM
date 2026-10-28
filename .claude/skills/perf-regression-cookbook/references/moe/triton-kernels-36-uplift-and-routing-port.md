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

# Triton MoE perf held back by the pinned triton_kernels — the fix is the 3.6.0 uplift

> Part of the [MoE regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5877121` · commit `e44df9e21a5f` · PR #12102 —
  "[TRTLLM-10820][infra] Update dependencies to align with NGC PyTorch 26.02
  stack". **The PR names no NVBug**, so a PR→bug lookup finds nothing; the
  public record is the diff.
- **Symptom:** a TRITON MoE backend perf deficit with **no culprit commit in
  TRT-LLM** — a gap, not a drop against a previous TRT-LLM build, so there is
  nothing to bisect.
- **Root cause:** the kernels lived in the **pinned third-party kernel
  library**, not in TRT-LLM. TRT-LLM vendors `triton_kernels` (pinned with
  `triton==3.5.1` in `requirements.txt`), and its Triton MoE backend called
  that library's `routing()` / `routing_from_bitmatrix()`. When the kernel
  library is the bottleneck, no amount of profiling of TRT-LLM code explains
  the deficit, and the fix is a version move.
- **How introduced:** `pre-existing-gap` — the backend was never faster; it
  inherited whatever the pinned kernel library shipped. Worth stating plainly
  because framing such a gap as a percentage against another framework invites
  a bisect that cannot succeed.
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
