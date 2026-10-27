---
id: case-deepgemm-pdl-import-time-cuda-context
type: regression-case
family: memory-and-capacity
module: gemm-and-quantization
maturity: full
regression_class: [memory-footprint-regression]
signals: [kv-capacity-drop, memory-usage-increase, throughput-drop, perf-ci-bar-failure]
subsystems: [runtime-python]
introduced_via: [new-feature]
phase: [any-phase]
patterns: [pattern-unaccounted-startup-residency]
nvbugs: ["6390244", "6402018", "6418453", "6419078", "6419139"]
commits: ["e8e1ade1c6e3"]
success_prs: [15632]
failed_prs: [15985, 16195]
---

# Import-time DeepGEMM PDL init creates a CUDA context, shrinking the KV pool

> Part of the [GEMM & quantization regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbugs `6419139` / `6418453` / `6419078` / `6390244` /
  `6402018` · commit `e8e1ade1c6e3` · PR #15632 —
  "[TRTLLM-12950][perf] DSv4 follow-up: DeepGEMM and MegaMoE" (merged
  2026-07-03). PR #15985 ("[https://nvbugs/6419139][test] Guard against
  CUDA context creation at import") describes the root cause; PR #16195
  ("[https://nvbugs/6402018][fix] ...") targets the same mechanism measured
  directly as `kv_cache_size`. Cross-link, **not** a fold: PR #15985 also
  lists nvbug `6405760` as sharing this root cause, but PR #16250 states that
  regression "persisted after the deep_gemm import-context fix (#15632)" — a
  second, independent memory-footprint defect — see
  [VLM vision tower loaded for text-only benchmarks](../model-definition/vlm-vision-tower-loaded-for-text-only-bench.md).
  Adding `6405760` to this case's `nvbugs:` would erase that second
  contributor.
- **Failed attempts:** PR #15985 — the same lazy-PDL fix, later rebased down
  to only its guard test `tests/unittest/others/test_import_side_effects.py`
  (`test_deep_gemm_pdl_configuration_is_lazy`, no GPU;
  `test_import_creates_no_cuda_context`, pynvml/GPU-gated) · closed unmerged
  as superseded — "Closing this PR because the same fix was included in 15632
  which has since merged", so **the regression test never landed and is still
  absent from `main`**. PR #16195 — went one step further than the merged
  fix: relocated the one-shot helper to
  `tensorrt_llm/_torch/utils.py::configure_deep_gemm_pdl()`, dropped the
  eager call from `PyTorchModelEngine.__init__`, and invoked it lazily from
  the `DeepGemmFusedMoE` / `MegaMoEDeepGemm` constructors so a workload that
  never instantiates a DeepGEMM MoE would not reserve the workspace at all ·
  closed unmerged. The merged fix removed a ~0.5–1.2 GiB CUDA **context**;
  #16195 was chasing the ~32 MiB **workspace** that legitimately remained
  after it.
- **Symptom:** two observables, one cause. (a) A 6–15% `kv_cache_size`
  regression in 1.3.0rc20 (PR #16195; its perf table: bad 13.17, good 15.54,
  after fix 15.85). (b) Inference-time / throughput regressions on
  memory-constrained GPUs: ~10% on RTX 6000D for
  `llama_v3.3_nemotron_super_49b_fp8`, 1.3.0rc19 → rc20 (PR #15985). Because
  GPUs with more headroom lose the same memory but show no regression, the
  signature looks GPU-specific from the outside (PR #15985).
- **Root cause:** `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py`
  called `_init_deep_gemm_pdl()` at **module level**, i.e. on
  `import tensorrt_llm`. That calls `deep_gemm.set_pdl()`, which instantiates
  DeepGEMM's `DeviceRuntime` (cuBLASLt handle + 32 MiB workspace tensor) and
  thereby creates a CUDA context — ~0.5–1.2 GiB including loaded modules,
  arch-dependent (550 MiB measured on H100, PR #15985) — in **every process that imports tensorrt_llm**: the `trtllm-bench`
  parent, which never launches a kernel, and every MPI worker *on its
  default device*, before `torch.cuda.set_device()` in `base_worker.py`. Those
  contexts are resident when the KV-cache pool is sized from free GPU memory,
  so the pool shrinks by that much. The estimator is not wrong — the
  footprint genuinely grew, and it grew *before* the free-memory probe.
- **How introduced:** commit `71613f9d8c`, PR #15402
  "[None][feat] DSv4 prep: MoE routing and backend support" — a DSv4
  preparation feature that added the bare `_init_deep_gemm_pdl()` call at
  import scope (PR #15985). Inert plumbing for the affected models: none of
  the regressing workloads use a DeepGEMM MoE path, so reasoning about the
  *MoE* content of #15402 dismisses it; the culprit is the import-scope
  side effect, not the feature.
- **Fix mechanism:** PR #15632 deleted `_init_deep_gemm_pdl()` and its
  module-level call from `torch_custom_ops.py`, and added an idempotent
  `_configure_deep_gemm_pdl()` (guarded by `_DEEP_GEMM_PDL_CONFIGURED`,
  reading `TRTLLM_ENABLE_PDL`, default `1`) to
  `tensorrt_llm/_torch/pyexecutor/model_engine.py`, called as the first
  statement of `PyTorchModelEngine.__init__` — i.e. after
  `torch.cuda.set_device()`, on the right device, and never in a process that
  does not build a model engine.
- **Detection signal:** the KV pool shrinks while free GPU memory and config
  are unchanged, and the loss shows up in the *outside-torch* term of the
  memory-usage profile, not in weights. Diff two builds with
  `grep -n "Memory used after loading model weights (outside torch)\|Estimated max memory in KV cache" <bench log>`
  across the rc19/rc20 boundary. To attribute it to import scope, check
  whether the bare import touches the device at all — `python -c "import tensorrt_llm"` then read the
  process's GPU memory via pynvml/`nvidia-smi`; on rc20 that import created a
  550 MiB context and on rc19 none (PR #15985: 2×H100, available KV memory
  44.14 GiB rc19 vs 43.59 GiB rc20, deterministic across 9 runs, and
  disabling *only* the import-time call restored 44.14 GiB exactly). Audit
  new module-level side effects with
  `git grep -n 'set_pdl' tensorrt_llm/` and, more generally, by looking for
  bare calls at import scope in files that touch a CUDA runtime.
- **Prevention/guard:** the guard was written and lost. PR #15985 added
  `tests/unittest/others/test_import_side_effects.py` —
  `test_deep_gemm_pdl_configuration_is_lazy` (asserts `import tensorrt_llm`
  does not configure DeepGEMM PDL; needs no GPU) and
  `test_import_creates_no_cuda_context` (asserts the import creates no CUDA
  context) — but the PR was closed once #15632 merged, so **no test on `main`
  forbids an import-time CUDA context today**; verify with
  `git grep -n 'test_import_creates_no_cuda_context'`, which returns nothing.
  Re-landing that file is the cheapest guard for this whole class. The
  existing bars are indirect: the perf-sanity `kv_cache_size` metric catches
  the memory loss, and only on GPUs where it is large enough to trip the 5%
  threshold.
- **Generalizes to:** `pattern-unaccounted-startup-residency` —
  memory reserved before the capacity planner probes free memory comes off
  the KV pool 1:1, with a correct estimator throughout. Carries to: any
  module-level initializer under `tensorrt_llm/` that constructs a
  cuBLAS/cuBLASLt/NCCL handle, a JIT runtime, or calls `torch.cuda.*` at
  import — worst in helper processes that never launch a kernel and in MPI
  workers before `torch.cuda.set_device()`, where the context also lands on
  the *wrong* device; new third-party runtimes whose handle construction
  implicitly creates a context; and perf cases tuned to an exact concurrency
  (this one sends exactly 512 uniform requests, so 48.33 → 47.39 GiB on
  RTX 6000D dropped max concurrent requests 513 → 503, spilling 9 requests
  into a ~750-iteration low-batch tail = the observed +10%, per PR #15985),
  where a sub-GiB shift crosses a batch
  boundary and reads as a large, arch-specific throughput regression — GPUs
  with more headroom lost the same memory and showed nothing.
