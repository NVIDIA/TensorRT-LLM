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
  2026-07-03). One defect, five ids: `6419139` carries the root-cause
  analysis and `6418453` / `6419078` were marked as its duplicates;
  `6390244` reached the same mechanism independently; `6402018` is the same
  regression measured directly as `kv_cache_size` instead of downstream
  throughput, and a bisect on it found the same culprit commit
  `71613f9d8c`. Cross-link, **not** a fold: nvbug `6405760` was also
  declared a duplicate of this issue and #15632 did fix its first half, but a
  second, independent memory-footprint defect remained there — see
  [VLM vision tower loaded for text-only benchmarks](../model-definition/vlm-vision-tower-loaded-for-text-only-bench.md),
  whose NVBug records the split explicitly: part of the loss persisted after
  the DeepGEMM import-context fix in PR #15632. Adding `6405760` to this
  case's `nvbugs:` would erase that second contributor.
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
  closed unmerged by the culprit PR's co-author, citing the NVBug: #15632
  should already have fixed the KV-cache drop, and the remaining deferral was
  not worth it — a model without DeepGEMM usage pays an extra 32M of GPU
  memory per rank, which should not make a large difference to KV-cache size.
  The merged fix removed a ~0.5–1.2 GiB CUDA **context**; #16195 was chasing
  the ~32 MiB **workspace** that legitimately remained after it.
- **Symptom:** two observables, one cause. (a) `kv_cache_size` down 6–15%
  across dense and MoE, bf16/fp8/fp4, on H100 and B200 — `-15.25%` on
  `deepseek_r1_0528_fp4-bench-pytorch-float4-...-ep:1-gpus:4` (nvbug
  `6402018`). (b) Inference-time / total-token-throughput regressions on
  memory-constrained GPUs: `llama_v3.3_nemotron_super_49b_fp8` TP2 −9–10%
  (`6419139`), `qwen3.5_122b_a10b` +17% inference time / −15% throughput
  (`6418453`), `qwen3.5_122b_a10b` + `qwen3.5_397b_a17b_fp8` 13–17%
  (`6419078`) — all on RTX 6000D / RTX PRO 6000 Blackwell SE — and
  `k25_thinking_fp4_dep4_8k1k-con256` on GB200 4-GPU, output token
  throughput 4196 → 3819 tok/s, −8.99% (`6390244`). All surfaced as
  1.3.0rc19 → rc20 QA perf-sanity bars. Two independent bisects of the
  123-commit range on `6418453` / `6419078` found **no** defensible culprit
  and recorded the signature as SM120-only, which is exactly what this
  mechanism looks like from the outside.
- **Root cause:** `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py`
  called `_init_deep_gemm_pdl()` at **module level**, i.e. on
  `import tensorrt_llm`. That calls `deep_gemm.set_pdl()`, which instantiates
  DeepGEMM's `DeviceRuntime` (cuBLASLt handle + 32 MiB workspace tensor) and
  thereby creates a CUDA context — ~0.5–1.2 GiB including loaded modules,
  arch-dependent (~700 MB/context measured on GB200; 550 MiB measured on
  H100) — in **every process that imports tensorrt_llm**: the `trtllm-bench`
  parent, which never launches a kernel, and every MPI worker *on its
  default device*, before `torch.cuda.set_device()` in `base_worker.py`. Those
  contexts are resident when the KV-cache pool is sized from free GPU memory,
  so the pool shrinks by that much. The estimator is not wrong — per nvbug
  `6402018` the KV cache manager behaves correctly against its estimated
  budget; the footprint genuinely grew, and it grew *before* the
  free-memory probe. (Caveat on `6402018`: an alternative attribution to the
  dlfw-26.04 dependency bump #12643 was raised there from an outside-torch
  weight-loading delta of 7.66 → 12.07 GiB and was never formally retracted;
  the bug was nonetheless closed as verified at rc21 crediting #15632.)
- **How introduced:** commit `71613f9d8c`, PR #15402
  "[None][feat] DSv4 prep: MoE routing and backend support" — a DSv4
  preparation feature that added the bare `_init_deep_gemm_pdl()` call at
  import scope. Inert plumbing for the affected models: none of the
  regressing workloads use a DeepGEMM MoE path, which is why every bisect
  that reasoned about the *MoE* content of #15402 correctly dismissed it.
  On `6390244` the bisect landed instead on `a51931ad` (CuteDSL NVFP4 EPLB
  weight layout), which only pushed rank-0 GPU memory past a threshold and
  exposed the pre-existing issue.
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
  (7.66 → 12.07 GiB and 33.53 → 29.51 GiB across the rc19/rc20 boundary on
  `6402018`). To attribute it to import scope, check whether the bare import
  touches the device at all — `python -c "import tensorrt_llm"` then read the
  process's GPU memory via pynvml/`nvidia-smi`; on rc20 that import created a
  550 MiB context and on rc19 none (2×H100, deterministic across 9 runs, and
  disabling *only* the import-time call restored the available KV memory
  44.14 GiB exactly). Audit new module-level side effects with
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
  (this one sends exactly 512 uniform 1000/1000-token requests, so 48.33 →
  47.39 GiB / 1034176 → 1014112 KV tokens dropped max concurrent requests
  513 → 503, spilling 9 requests into a ~750-iteration tail at batch ≤ 9 =
  ~23 s = the observed +10%), where a sub-GiB shift crosses a batch boundary
  and reads as a large, arch-specific throughput regression — per-step GPU
  time at equal batch size was unchanged, and GPUs with more headroom lost
  the same memory and showed nothing.
