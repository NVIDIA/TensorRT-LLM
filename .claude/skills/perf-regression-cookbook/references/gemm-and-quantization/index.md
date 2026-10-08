# Regression Cookbook — GEMM & quantization

This module is the dense GEMM and quantization layer: the FP8 / NVFP4 quantize
kernels and their TE-equivalent alternatives, and the DeepGEMM integration that
wraps a third-party GEMM library. Two rather different failure shapes live here.
One is the ordinary kernel comparison — a quantize kernel that is simply slower
than the alternative implementation on some arch, which only microbenchmarks
across a shape sweep will show. The other is a *side effect of the integration
rather than of the math*: module-import work that creates a CUDA context before
the capacity planner measures free GPU memory, which costs KV blocks and never
appears in a steady-state profile. First thing to check: whether the reported
loss is time or capacity, because the two point at opposite ends of the file.

## Recurring patterns in this module

- **Kernel swap regressed / below an alternative** — a quantize kernel loses to
  the equivalent library implementation across a whole shape sweep, with no
  culprit commit and no end-to-end number attached. That is still a real
  deficit; record it from the microbenchmark and do not manufacture an e2e delta.
  _(Instance: FP8 per-tensor quantization slower than TE's on H100.)_
- **Unaccounted startup residency** — the KV pool is sized from *free* GPU
  memory, so anything resident before that probe comes off capacity 1:1. An
  import-time CUDA context is the purest form: per-step GPU time is flat, only
  workloads near a capacity threshold move, and the fix is to keep device
  initialization out of import scope.
  _(Instance: import-time DeepGEMM PDL init creating a CUDA context.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [Import-time DeepGEMM PDL init creates a CUDA context, shrinking the KV pool](deepgemm-pdl-import-time-cuda-context.md) | `kv_cache_size` down 6–15% across models; −8.99% output token throughput on `k25_thinking_fp4_dep4_8k1k-con256` (GB200) | memory-footprint-regression |
| [Native FP8 per-tensor quantization kernel slower than TE's](fp8-pertensor-quant-slower-than-te.md) | TE wins all 30 benchmarked shapes, 0.68×–0.88×; no e2e delta stated | kernel-selection-regression |
