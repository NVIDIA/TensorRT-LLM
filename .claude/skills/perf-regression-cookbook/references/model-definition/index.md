# Regression Cookbook — Model definition

This module is the per-model PyTorch definition layer: the model classes and
their layer implementations, the weight-loading / weight-restore paths, the
eligibility allowlists that decide which optimized kernel a layer may call, and
which submodules get constructed for a given request shape. Regressions here are
almost never a slow kernel — they are a *gate* that stops matching (a
head-group-ratio allowlist, a memory-threshold check) so a correct but slower
reference path runs, or a submodule that gets built and left resident when the
workload never uses it. First thing to check for a single-model regression with
the harness unchanged: whether the model's fast-path predicates still evaluate
true for this config, and what the model constructs at load time.

## Recurring patterns in this module

- **Fast path silently fell back** — a model-side eligibility gate rejects a
  legitimate configuration and the native / on-the-fly reference implementation
  runs correctly and slower. Both shapes appear here: a hard-coded allowlist of
  supported `head_group_ratio` values that a new model's ratio was not in, and a
  free-memory threshold that decides between a cached weight snapshot and an
  on-the-fly subtract. Enumerate what real configs produce for the gated
  quantity, and make the fallback observable — neither case logged anything.
  _(Instances: the Mamba2 head-group-ratio allowlist; the LTX-2 BF16 LoRA
  restore.)_
- **Unaccounted startup residency** — the definition builds submodules a
  text-only benchmark never invokes, and their weights stay resident, shrinking
  the free pool the KV-cache estimate is derived from. The symptom is capacity,
  not kernel time, and no steady-state profile shows it; diff the reported KV
  pool size across builds at identical config.
  _(Instance: the VLM vision tower loaded for a text-only benchmark.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [LTX-2 stage-2 BF16 LoRA restore runs the slow on-the-fly subtract path](ltx2-bf16-lora-restore.md) | stage-2 restore takes the subtract path when `_should_save_bf16_weights()` is false (115.0 GiB free-memory threshold); no percentage stated | fast-path-fallback |
| [Mamba2 selective-state-update falls back on a too-narrow head_group_ratio allowlist](mamba2-flashinfer-head-group-ratio-gate.md) | NT3 NVFP4 perf regression on Blackwell; allowlist `[1, 8, 16]` sends decode to the native kernel | fast-path-fallback |
| [VLM vision tower loaded for a text-only benchmark, shrinking the KV pool](vlm-vision-tower-loaded-for-text-only-bench.md) | ~7.5% on `qwen3.5_9b…input_output_len:500,2000` (L40S); KV pool 22.55 → 21.69 GiB, ~0.86 GiB of vision-tower weights | memory-footprint-regression |
