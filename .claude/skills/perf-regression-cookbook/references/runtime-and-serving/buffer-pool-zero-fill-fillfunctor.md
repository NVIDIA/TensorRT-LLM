---
id: case-buffer-pool-zero-fill-fillfunctor
type: regression-case
family: execution-and-graph
module: runtime-and-serving
maturity: full
regression_class: [redundant-device-work]
signals: [slower-kernel-in-trace, ttft-increase, throughput-drop]
subsystems: [runtime-python, kv-cache, cuda-graph]
introduced_via: [pre-existing-gap]
phase: [prefill]
patterns: [pattern-unneeded-buffer-initialization]
nvbugs: ["5629833"]
commits: ["6dd2fcd7b3f8"]
success_prs: [9296]
failed_prs: []
---

# Buffer pool used torch.zeros — FillFunctor kernels took 43 % of a context block

> Part of the [Runtime & serving regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5629833` · commit `6dd2fcd7b3f8` · PR #9296 —
  "[https://nvbugs/5629833][fix] Don't fill tensors".
- **Symptom:** the bug itself *is* the profile reading: 4 big FillFunctor
  kernels account for 43 % of one ctx block, reported as a probable regression.
  **The PR body is empty and states no number, model or hardware** — the 43 %
  figure and the four-kernel count come from the bug, and there is no throughput
  percentage anywhere. Do not invent one.
- **Root cause:** `Buffers.get_buffer()` allocated with
  `torch.zeros((required_memory_size,), device='cuda', dtype=torch.uint8)` on
  **both** of its paths. Every pool miss therefore launched a full device-wide
  zero-fill of the requested byte range — visible in a trace as
  `at::native::FillFunctor` — for a scratch buffer whose every consumer overwrites
  it before reading. The consumers are attention metadata, the CUDA-graph runner,
  and DeepGEMM MoE workspaces, i.e. large allocations on the context path, which
  is why four kernels can dominate a prefill block.
- **How introduced:** `pre-existing-gap`. `torch.zeros` is the safe default and
  reads as harmless; nothing regressed it. The bug's own hedged framing (it only
  *looks* like a regression) is the useful artifact — a large unfamiliar kernel
  appearing in a profile is routinely reported as a regression when it has been
  there all along. Confirm a bisect exists before triaging this class as a
  regression.
- **Fix mechanism:** `torch.zeros` → `torch.empty`, plus a docstring recording
  that the buffer is intentionally uninitialized (the docstring was added at
  reviewer request — it is the only durable trace of *why* the call must not be
  "fixed" back). Nothing else changes: same pool, same sizes, same dtype.
- **Detection signal:** in nsys, `FillFunctor` / `elementwise_kernel` filling
  `uint8` with a size that matches a workspace allocation, on the context path,
  not attributable to any model op. Static probe:
  `git grep -n "torch.zeros((required_memory_size" tensorrt_llm/` — and more
  generally `git grep -n "torch.zeros" -- '*/pyexecutor/*' '*/attention_backend/*'`
  and ask, per hit, whether any consumer reads before writing. The runtime
  breadcrumb for the pool path itself is
  `grep -n "Exception happened to create tensor from given memory pool" <run_log>`.
- **Prevention/guard:** **no test and no assert.** The asymmetry is what makes
  this class persist: `zeros` → `empty` is a silent perf win, `empty` → `zeros` is
  a silent perf loss, and **neither direction fails any test** — an
  uninitialized-read bug introduced by the change would surface as flaky
  accuracy, far from this file. So the docstring is doing real work, and a
  reviewer's instinct that `empty` is "unsafe" is exactly how this regresses back.
  If a buffer genuinely needs zeroing, zero it in the consumer that requires it,
  not in the shared allocator.
- **Generalizes to:** `pattern-unneeded-buffer-initialization`; carries to every
  shared scratch/workspace allocator (`get_buffer`, workspace pools, graph-capture
  pads), to `torch.zeros_like` / `.zero_()` on the allocation path, and to the
  broader class of **device work that produces no needed result** — a
  fill/copy/cast whose output is overwritten. Distinct from
  `pattern-host-work-on-hot-path`: nothing here is host-bound, the GPU is genuinely
  busy doing something useless.
