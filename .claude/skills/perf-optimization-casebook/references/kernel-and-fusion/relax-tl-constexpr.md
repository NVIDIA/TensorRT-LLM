---
id: case-relax-tl-constexpr
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [host-overhead]
signals: [recompilation-churn, host-prep-on-critical-path]
architectures: [any-sm]
model_scope: [sparse-attention, deepseek-v32, model-agnostic]
phase: [any-phase]
patterns: [pattern-constexpr-only-when-required]
accuracy_risk: lossless
apply_via_kind: [kernel-change]
knobs: []
specialists: [kernel-triton-writing]
commits: []
interactions:
  - {case: case-triton-to-cpp-op, relation: alternative-to, note: the heavier fix when host overhead persists after relaxing constexpr}
measured: []
---

# Relax unnecessary tl.constexpr so a Triton kernel isn't recompiled per layer

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Applies when:** a Triton kernel marks parameters `tl.constexpr` that don't
  need to be compile-time constants, and at least one such param takes many
  distinct runtime values across calls — e.g. a per-layer `layer_id`, or a stride
  / block-count that varies — so Triton recompiles a fresh kernel per value.
  Symptoms: repeated Triton autotune/compile, a growing compile cache, host time
  across the first N layers/shapes. Instance: the DSA
  `_convert_req_index_to_global_index_kernel_with_stride_factor` (had `layer_id`,
  `stride_factor`, `max_num_blocks_per_req`, `BLOCK_SIZE` all `tl.constexpr`).
- **Mechanism:** Triton specializes (recompiles) per distinct value of every
  `tl.constexpr` argument. Params used only as plain runtime scalars — passed to
  arithmetic, indexing, or masks — don't need to be `constexpr`; marking them so
  forces a recompile per value (`layer_id` → one compile per layer). Dropping
  `tl.constexpr` from those, keeping it only where the value must be compile-time
  (tile sizes used in `tl.arange`, static unrolling — here `BLOCK_N`), yields a
  single kernel reused across all values.
- **Generalizes to:** the pattern "**mark a Triton param `tl.constexpr` only when
  it must be a compile-time constant (tiling/shape/`tl.arange`/unroll); otherwise
  pass it as a runtime arg to avoid per-value recompilation.**" Carries to: any
  Triton kernel with a per-layer/per-step/per-shape id or scalar marked
  `constexpr`; block-count or stride params not used for tiling; "`constexpr` by
  habit" annotations. Adapt by: for each `constexpr` param, check whether it is
  used where a compile-time constant is *required* (`tl.arange` bounds, static
  loop ranges); if not, drop the annotation. Trade-off: runtime args forgo a few
  specialization-time optimizations, but on a host-bound path avoiding recompiles
  wins.
- **Apply via:** **not a server config knob** — a Triton-kernel edit. Delegate to
  **kernel-triton-writing**. Touch point (prior art, TRT-LLM repo):
  `tensorrt_llm/_torch/attention_backend/sparse/kernel.py` — the kernel signature
  (keep `BLOCK_N: tl.constexpr`, relax the rest).
- **Expected effect:** fewer Triton recompilations across layers/shapes → lower
  first-iteration and steady host time, smaller GPU-idle; no change to GPU
  compute. Record the local Δ (distinct-compile count, per-iteration host time) —
  PR #12581 ships it without a public number; do **not** cite one not measured.
- **Accuracy risk:** lossless — identical kernel math; only the compile-time-vs-
  runtime treatment of scalar args changes.
- **Verify:** correctness — kernel output unchanged vs reference (same DSA tests).
  Perf — count distinct Triton compiles across a multi-layer run before vs after
  (e.g. Triton compile logs / cache size); per-iteration host time. Confirm the
  kept-`constexpr` param (`BLOCK_N`) still drives the `tl.arange` tiling.
- **Rollback:** re-add `tl.constexpr`; trigger: a kernel-compute regression (if a
  relaxed param turned out to enable a needed specialization) or any correctness
  mismatch.
- **Prior art:** PR #12581; `kernel.py` path above.
  Owning specialist: **kernel-triton-writing**. Related: the [Triton→C++-op case](triton-to-cpp-op.md) (the heavier fix when host overhead persists after relaxing `constexpr`).
