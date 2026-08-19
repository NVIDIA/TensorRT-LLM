# Casebook — Kernel & Fusion

Optimizations at the kernel layer: fusing ops, swapping in fused/specialized
kernels, and writing custom kernels. **Implementation is always delegated to a
`kernel-*` specialist/skill** — this casebook only records when each precedent
applies and how to verify it. A numerically-equivalent kernel rewrite is
`lossless`, but equivalence is "within tolerance" and **must be verified
against a reference**, since a kernel bug changes outputs. Add entries using
the schema in [case-template.md](../case-template.md).

> Routing note: classify the bottleneck first (`perf-analysis` / SOL% from
> `perf-nsight-compute-analysis`). Kernel work pays off when a kernel is the
> hot path and is far from SOL, or when many small ops can be fused to cut
> launch/memory traffic. See `perf-optimization` for the specialist routing
> table (Triton vs TileIR two-step vs CuTe DSL).

## Recurring patterns in this family

Match on these transferable patterns, not on a case title — most situations are
a *variation* of a case, not its exact instance. Each pattern is the reusable
idea; each links to its worked case.

- **Collapse a small, launch-bound op-chain into one fused kernel.** Pre/post
  steps around the big GEMMs operate on small per-token tensors — routing/gating,
  sampling, RoPE, norm epilogues, KV scatter — yet run as many tiny kernels with
  HBM round-trips between them. On the decode hot path their launch + memory
  overhead is disproportionate. Fuse the chain into one block/warp-per-token
  kernel, keep intermediates in registers/shared, and reuse an existing
  purpose-built primitive instead of a generic algorithm. A fused kernel needs a
  bounded-config guard + fallback to the generic path. _(Instance: [the noaux_tc routing case](fuse-moe-routing-kernel.md).)_
- **When a Triton kernel is host-bound, move it to a precompiled C++/CUDA custom
  op.** A small Triton kernel on the decode hot path can be cheap on the GPU yet
  dominated by Triton's *host-side* cost — per-launch Python dispatch,
  autotune-cache lookup, argument marshalling, and JIT/recompilation. Re-implement
  it as a C++/CUDA `torch.ops.trtllm.*` custom op (precompiled, thin launch path)
  with a `register_fake` meta impl so it still traces under `torch.compile`. This
  is a host-overhead fix, not a SOL fix — classify host-bound first. _(Instance:
  [the DSA indexer gather / index-convert ops](triton-to-cpp-op.md).)_
- **Mark a Triton param `tl.constexpr` only when it must be compile-time.** Triton
  recompiles a fresh kernel per distinct value of every `tl.constexpr` argument,
  so a runtime scalar that varies across calls (e.g. a per-layer `layer_id`, a
  stride, a block count) marked `constexpr` forces a recompile per value. Keep
  `tl.constexpr` only where the value is genuinely needed at compile time (tile
  sizes used in `tl.arange`, static loop unrolling); pass the rest as runtime
  args. _(Instance: [the DSA `convert_req_index_to_global` kernel](relax-tl-constexpr.md).)_
- **Fuse the chain that feeds the next GEMM.** Beyond a generic op-chain: the TP
  collective *epilogue* (AllReduce + residual-add + RMSNorm + quant), a local
  add + RMSNorm + quant, QK-Norm + RoPE, or folding a standalone layout pre-pass
  (scale-factor swizzle) into the consumer kernel — and, when every stage is small,
  mega-fusing a whole multi-stage pipeline (EP-dispatch + GEMM1 + SwiGLU + GEMM2 +
  EP-combine) into one launch. Removes launches + HBM round-trips; **the quant step
  makes it lossy** → parity-check. Most of these land as `torch.compile`
  PatternMatcher rewrites or a single custom op. _(Instances: the [AR-epilogue](fuse-ar-epilogue.md), [add+norm+quant](fuse-add-norm-quant.md), [QK-Norm+RoPE](fuse-qk-norm-rope.md), [scale-swizzle](fold-scale-swizzle-into-kernel.md), and [MegaMoE](mega-fuse-moe-deepgemm.md) cases.)_
- **Quantize the memory-bound side, keep the precision-sensitive boundary high.**
  Store the KV cache (and the matching operand) in FP8/FP4 to cut HBM bytes, but
  keep accumulation/output in BF16 where the next op (e.g. `o_proj`) is sensitive.
  Lossy → accuracy record + rollback. Carries to GQA/MHA FP8 KV, FP8 context FMHA,
  INT8/FP4 KV. _(Instance: [the FP8 MLA KV case](fp8-mla-kv-cache.md).)_
- **Split a monolithic kernel that does compute + a serial cross-partition
  reduction into two specialized kernels (main + reduction)** to shed the
  reduction's register/smem footprint, raise occupancy, and parallelize the reduce.
  Lossless but a kernel rewrite — verify parity. Carries to flash-attn split-KV/
  split-K combine, GEMM split-K epilogue, any persistent kernel whose tail
  reduction throttles occupancy. _(Instance: [the separate-reduction case](split-mla-reduction-kernel.md).)_
- **Pick the hardware-matched low-precision grouped-GEMM backend over the generic
  one** when the GEMM is the bottleneck and the dtype is natively supported
  (trtllm-gen / DeepGEMM / MegaMoE on Blackwell, W4A8 on Hopper, vs generic
  CUTLASS). Match the cubin/scale-layout to the SM and keep a non-quant fallback.
  Lossy → accuracy record. _(Instances: the [trtllm-gen FP4](trtllm-gen-fp4-moe-backend.md), [MegaMoE](mega-fuse-moe-deepgemm.md), and [HW-matched-GEMM](hw-matched-lowprec-moe-gemm.md) cases.)_
- **Attend to only a selected subset of KV in the attention kernel.** For sparse /
  block-sparse / windowed attention (DeepSeek V3.2 DSA picks top-k KV per query),
  pass the selected page/block indices and bound the kernel's KV loop **and** its
  cross-CTA reduction by the selection count instead of the full sequence. Two
  corollaries travel with it: **write K/V straight into the paged cache**
  (absorption/latent layout) to skip a materialization pass, and **early-exit CTAs
  beyond the ragged per-sequence length**. _(Instance: [the sparse-MLA top-k attention case](sparse-mla-topk-attention.md).)_
- **The hot-path top-k / selection is a specialized kernel, not a generic sort.**
  Fuse the selection with its index/mask post-processing into one radix-select,
  keep several exact variants (single-CTA / multi-CTA / multi-pass radix) and
  dispatch by `(k, N, dtype, SM-wave-occupancy)`; short-circuit with a coarse
  low-precision first pass and a **warm-start from the previous decode step's
  answer** (temporal correlation). Carries to MoE routing top-k, sampling,
  beam-search. _(Instance: [the top-k selection kernel case](specialize-topk-selection-kernel.md); cf. the [MoE routing kernel](fuse-moe-routing-kernel.md).)_
- **Fuse a data-movement prologue into the quantize that follows.** When a
  `cat`/`split`/`gather`/`rotate` immediately precedes an FP8/FP4 quantize, collapse
  both into one kernel and **read the strided input views directly** to skip the
  contiguous-copy round-trip. The quantize-consumer sibling of "fold a layout
  pre-pass into the consumer." _(Instance: [the fuse-data-movement-into-quantize case](fuse-datamovement-into-quantize.md).)_
- **A ranking-only computation tolerates reduced precision — and verify the fast HW
  path is actually taken.** A GEMM whose output only feeds an `argmax`/top-k/threshold
  (a selector/scoring GEMM) can run on TF32/BF16 tensor cores; keep the
  higher-precision contract only where a scale/softmax needs it, and **confirm the
  accelerated kernel is dispatched** — small-M GEMMs silently fall back to CUDA-core
  SGEMM even when you asked for it. _(Instance: [the ranking-only TF32 case](ranking-only-precision-tf32.md).)_
- **Fusion is not monotonic — re-evaluate the fuse boundary per dtype/checkpoint.**
  Concatenating co-resident projections into one wide GEMM wins **only when the fused
  weights share a dtype**; a mixed-precision checkpoint can force *unfusing*, and
  unfusing can unlock a conditional skip. The caveat that keeps every "fuse!" pattern
  above from being applied blindly. _(Instance: [the fusion-boundary case](reevaluate-fusion-boundary-per-dtype.md).)_

## Cases

Match on **Applies when** / **Generalizes to**, then open that case file.
Implementation is always delegated to a `kernel-*` specialist/skill.

> Risk key: **lossless** = numerically equivalent (still verify vs a reference —
> a kernel bug changes outputs) · **lossy** = needs an accuracy record + rollback
> before promotion · **mixed** = base fusion lossless but the quant variant lossy.

### Fusion (collapse op-chains)

| Case | Applies when (signal) | Pattern (generalizes to) | Risk |
|------|----------------------|--------------------------|------|
| [Fuse the TP AR epilogue](fuse-ar-epilogue.md) | comm-bound TP>1; `allreduce → add → RMSNorm → (quant)` chain per layer | fuse the collective's epilogue into the collective kernel | mixed |
| [Fuse add + norm + quant](fuse-add-norm-quant.md) | memory-bound epilogue, NO collective; separate `add → RMSNorm → quant` kernels | fuse a pointwise+reduction+quant chain that feeds the next GEMM | lossy |
| [Fuse QK-Norm + RoPE](fuse-qk-norm-rope.md) | launch/mem-bound attention pre-proc; per-head Q/K RMSNorm before RoPE (Qwen3/Gemma3), bf16 | fold the norm preceding a positional/elementwise transform into that kernel | lossless |
| [Fold scale-swizzle into the kernel](fold-scale-swizzle-into-kernel.md) | NVFP4 WideEPMoE; a standalone `swizzle_sf` relayout runs before the grouped GEMM | fold a standalone layout pre-pass into the kernel that consumes the data | lossless |
| [Fuse MoE grouped-top-k routing](fuse-moe-routing-kernel.md) | MoE router = many tiny kernels on decode hot path (noaux_tc: DS-V3/Kimi-K2), or fell back to PyTorch router | collapse a small launch-bound op-chain into one per-token kernel, reuse a warp primitive | lossless |
| [Fuse data-movement into the quantize](fuse-datamovement-into-quantize.md) | a `cat`/`split`/`gather`/`rotate` immediately precedes an FP8/FP4 quantize; standalone layout kernel + copy in the profile | fuse the layout prologue into the quantize; read strided views to skip the copy | lossless |
| [Re-evaluate the fusion boundary per dtype](reevaluate-fusion-boundary-per-dtype.md) | deciding whether to fuse GEMMs sharing an input; a mixed-precision checkpoint splits participants, or a fused op blocks a skip | fusion is not monotonic — the optimal boundary is dtype/checkpoint-dependent | lossless |

### Attention / MLA kernels

| Case | Applies when (signal) | Pattern (generalizes to) | Risk |
|------|----------------------|--------------------------|------|
| [FP8 MLA KV cache](fp8-mla-kv-cache.md) | memory-bound MLA decode; KV reads dominate; DS-MLA, SM90/100, long ctx / high conc | quantize the memory-bound side (KV+operand) to FP8, keep output BF16 | lossy |
| [Split MLA into a reduction kernel](split-mla-reduction-kernel.md) | compute/occupancy-bound MLA decode on TRTLLM-Gen FMHA multi-CTA-KV; big reduction tiles cap occupancy | split a compute+serial-reduction kernel into main + separate reduction | lossless |
| [Sparse-MLA top-k attention](sparse-mla-topk-attention.md) | attention over a top-k–selected KV subset (DSA / block-sparse / windowed), SM100+; full-seq FMHA wasteful | attend to only the selected KV — bound the kernel's loop/reduction by the selection count | mixed |
| [Specialize the top-k selection kernel](specialize-topk-selection-kernel.md) | hot-path top-k (indexer / MoE router / sampling) = a `torch.topk` sort + index/mask ops, every step | specialized radix-select; dispatch by (k,N,dtype,wave); coarse-pass + prev-step warm-start | lossless |

### MoE / grouped-GEMM kernels

| Case | Applies when (signal) | Pattern (generalizes to) | Risk |
|------|----------------------|--------------------------|------|
| [trtllm-gen FP4 MoE backend](trtllm-gen-fp4-moe-backend.md) | compute-bound MoE grouped-GEMM, sm_100a, NVFP4/FP8-block-scale, on default CUTLASS | pick a HW-specialized low-precision grouped-GEMM backend over the generic one | lossy |
| [Mega-fuse MoE into DeepGEMM](mega-fuse-moe-deepgemm.md) | launch-bound MoE (many per-stage kernels + EP a2a), sm_100, W4A8_MXFP4_MXFP8, EP-only, shape%128 | mega-fuse a multi-stage pipeline whose stages are each launch/mem-bound | lossy |

### Quantization (low-precision GEMM)

| Case | Applies when (signal) | Pattern (generalizes to) | Risk |
|------|----------------------|--------------------------|------|
| [HW-matched low-precision MoE GEMM](hw-matched-lowprec-moe-gemm.md) | compute/weight-mem-bound MoE (DeepSeek); pick precision by GPU (FP8-block Blackwell / W4A8 Hopper) | select the low-precision GEMM format + backend by GPU tensor-core capability | lossy |
| [Ranking-only precision (TF32)](ranking-only-precision-tf32.md) | a scoring/selector GEMM feeds only an argmax/top-k; runs as CUDA-core SGEMM at small M | a ranking-only output tolerates TF32 tensor cores — and verify the path is actually taken | lossy |

### Triton / kernel host-overhead hygiene

| Case | Applies when (signal) | Pattern (generalizes to) | Risk |
|------|----------------------|--------------------------|------|
| [Triton kernel → C++ custom op](triton-to-cpp-op.md) | a Triton kernel is host-bound (tiny GPU work, heavy launch/JIT); DSA indexer gather/index-convert | move a host-bound Triton kernel to a precompiled C++/CUDA op (+`register_fake`) | lossless |
| [Relax unnecessary tl.constexpr](relax-tl-constexpr.md) | a Triton kernel marks runtime-varying scalars (`layer_id`, stride) `tl.constexpr` → recompiles per value | mark `tl.constexpr` only when compile-time-required; pass the rest as runtime args | lossless |

## Suggested slots (optional — replace or delete)

Still open in this family: elementwise/activation fusion (delegate to
**kernel-triton-writing** / **kernel-cute-writing**), TileIR optimization of an
existing Triton kernel on Blackwell (the two-step pipeline via
**kernel-tileir-optimization**), CuTe DSL generation (**kernel-cute-writing**),
and GEMM tactic / autotune selection.
