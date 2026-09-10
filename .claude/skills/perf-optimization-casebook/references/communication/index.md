# Casebook — Communication

Optimizations for **multi-GPU collectives** — expert-parallel (EP) all-to-all for
MoE dispatch/combine, and tensor-parallel (TP) AllReduce — plus the buffers and
strategy-selection that feed them. These only matter at **TP>1 / EP>1** and only
when a collective is a real share of step time; classify communication-bound
first (`perf-analysis` / `perf-nsight-systems` — collective time as a fraction of
the step, NCCL/all-to-all kernels on the critical path). Add entries using the
schema in [case-template.md](../case-template.md); match on **Applies when** signals,
adapt to the current world size / interconnect, and measure before claiming a win.

> Routing note: a collective's *epilogue* fusion (AllReduce + Residual + RMSNorm
> (+Quant)) is a `torch.compile` fusion pattern and lives in
> the kernel-and-fusion casebook ([Fuse the TP collective epilogue](../kernel-and-fusion/fuse-ar-epilogue.md)); the [UserBuffers case](userbuffers-symmetric-memory.md) is what makes the *in-place* fused UB
> variant of that possible. Many of these are bf16-only or quant-checkpoint-gated
> — read the dtype gate in each case.

## Recurring patterns in this family

Match on these transferable patterns, not on a case title — most situations are a
*variation* of a case, not its exact instance.

- **Swap a generic collective for a workload-specialized comm library.** Generic
  `allgather`/`reducescatter` moves padded, full-width data; a purpose-built EP
  all-to-all (DeepEP) routes only each rank's selected tokens to their expert
  owners and overlaps NVLink/RDMA transfer with layout compute. Carries to EP
  combine, attention-DP all-to-all, PP send/recv. Pick the CUDA-graph-safe variant
  for decode. _(Instance: [the DeepEP case](deepep.md).)_
- **Send activations post-quant, not pre-quant.** When the model is already
  quantized, quantize tokens *before* the collective and ship the packed
  low-precision payload (~½ for FP8, ~¼ for FP4) instead of bf16 — pack scale
  factors with the payload and re-swizzle on receive. **Lossy** (precision crosses
  the wire) → accuracy record + rollback. Carries to EP dispatch+combine, TP
  allgather/reducescatter of quantized activations, KV transfer. _(Instance: [the low-precision dispatch case](low-precision-dispatch.md).)_
- **Make the collective *kernel* faster instead of changing the algorithm.**
  Buffer-flag / Lamport sync, grid right-sizing, vectorized multicast loads tune a
  fixed algorithm (e.g. MNNVL two-shot) rather than swapping it. Lossless (same
  reduction math). Carries to one-shot AllReduce, allgather/reduce-scatter, custom
  NVLS/multimem collectives — mind the alignment/divisibility assumptions.
  _(Instance: [the MNNVL two-shot case](mnnvl-twoshot-allreduce.md).)_
- **Pre-register symmetric comm buffers to kill per-step allocation and unlock
  in-place fused collectives.** A warm-up-time symmetric-memory pool (UserBuffers /
  NCCL-symmetric) removes dynamic allocation during inference and lets the producer
  write directly into the comm buffer so AR + RMSNorm (+quant) fuse in place.
  Carries to NCCL-symmetric AllReduce, AR+norm/quant fusion. _(Instance: [the UserBuffers case](userbuffers-symmetric-memory.md).)_
- **Autotune / heuristically pick the collective per shape & concurrency.** No
  single AllReduce algorithm wins across message sizes — choose by
  `(SM, TP, fusion, hidden, tokens)` via an offline LUT or a runtime `TunableRunner`.
  Lossless (bit-equivalent implementations). Carries to allgather/reduce-scatter,
  GEMM tactic selection, attention-backend choice. _(Instance: [the shape-aware autotune case](shape-aware-allreduce-autotune.md).)_

## Cases

Match on **Applies when** / **Generalizes to**, then open that case file. These
only matter at **TP>1 / EP>1** when a collective is a real share of step time.

> Risk key: **lossless** = bit-/math-equivalent (only transport or kernel changes)
> · **lossy** = precision crosses the wire; needs an accuracy record + rollback ·
> **mixed** = the AllReduce+norm fusion is lossless but its quant variant is lossy.

### EP all-to-all (MoE dispatch / combine)

| Case | Applies when (signal) | Pattern (generalizes to) | Risk |
|------|----------------------|--------------------------|------|
| [DeepEP all-to-all backend](deepep.md) | comm-bound MoE EP; dispatch/combine all-to-all a large share of step time; NVLink/RDMA | swap a generic collective for a workload-specialized comm library | lossless |
| [Low-precision dispatch/combine](low-precision-dispatch.md) | comm-bound MoE already on DeepEP + quantized model (FP8/NVFP4/W4A8); bf16 dispatch bytes dominate | send activations post-quant, not pre-quant | lossy |

### TP AllReduce (kernel, buffers, strategy)

| Case | Applies when (signal) | Pattern (generalizes to) | Risk |
|------|----------------------|--------------------------|------|
| [MNNVL two-shot AllReduce kernel](mnnvl-twoshot-allreduce.md) | comm-bound TP AllReduce on MNNVL/NVLink; mid/large messages; the AR kernel (not the network) is the bottleneck | make the collective kernel faster instead of changing the algorithm | lossless |
| [UserBuffers / symmetric memory](userbuffers-symmetric-memory.md) | comm-bound TP AllReduce (PyTorch flow); AR→add→RMSNorm(+quant); per-step buffer alloc / unfused chain | pre-register symmetric comm buffers to enable in-place fused collectives | mixed |
| [Shape-aware AllReduce autotune](shape-aware-allreduce-autotune.md) | comm-bound TP AllReduce; no single algo wins across shapes (one-shot regresses vs NCCL somewhere); strategy pinned | autotune/heuristically pick the collective per shape & concurrency | lossless |
