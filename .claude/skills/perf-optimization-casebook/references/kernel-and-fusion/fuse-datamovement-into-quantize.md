---
id: case-fuse-datamovement-into-quantize
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [memory, launch]
signals: [many-small-kernels, hbm-roundtrip-between-ops]
architectures: [any-sm]
model_scope: [sparse-attention, deepseek-v32, model-agnostic]
phase: [any-phase]
patterns: [pattern-fuse-datamovement-into-quantize]
accuracy_risk: lossless
apply_via_kind: [kernel-change]
knobs: []
specialists: [kernel-cuda-specialist, kernel-triton-writing]
commits: ['9a070ed709', '6601758d3a', '5f737b8dbe', '897c4bffd7']
measured: []
---

# Fuse a concat/gather data-movement op into the quantize that follows it

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `9a070ed709` [perf] Add fused cat+fp8_quantize CUDA kernel for DSA
  indexer (#11899); related: `6601758d3a` Kernel fusions in
  `_gather_k_cache_for_chunk` of the DSA indexer (#12322), `5f737b8dbe` Use fp8
  quant kernel in the DS3.2 indexer (#8701), `897c4bffd7` FP4 indexer
  (`fused_cat_fp4`) (#13340).
- **Applies when:** a **data-movement / layout op immediately precedes a quantize
  (or dequantize)** on the hot path — `cat([pe, nope]) → rotate → view →
  fp8_quantize`, or an index/`unravel` + advanced-indexing gather from a paged
  cache followed by a reinterpret. Signals: a standalone `torch.cat` / `torch.split`
  / `gather` / transpose kernel plus a separate quant kernel in the profile, with a
  contiguous-copy intermediate between them; many small PyTorch ops
  (`arange`/`unsqueeze`/broadcast/`_unravel_indices`/`.view`) building indices per
  step; memory-bound activation prep for a quantized model.
- **Mechanism:** collapse the layout op + the quantize into **one kernel** that
  reads the (possibly non-contiguous) input **views directly via explicit row
  strides** and emits the packed low-precision output + scale factors in a single
  pass. #11899 reads the two BF16 halves straight from `torch.split` views
  (`pe_row_stride`/`nope_row_stride`) — no contiguous concat copy — and writes FP8
  E4M3 + one 1×128 scale/row (`fused_cat_fp8`). #12322 replaces ~8–12 PyTorch ops
  with one Triton `triton_gather_k_cache` that gathers FP8 data + scales from flat
  byte offsets. This removes the layout op's launch, the intermediate HBM
  round-trip, and (for the gather) the index-tensor materialization. Independent
  quantizes can also be **overlapped on a second stream** (#8701 runs Q and K quant
  concurrently).
- **Generalizes to:** the pattern "**fuse the data-movement/layout prologue
  (concat / split / gather / transpose / RoPE-rotate) into the quantize (or
  dequant) that consumes it, reading strided input views directly to skip the
  contiguous-copy round-trip.**" A sibling of [folding a layout pre-pass into a
  GEMM](fold-scale-swizzle-into-kernel.md) and [add+norm+quant fusion](fuse-add-norm-quant.md),
  but here the consumer is the *quantizer*. Carries to: activation `cat`/`split` +
  FP8/FP4 quant in any quantized model, gather-from-paged-cache + dequant,
  transpose+quantize, KV write/read prep. Adapt by: writing the kernel to accept
  strided input pointers, doing the layout in registers, and emitting quantized
  data + scales in one pass; keep the unfused path as the reference.
- **Apply via:** **not a server knob** — a fused custom op (CUDA or Triton).
  Delegate the CUDA kernel + thop binding to **kernel-cuda-specialist**, a Triton
  gather to **kernel-triton-writing**; add a `register_fake` so it traces under
  `torch.compile`.
- **Expected effect:** one kernel instead of two-plus + no contiguous-copy
  intermediate → lower launch/HBM overhead in activation prep. Direction only —
  measured Δ (op count, prep time, throughput) to be recorded from the run.
- **Accuracy risk:** **lossless** — the fused kernel produces the *same quantized
  result* as the unfused `cat/gather → quantize` path; it is a kernel-fusion
  rewrite, so **verify vs the unfused reference** (a kernel bug changes outputs).
  (The FP8/FP4 quantization itself is lossy, but that is pre-existing — this fusion
  does not add precision loss.)
- **Verify:** correctness — fused-op output equals the unfused `cat/gather +
  quantize` reference within tolerance, incl. non-contiguous / edge strides
  (`tests/unittest/_torch/attention/kernels/serial/test_fused_cat_fp8.py`,
  `test_triton_gather_k_cache.py`). Perf — kernel/op count
  and prep-phase time before vs after; confirm `register_fake` lets `torch.compile`
  trace the op.
- **Rollback:** swap the call site back to the unfused `cat/split/gather + quantize`
  path (kept as reference). Trigger: any parity mismatch outside tolerance or a perf
  regression.
- **Prior art:** PRs #11899, #12322, #8701, #13340. Files:
  `cpp/tensorrt_llm/kernels/fusedCatFp8.{cu,h}`, `fusedCatFp4.{cu,h}`,
  `cpp/tensorrt_llm/thop/fusedCatFp8Op.cpp`,
  `tensorrt_llm/_torch/attention_backend/sparse/{dsa.py,kernel.py}`
  (`triton_gather_k_cache`, `_prep_q_or_k`). Owning specialists:
  **kernel-cuda-specialist**, **kernel-triton-writing**. Related:
  [fold-scale-swizzle](fold-scale-swizzle-into-kernel.md) and
  [add+norm+quant](fuse-add-norm-quant.md) (adjacent fusion patterns).
