---
id: case-split-mla-reduction-kernel
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [compute]
signals: [attention-hot-path, kernel-far-from-sol, small-batch-decode]
architectures: [any-sm]
model_scope: [mla, deepseek-v3r1, model-agnostic]
phase: [decode]
patterns: [pattern-split-compute-from-reduction]
accuracy_risk: lossless
apply_via_kind: [kernel-change]
knobs: []
specialists: [kernel-cuda-specialist, perf-nsight-compute-analysis]
commits: ['da6cb541a286']
eligibility:
  - "TRTLLM-Gen FMHA generation with multi-CTA-along-KV enabled (isMultiCtasKvEnabled); 1-/2-CTA keepsMmaAbForGeneration MLA kernels"
measured: []
---

# Split MLA generation attention into a separate GMEM reduction kernel

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `da6cb541a286` [feat] Optimize MLA kernels with separate reduction kernels (#7597).
- **Applies when:** compute/occupancy-bound MLA decode using TRTLLM-Gen FMHA generation kernels with multi-CTA-along-KV splitting (`isMultiCtasKvEnabled`), specifically 1-/2-CTA `keepsMmaAbForGeneration` MLA kernels with large reduction tiles. Symptom: the monolithic kernel does cross-CTA softmax/output reduction inline, inflating register/smem pressure and capping occupancy, reduction latency exposed at low/medium batch.
- **Mechanism:** moves the cross-CTA-KV combine (rescale partials by `exp2f(softmaxScaleLog2·(localMax−maxVal))`, accumulate, finalize) into a dedicated `fmhaReductionKernel` (new `MultiCtasKvMode::GmemReductionWithSeparateKernel` replacing in-kernel `GmemReduction`), 512-thread block that splits the reduction across CtasKv to cut latency; the specialized main kernel sheds reduction register/smem footprint → higher occupancy. `runFmhaReduction(...)` dispatched after the main kernel on the same stream.
- **Generalizes to:** "split a monolithic kernel doing compute + a serial cross-partition reduction into two specialized kernels (main + reduction) to raise occupancy and parallelize the reduce"; carries to flash-attn split-KV/split-K combine, GEMM split-K epilogue, any persistent kernel whose tail reduction throttles occupancy; adapt by choosing where to materialize partials (GMEM via `ptrSoftmaxStats`/partial-O) and gating on the regime where the extra launch pays off.
- **Apply via:** internal kernel/dispatch change (auto-selected when multi-CTA-KV enabled for qualifying MLA kernels) — not a server knob. Delegate to **kernel-cuda-specialist**; profile occupancy/SOL with **perf-nsight-compute-analysis**.
- **Expected effect:** higher generation-kernel occupancy, lower MLA decode latency in the multi-CTA-KV regime; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossless (numerically equivalent online-softmax reduction, just relocated). Accuracy tests updated in `test_llm_api_pytorch.py` as guard.
- **Verify:** MLA decode latency/throughput + ncu occupancy of the generation kernel before/after; numeric parity vs in-kernel-reduction path.
- **Rollback:** fall back to `MultiCtasKvMode::GmemReduction`. Trigger: separate-kernel launch adds net latency at small batch, or correctness mismatch.
- **Prior art:** PR #7597. Files: `cpp/.../trtllmGenKernels/fmha/fmhaReduction.cu` (`runFmhaReduction`), `fmhaKernels.h`, `fmhaRunnerParams.h` (`MultiCtasKvMode`). Owning specialist: **kernel-cuda-specialist**.
