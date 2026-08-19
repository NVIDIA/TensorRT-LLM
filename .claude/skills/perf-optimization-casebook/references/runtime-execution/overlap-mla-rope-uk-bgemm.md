---
id: case-overlap-mla-rope-uk-bgemm
type: case
family: runtime-execution
maturity: full
bottleneck: [launch, sync]
signals: [small-batch-decode, many-small-kernels]
architectures: [any-sm]
model_scope: [model-agnostic, mla, deepseek-v3r1]
phase: [decode]
patterns: [pattern-overlap-independent-work]
accuracy_risk: lossless
apply_via_kind: [code-change, kernel-change]
knobs: []
specialists: [kernel-cuda-specialist, perf-torch-cuda-graph-specialist, perf-nsight-systems]
commits: ['51545560da00']
eligibility:
  - "multi-stream gated on do_multi_stream() — CUDA graphs / low-latency path enabled"
measured: []
---

# Overlap MLA RoPE with the up/K batched-GEMM on an aux CUDA stream

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `51545560da00` [feat] Add rope and uk-bgemm overlap for mla generation (#8495).
- **Applies when:** sync/launch-bound MLA decode where two independent pre-attention steps run serially: the q_nope up-projection BGEMM (`q_nope × k_b_proj_trans`, "uk-bgemm") and RoPE-on-Q + latent-cache write. Best on the low-latency path with CUDA graphs (multi-stream gated on `do_multi_stream()`); small per-step tensors leave SM/issue slots idle while these run back-to-back.
- **Mechanism:** runs the two ops concurrently via `maybe_execute_in_parallel(fn0, fn1, event0, event1, aux_stream)` — `fn0` = up-proj BGEMM (`torch.ops.trtllm.bmm_out`, or `fp8_block_scaling_bmm_out` for FP8) on the default stream, `fn1` = `mla_rope_generation(...)` on `aux_stream`, synchronized with record/wait events. BGEMM writes into `fused_q[..., :kv_lora_rank]` while RoPE fills the rope portion — disjoint writes, no extra copy. New fused C++ op `torch.ops.trtllm.mla_rope_generation` (`dsv3RopeOp.cpp`) packages RoPE+cache-write into one aux-stream callable.
- **Generalizes to:** "overlap two independent intra-op steps on an aux stream, writing into disjoint slices of a shared output, gated to the CUDA-graph/low-latency regime"; carries to MoE shared-vs-routed overlap, RoPE-vs-KV-write in GQA, projection-vs-norm pairs; adapt by proving data-independence (disjoint output regions), bracketing with record/wait events, and disabling the second stream off the CUDA-graph path.
- **Apply via:** code-level multi-stream via `maybe_execute_in_parallel` (`_torch/modules/multi_stream_utils.py`) + aux_stream + two events; enable CUDA graphs to activate. The fused RoPE op (`dsv3RopeOp.cpp`) is a **kernel-cuda-specialist** change. Delegate to **perf-torch-cuda-graph-specialist**; confirm gap closure with **perf-nsight-systems**.
- **Expected effect:** reduced exposed latency between up-proj and RoPE → lower MLA decode step time under CUDA graphs; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossless (pure scheduling; disjoint writes). The FP8 BGEMM path carries its own FP8 risk independent of this overlap.
- **Verify:** decode step latency + nsys timeline showing BGEMM and RoPE overlapping on two streams; parity-equal attention output vs serial path; correctness with CUDA graphs on/off.
- **Rollback:** pass `aux_stream=None` (or disable multi-stream / CUDA graphs) to run serially. Trigger: capture failures or no measured overlap.
- **Prior art:** PR #8495. Files: `_torch/attention/attention.py`, `cpp/.../thop/dsv3RopeOp.cpp` (`mla_rope_generation`), `attention_backend/trtllm.py`. Related: the [shared/routed-expert overlap case](multi-stream-shared-routed-expert.md) (same `maybe_execute_in_parallel` pattern).
