---
id: case-attention-dp-padding
type: case
family: runtime-execution
maturity: full
bottleneck: [host-overhead, compute, memory]
signals: [expert-load-imbalance, host-prep-on-critical-path]
architectures: [sm90, sm100]
model_scope: [model-agnostic, moe, deepseek-v3r1, qwen3-moe, llama-family]
phase: [any-phase]
patterns: [pattern-process-only-real-work]
accuracy_risk: lossless
apply_via_kind: [default-on]
knobs: []
specialists: [perf-host-optimization, perf-host-analysis]
commits: ['07e8813984cd', 'b618e1f55b88', '15823614000b']
measured: []
---

# Eliminate padding / wasted compute in data-parallel attention (attention-DP)

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `07e8813984cd` feat: Remove padding in attention DP (#6064); related: `b618e1f55b88` Eliminate the need for attention DP padding when possible (#3439, variable-length allgather/reducescatter), `15823614000b` only pad one dummy request for attention dp (#4664).
- **Applies when:** attention-DP on and per-rank token counts are imbalanced/ragged (mixed ISL, uneven request assignment) so the framework padded every rank to the max — wasting attention/MoE compute and host bookkeeping on dummy tokens; signals: low MFU with attention-DP, throughput scaling poorly with DP degree, dummy work in profiles.
- **Mechanism:** (a) #3439 replaces fixed all-padded collectives with variable-length allgather/reducescatter so ranks exchange only real tokens. (b) #4664 collapses per-step "pad to equal length" to a single dummy request (`_pad_attention_dp_dummy_request`) and drops the cleanup across executor loops, cutting per-step host overhead. (c) #6064 removes residual padding/swizzle bookkeeping in the MoE path so the un-padded path is plumbed end to end.
- **Generalizes to:** "process only real tokens, not padded-to-max" for any DP/ragged-batch collective; carries to MoE all-to-all dispatch, other DP'd modules, any allgather/reducescatter on variable per-rank counts; adapt by making the collective length-aware and removing downstream padded-shape assumptions.
- **Apply via:** keep attention-DP on with the variable-length collective path (default after these PRs); no extra knob beyond a recent build. Delegate residual host cleanup to **perf-host-optimization**.
- **Expected effect:** higher throughput and MFU under ragged DP load by removing dummy-token compute + per-step host padding; direction only — measured Δ (tok/s, MFU, host prep) to be recorded from run.
- **Accuracy risk:** lossless — removes computation on padding/dummy tokens that never contributed to outputs.
- **Verify:** throughput/MFU up with imbalanced ISL; profile shows reduced dummy/padding work + lower host prep; output parity vs padded path.
- **Rollback:** revert to padded collective / full per-step padding. Trigger: correctness mismatch on ragged batches or collective hang.
- **Prior art:** PRs #6064, #3439, #4664. Files: `cpp/.../thop/{allgatherOp,reducescatterOp}.cpp`, `_torch/distributed/ops.py`, `_torch/pyexecutor/py_executor.py`, `_torch/moe/fused_moe/{fused_moe_cutlass,fused_moe_wide_ep}.py`. Detection: **perf-host-analysis**.
