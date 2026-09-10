---
id: case-overlap-scheduler
type: case
family: runtime-execution
maturity: full
bottleneck: [host-overhead, launch]
signals: [gpu-idle-between-steps, host-prep-on-critical-path, small-batch-decode, high-concurrency]
architectures: [any-sm]
model_scope: [model-agnostic, dense, moe]
phase: [decode]
patterns: [pattern-host-prep-pipelining]
accuracy_risk: lossless
apply_via_kind: [default-on, config-knob]
knobs: [disable_overlap_scheduler, enableTrtOverlap]
specialists: [perf-host-optimization, perf-host-analysis]
commits: ['72057a0a64bf']
eligibility:
  - "no beam search, no speculative decoding, no encoder model (mechanism auto-disables)"
interactions:
  - {feature: beam-search, relation: incompatible-with, note: auto-disables}
  - {feature: spec-decode, relation: incompatible-with, note: auto-disables in the generic path}
  - {feature: encoder-models, relation: incompatible-with, note: auto-disables}
  - {case: case-two-model-mtp-eagle, relation: composes-with, note: re-enables overlap under MTP-Eagle two-model spec-decode}
measured: []
---

# Overlap host scheduling with GPU compute (overlap scheduler)

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `72057a0a64bf` [TRTLLM-3429] feat: Overlap scheduling in C++ runtime (#3625); the same mechanism is the default in the PyTorch executor (`_executor_loop_overlap`, knob `disable_overlap_scheduler` in `tensorrt_llm/llmapi/llm_args.py`).
- **Applies when:** launch/host-prep-exposed bottleneck — nsys shows GPU idle between forward steps and host prep (scheduling, request bookkeeping, decoder/sampler prep, input tensor build) is on the critical path; steady-state generation-heavy serving with many in-flight requests; small-batch / low-arithmetic-intensity decode where per-step host work rivals GPU step time.
- **Counter-signals:** GPU-bound steps (host prep already hidden — nothing to win, and the extra in-flight state costs memory); beam search / speculative decoding / encoder models (mechanism auto-disables; the two-model MTP-Eagle path restores overlap — see [that case](two-model-mtp-eagle.md)); debugging sessions that need strictly serialized step semantics.
- **Mechanism:** runs CPU-side preparation for step N+1 (micro-batch scheduling, decoder input/output marshaling) concurrently with engine execution of step N, hiding host latency behind GPU compute instead of serializing before it. C++ adds `enableTrtOverlap` to `ExecutorConfig` and an `invokeGatherBatch` runtime kernel so the overlapped same-batch token gather stays on GPU; keeps the two steps' state disjoint.
- **Generalizes to:** producer/consumer pipelining of host-prep against device compute; carries to the PyTorch `_executor_loop_overlap`, to spec-decode draft/target overlap, and to any per-iteration loop where input marshaling can be hoisted one step ahead; adapt by ensuring step N+1 prep does not read state mutated by step N's results (disjoint-state plumbing is the hard part).
- **Apply via:** PyTorch backend — leave `disable_overlap_scheduler=False` (default on). C++ runtime — `ExecutorConfig.setEnableTrtOverlap(true)`. Auto-disabled for beam search, speculative decoding, encoder. Delegate host-overhead validation to **perf-host-optimization**.
- **Expected effect:** higher throughput and lower per-step latency by shrinking inter-step GPU idle; direction only (no number quoted) — measured Δ (tok/s, GPU-idle ratio) to be recorded from run.
- **Accuracy risk:** lossless — pure scheduling reorder; identical math.
- **Verify:** A/B the knob (`disable_overlap_scheduler` on/off) at fixed concurrency: nsys inter-step GPU-idle / host-prep-exposed ratio drops (detection recipe: **perf-host-analysis**); throughput up; output parity vs overlap-off.
- **Rollback:** `disable_overlap_scheduler=True` (PyTorch) or `enableTrtOverlap=false` (C++). Trigger: throughput regression or correctness mismatch.
- **Prior art:** PR #3625. Files: `cpp/include/.../executor/executor.h`, `batch_manager/trtGptModelInflightBatching.cpp`, `runtime/runtimeKernels.cu` (`invokeGatherBatch`), `_torch/pyexecutor/decoder.py`. Detection: **perf-host-analysis**.
