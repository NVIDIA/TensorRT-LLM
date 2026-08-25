---
id: case-two-model-mtp-eagle
type: case
family: runtime-execution
maturity: full
bottleneck: [launch, sync]
signals: [draft-forward-exposed, gpu-idle-between-steps]
architectures: [any-sm]
model_scope: [model-agnostic, spec-decode, deepseek-v3r1]
phase: [decode]
patterns: [pattern-overlap-independent-work]
accuracy_risk: lossless
apply_via_kind: [config-knob]
knobs: [mtp_eagle_one_model]
specialists: []
commits: ['80dd8fe19733', '6151a4c9d600']
interactions:
  - {case: case-overlap-scheduler, relation: composes-with, note: two-model path re-enables the overlap scheduler on the draft forward}
measured: []
---

# Run MTP-Eagle as two-model speculative decoding with overlap scheduler on the draft forward

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `80dd8fe19733` Enable two-model spec dec for MTP Eagle (#7001); related: `6151a4c9d600` Add simple optimizations for MTP 2-model (#9176) — torch.compile + re-enable overlap for mtp_eagle.
- **Applies when:** running MTP-Eagle spec decode where the single-engine (one-model) path constrains, and the draft (MTP) forward is exposed on the critical path; you want the draft forward to overlap target-side work.
- **Mechanism:** adds a two-model MTP-Eagle mode (`SpeculativeDecodingMode.MTP_EAGLE` vs one-model `MTP_EAGLE_ONE_MODEL`), routing the DeepSeek-V3 draft as a separate `MTPDraftModelForCausalLM` (selected in `modeling_auto.py` when `max_draft_len==0`) with its own `DeepseekV3WeightLoader(is_draft_model=...)`. The two-model path participates in the overlap scheduler (`support_overlap_scheduler()` True). #9176 removes the temporary `is_mtp_eagle()->return False` block (re-enabling overlap after an accuracy fix) and adds `@torch.compile(max-autotune)` to `prepare_for_generation` in `drafting_loops.py`.
- **Generalizes to:** "split a fused single-engine speculator into target+draft models so the draft forward can overlap target work, and torch.compile the per-step draft-prep glue"; carries to Eagle3 two-model, draft-target spec-decode, any two-engine drafting loop; adapt by wiring a separate draft model + weight loader and ensuring the spec mode reports `support_overlap_scheduler()==True`.
- **Apply via:** `speculative_config` knob `mtp_eagle_one_model` (default True). Set `mtp_eagle_one_model=False` (`--use_one_model` controls it: not-one-model ⇒ two-model) to get the two-model path; overlap scheduler then used automatically.
- **Expected effect:** draft forward overlaps target work → lower exposed spec-decode latency / higher throughput; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossless in intent — execution restructuring, not a change to the acceptance rule (output should match one-model). Note: overlap for mtp_eagle was once disabled for an accuracy issue and re-enabled in #9176 — re-verify accuracy.
- **Verify:** accuracy parity two-model vs one-model MTP-Eagle (e2e tests in `test_e2e.py`); confirm draft/target overlap + reduced GPU idle in nsys.
- **Rollback:** `mtp_eagle_one_model=True` (`--use_one_model`) → single-engine MTP-Eagle (no overlap). Trigger: accuracy regression or scheduler instability.
- **Prior art:** PRs #7001, #9176. Files: `_torch/speculative/interface.py` (`support_overlap_scheduler`), `speculative/drafting_loops.py`, `models/modeling_auto.py` (`MTPDraftModelForCausalLM`), `modeling_deepseekv3.py`, `llmapi/llm_args.py`.
