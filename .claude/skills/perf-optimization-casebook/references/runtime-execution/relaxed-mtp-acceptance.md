---
id: case-relaxed-mtp-acceptance
type: case
family: runtime-execution
maturity: full
bottleneck: [compute]
signals: [low-accept-rate]
architectures: [any-sm]
model_scope: [model-agnostic, spec-decode, deepseek-v3r1]
phase: [decode]
patterns: [pattern-bounded-relaxed-acceptance]
accuracy_risk: lossy
apply_via_kind: [config-knob]
knobs: [use_relaxed_acceptance_for_thinking, relaxed_topk, relaxed_delta]
specialists: [perf-sweep-challenger]
commits: ['b1621e8d4e11', '1e5e71aa4277']
eligibility:
  - "MTP spec-decode enabled on a reasoning model with a thinking phase (DeepSeek-R1); incompatible with attention_dp"
interactions:
  - {feature: attention-dp, relation: incompatible-with, note: cannot be enabled together}
measured: []
---

# Raise MTP spec-decode acceptance with relaxed (top-N + delta) acceptance during the thinking phase

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `b1621e8d4e11` feat: add relaxed acceptance for DS (#3865); related: `1e5e71aa4277` Mtp optimizations round1 (#5689) — torch.compile fast-path for relaxed top-k.
- **Applies when:** MTP spec decode enabled, acceptance rate (tokens accepted/step) low and capping speedup, model is a reasoning model with a thinking phase (DeepSeek-R1). NOT compatible with attention_dp (cannot be enabled together).
- **Mechanism:** during `<think>...</think>`, instead of strict top-1 equality, accept a draft token if it lies in a candidate set: take target top-`relaxed_topk` logits, keep tokens with prob > (top-1 prob − `relaxed_delta`), accept the draft if in that set (CUDA kernel `mtpRelaxedAcceptanceKernel` / `torch.ops.trtllm.mtp_relaxed_acceptance_op`; per-request `mtpRelaxedDelta` set at think-start, 0 at think-end → greedy). More drafts accepted/step → higher tokens/step. #5689 wraps the top-k construction in `@torch.compile(max-autotune)` to cut overhead.
- **Generalizes to:** "trade exact-match draft verification for a bounded relaxed acceptance criterion to lift accept rate where small token divergence is tolerable"; carries to Eagle/draft-target spec-decode and other reasoning models with a delimited low-stakes phase; adapt by choosing where relaxed acceptance is safe (here: thinking phase only) and tuning (topk, delta).
- **Apply via:** `speculative_config` (MTPDecodingConfig): `use_relaxed_acceptance_for_thinking=True`, `relaxed_topk` (default 1; README ex 15), `relaxed_delta` (default 0.0; README ex 0.5). CLI `--use_relaxed_acceptance_for_thinking --relaxed_topk 15 --relaxed_delta 0.5`. Gate the accuracy trade with **perf-sweep-challenger**.
- **Expected effect:** higher acceptance → higher tokens/step → "positive speedup" (README, no number) — measured Δ (accept rate, tokens/s) to be recorded from run.
- **Accuracy risk:** lossy — changes which draft tokens are accepted (accepts non-top-1 tokens), so output is NOT token-equivalent to strict acceptance; bets this is acceptable inside the thinking phase only. Restricted to DeepSeek-R1. Needs accuracy record + rollback criterion.
- **Verify:** task accuracy on R1 (PR adds e2e tests in `test_e2e.py`) AND accept-rate / tokens-per-step delta; compare strict vs relaxed output quality.
- **Rollback:** `use_relaxed_acceptance_for_thinking=False` (strict top-1, default). Trigger: accuracy/quality regression, or need to combine with attention_dp.
- **Prior art:** PRs #3865, #5689. Files: `cpp/.../kernels/speculativeDecoding/mtpKernels.cu`, `thop/mtpOp.cpp`, `_torch/speculative/mtp.py`, `llmapi/llm_args.py`, `examples/models/core/deepseek_v3/README.md`.
