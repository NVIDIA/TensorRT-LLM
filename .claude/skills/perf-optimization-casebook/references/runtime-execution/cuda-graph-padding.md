---
id: case-cuda-graph-padding
type: case
family: runtime-execution
maturity: full
bottleneck: [launch]
signals: [slow-path-fallback, expert-load-imbalance]
architectures: [any-sm]
model_scope: [model-agnostic, spec-decode]
phase: [decode]
patterns: [pattern-pad-to-captured-bucket]
accuracy_risk: lossless
apply_via_kind: [config-knob, default-on]
knobs: []
specialists: [perf-torch-cuda-graphs, perf-torch-cuda-graph-specialist]
commits: ['097636020451', '6b583f6f83cb']
measured: []
---

# CUDA-graph padding for variable batch / speculative-decode token counts

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `097636020451` add support for MTP+cuda_graph_padding (#3096); related: `6b583f6f83cb` Enable CUDA graphs when attention DP is used and active requests are uneven (#3010).
- **Applies when:** launch-bound generation where CUDA graphs need a fixed shape but real per-step token/batch count varies, so graphs miss. (a) MTP makes each gen request carry `1 + max_draft_tokens` tokens — padded dummies must match that width; (b) attention-DP across GPUs with uneven active-request counts → per-rank shape differs, graphs can't replay.
- **Mechanism:** make padded/dummy work match the captured shape. (a) MTP: dummy requests created with `max_num_draft_tokens = spec_config.max_draft_tokens` so each padded request has the same `1 + max_draft_tokens` width as real ones. (b) Attention-DP: compute `expected_num_active_requests` = max active count across ranks and inject `_get_num_dummy_request()` dummies via `_merge_dummy_request(n)` so every rank runs the same number of (padded) requests → uniform graph-able shape.
- **Generalizes to:** "pad the dynamic dimension to a captured bucket so the graph always matches"; carries to variable batch size, speculative/medusa/eagle draft widths, attention-DP/EP rank imbalance, any fixed-shape graph fed by variable work; adapt by identifying which dim varies and padding it with shape-correct dummies up to the nearest captured size.
- **Apply via:** enable CUDA-graph padding with the spec config (MTP path in `pyexecutor/model_engine.py`); attention-DP equalization in `py_executor.py` (`expected_num_active_requests`). Delegate to **perf-torch-cuda-graphs**.
- **Expected effect:** higher CUDA-graph hit rate (fewer eager fallbacks) under MTP and uneven DP → lower per-step launch overhead; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossless — padding adds dummy requests/tokens that are discarded. Care: dummies must be cleaned up (`_finish_dummy_request`) with reserved non-conflicting IDs.
- **Verify:** confirm CUDA graph used instead of eager on MTP / uneven-DP steps (nsys, ad-conf-check); padded dummy width == `1 + max_draft_tokens`; outputs/accuracy unchanged.
- **Rollback:** disable CUDA-graph padding (eager on mismatched steps). Trigger: dummy-request lifecycle bugs or DP rank-balancing regressions.
- **Prior art:** PRs #3096, #3010. Files: `_torch/pyexecutor/model_engine.py`, `pyexecutor/py_executor.py`, `pyexecutor/resource_manager.py`. Owning specialist: **perf-torch-cuda-graph-specialist**.
