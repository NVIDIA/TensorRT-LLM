---
id: case-ltx2-bf16-lora-restore
type: regression-case
family: execution-and-graph
module: model-definition
maturity: full
regression_class: [fast-path-fallback]
signals: [throughput-drop, midrun-stall]
subsystems: [model-definition]
introduced_via: [prior-fix-side-effect]
phase: [any-phase]
patterns: [pattern-fast-path-silent-fallback]
nvbugs: ["6179761"]
commits: ["5db9414cbeef"]
success_prs: [14639]
failed_prs: []
---

# LTX-2 stage-2 BF16 LoRA restore runs the slow on-the-fly subtract path

> Part of the [Model definition regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6179761` · commit `5db9414cbeef` · PR #14639 —
  "[https://nvbugs/6179761][fix] Save LTX-2 BF16 weights to speed up perf".
- **Symptom:** Slow LTX-2 two-stage visual-gen pipeline perf around the
  stage-2 distilled-LoRA merge/restore: BF16 weights touched by LoRA were
  restored by re-subtracting the deltas after stage 2 instead of a snapshot
  copy ("the slower on-the-fly subtract path" per the PR). No percentage is
  stated in the PR.
- **Root cause:** In `_apply_lora_deltas`
  (`tensorrt_llm/_torch/visual_gen/models/ltx2/pipeline_ltx2_two_stages.py`)
  only quantized (FP8/FP4) parameters were snapshotted into
  `saved_lora_state`; dense BF16/FP16/FP32 weights were never saved, so
  restore after stage 2 always went through
  `_subtract_dense_lora_deltas` — a per-parameter delta cast + subtract —
  even when GPU memory could hold a BF16 snapshot.
- **How introduced:** prior-fix side effect. The fix PR names no regressing
  commit; `git log -S "_subtract_dense_lora_deltas"` on
  `pipeline_ltx2_two_stages.py` points at PR #13244 ("[None][fix] Use bf16 for
  LTX-2 FP4 stage 2", merged 2026-04-30 to `main`), whose diff drops the
  `saved_state[param_name] = param.data.clone()` snapshot for dense weights and
  adds `_subtract_dense_lora_deltas`, so stage 2 restores dense BF16 weights by
  subtracting deltas on the request path. That trade was not careless: the
  snapshot path clones almost the whole LoRA-touched BF16 transformer, and the
  fix's own diff comment puts baseline BF16 peak at ~75 GiB vs ~108 GiB with
  snapshots — so it traded latency for memory deliberately. That makes this a
  *tradeoff reintroduced under a gate* rather than a straight regression fix —
  the same shape as `case-warmup-token-cap-revert`, where a protective clamp and
  the perf it cost are the two ends of one decision. Do not "simplify" the
  memory gate away. The faster snapshot-copy restore had survived for FP8/FP4
  quantized state throughout.
- **Fix mechanism:** Adds a memory-aware gate `_should_save_bf16_weights()`:
  when `torch.cuda.mem_get_info()` reports free memory above
  `_BF16_WEIGHTS_SNAPSHOT_FREE_MEMORY_THRESHOLD_GIB = 115.0` GiB (diff
  comment: baseline BF16 peak ~75 GiB, with snapshots ~108 GiB total), BF16
  params touched by LoRA are cloned into `saved_lora_state` and restored by
  `copy_` after stage 2; below the threshold (or with no CUDA memory query)
  it keeps the subtract fallback. FP8/FP4 handling is unchanged.
- **Detection signal:** debug log line `BF16 weight snapshots
  enabled/disabled: free GPU memory ... GiB ... threshold` on stage-2 LoRA
  merge; inspect the gate and threshold with `grep -n
  "_should_save_bf16_weights\|_BF16_WEIGHTS_SNAPSHOT_FREE_MEMORY_THRESHOLD"
  tensorrt_llm/_torch/visual_gen/models/ltx2/pipeline_ltx2_two_stages.py`.
- **Prevention/guard:** PR #14639 added unit tests in
  `tests/unittest/_torch/visual_gen/test_ltx2_pipeline.py`
  (`test_bf16_weight_snapshot_gate_uses_cuda_free_memory`,
  `test_bf16_weight_snapshot_saved_when_requested`,
  `test_fp32_state_not_saved_and_subtract_restores`); these guard gate and
  restore correctness — no visual-gen perf CI bar is named (gap).
- **Generalizes to:** pattern-fast-path-silent-fallback — a cheaper restore
  path exists but a whole dtype class silently takes the generic slow path;
  carries to LoRA merge/unmerge round-trips in other pipelines that
  recompute instead of snapshotting, dequant->apply->requantize round-trips
  used where a saved copy would do, memory-thresholded fast paths that
  silently disable on smaller GPUs (watch the gate's log line), and
  optimizations shipped for quantized weights but missing the plain-dtype
  path (here FP16/FP32 dense weights still restore by subtraction).
