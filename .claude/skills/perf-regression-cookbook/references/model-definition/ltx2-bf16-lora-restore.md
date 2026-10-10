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
  Save LTX-2 BF16 weights to speed up perf. Closed as verified.
  **Filed as a functional bug, and that is a misfiling — do not delete this
  case on the strength of the severity field.** The bug itself reports a perf
  regression in the LTX2 two-stage case, and its description is a quantified,
  bisected regression: all 40 matched LTX2 two-stage configs regressed 30–50%
  (median −31%, worst −51%) between visual-gen development-branch commits
  4c84e1785 (Apr 27) and a21f7945a (May 13), with one-stage configs
  unaffected (median +1.0%) and a 3-point comparison separating the cluster
  move from the framework delta (<old cluster>→<new cluster> at the same SHA = +17%, 16.49 →
  19.35 s; framework 4c84e→a21f7 on the same <new cluster> = +50%, 19.35 → 29.09 s)
  across a 106-commit bisect range. This cookbook uses a bug's severity as its
  perf-vs-functional discriminator, but the rule is "severity, *when it and
  the bug body agree*": where the bug's own title and body describe a
  quantified, bisected perf regression, the body wins, and the discrepancy is
  recorded here so the next audit does not re-litigate it.
- **Symptom:** Slow LTX-2 two-stage visual-gen pipeline perf around the
  stage-2 distilled-LoRA merge/restore: BF16 weights touched by LoRA were
  restored by re-subtracting the deltas after stage 2 instead of a snapshot
  copy ("the slower on-the-fly subtract path" per the PR). Surfaced via
  nvbug perf investigation.
- **Root cause:** In `_apply_lora_deltas`
  (`tensorrt_llm/_torch/visual_gen/models/ltx2/pipeline_ltx2_two_stages.py`)
  only quantized (FP8/FP4) parameters were snapshotted into
  `saved_lora_state`; dense BF16/FP16/FP32 weights were never saved, so
  restore after stage 2 always went through
  `_subtract_dense_lora_deltas` — a per-parameter delta cast + subtract —
  even when GPU memory could hold a BF16 snapshot.
- **How introduced:** prior-fix side effect. **The culprit is PR #13244**
  ("[None][fix] Use bf16 for LTX-2 FP4 stage 2", merged 2026-04-30 to `main`,
  touching the same `pipeline_ltx2_two_stages.py` this fix touches). The fix PR
  itself names no regressing commit — the bug does, and that is the correction
  worth carrying: before #13244 the pipeline snapshotted the original dense
  weights, and #13244 made stage 2 merge distilled LoRA into the BF16 diffusion
  transformer and restore by subtracting deltas from **1178 dense transformer
  parameters on the request path**, which is the ~40–50% cost. #13244 was not
  careless: the snapshot path clones almost the whole LoRA-touched BF16
  transformer and raised peak GPU memory from ~72.9 GiB to ~108.3 GiB (+35.4
  GiB), so it traded latency for memory deliberately. That makes this a
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
