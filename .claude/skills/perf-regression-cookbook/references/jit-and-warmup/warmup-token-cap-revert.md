---
id: case-warmup-token-cap-revert
type: regression-case
family: kernel-and-fusion
module: jit-and-warmup
maturity: full
regression_class: [warmup-jit-gap]
signals: [throughput-drop, perf-ci-bar-failure, midrun-stall]
subsystems: [autotuner, runtime-python]
introduced_via: [prior-fix-side-effect]
phase: [any-phase]
patterns: [pattern-warmup-coverage-gap]
nvbugs: ["6185713"]
commits: ["d42ec3df56a9"]
success_prs: [14252]
failed_prs: []
---

# Warmup token cap left large-token shapes untuned at runtime — revert

> Part of the [JIT & warmup regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6185713` · commit `d42ec3df56a9` · PR #14252 —
  Revert PR13758's code changes on Limiting maximum warmup token count.
  **The culprit is #13758**, which this PR reverts. #13758 fixed an int32 IMA
  at one shape (nvbug `5805494`; its PR describes an illegal-memory-access
  crash at init) and created this regression at another, so the two are the
  two ends of one tradeoff and neither is safe to re-apply alone.
  #13758 had its own case in the instability cookbook until 2026-08-12; it was
  removed because its PR describes a crash, not a varying metric, so this case
  is now the only cookbook record of the tradeoff — which is why the cap side
  is documented in full below rather than by reference.
  A *different* warmup defect is documented in
  [trtllmgen-fmha-jit-warmup](trtllmgen-fmha-jit-warmup.md); the two are
  cross-referenced here, deliberately not folded.
- **Symptom:** the revert PR states no number. A cap-shaped regression hits
  a *broad* row set — every config whose `max_num_tokens` exceeds the cap —
  which is itself the signal.
  Mechanism: the `max_warmup_tokens = 8192` cap added by #13758 protected the
  extreme attention_dp config from an int32 IMA at init — per PR #13758 /
  #15887: on the attention-DP AllGather MoE path, the trtllm-gen
  DeepSeek-FP8 block-scale MoE kernels computed global-memory offsets in
  32-bit arithmetic and overflowed into an illegal memory access — but
  silently dropped autotuner / warmup coverage of every shape between 8192 and
  `max_num_tokens`, so live traffic at those larger shapes JIT-compiled inline
  or hit an autotuner cache miss during graph capture. Because the affected
  disagg cases serve a fixed large ISL every rep, the lost coverage reads as a
  steady mean shift on the CI bar rather than as an outlier iteration.
- **Root cause:** a fixed *global* cap on warmup token count cannot
  simultaneously (a) stay below the smallest downstream int32 index limit
  and (b) cover the largest legitimate runtime shape; the two properties
  are at odds when `dp_size > 1` scales the downstream index.
- **How introduced:** prior fix #13758 ("Limit maximum warmup token count to
  prevent crash in autotuner", `8a11da8bff90`, merged 2026-05-15) capped
  `curr_max_num_tokens` at 8192 in `model_engine.py` (`+11/-2`) and shipped a
  133-line regression test with it (`+144/-2` total) — a well-tested change,
  not an unreviewed one-liner, which is why review did not catch it: the
  coverage-side cost is invisible to any test of the capped path. Its PR
  argued the cap was safe because the autotuner only validates tactics on
  profile buckets up to 8192 (`MAX_PROFILE_BUCKET`), so warmup shapes above
  it "yield no tuning benefit". The revert landed three days later.
- **Fix mechanism:** revert #13758's cap on the model-engine warmup /
  autotuner-warmup path so full coverage is restored (#14252, `+2/-144` —
  note it removes #13758's unit test too, so nothing on `main` now pins the
  cap's absence); the underlying int32 overflow needs a per-kernel clamp
  rather than a global warmup cap.
- **Detection signal:** autotuner cache-miss warnings during CUDA-graph
  capture on shapes between 8192 and `max_num_tokens`, or a midrun-stall
  when a large-token shape first arrives — inspect the warmup dump vs the
  live shape histogram; `grep -n 'max_warmup_tokens' tensorrt_llm/_torch/`
  before and after the revert to confirm the cap is gone.
- **Prevention/guard:** protective caps must be **per-kernel** (matching the
  downstream index bound) rather than a single global warmup cap. **The real
  fix has since landed:** PR #15887 "[https://nvbugs/5805494][fix] Use int64
  indexing in trtllm-gen block-scale MoE kernels" (`60ff8b7843d6`, merged
  2026-08-05, `+19/-12`, `blockScaleMoe/DevKernel.cu`) removed the overflow
  at its source instead of clamping the warmup grid — per its PR, the
  autotuner "still profiles the largest token bucket, so there is no
  tuning-driven performance regression". So the
  tradeoff this case documents no longer exists on `main`, and re-capping
  warmup would now buy nothing. The general lesson stands: when a crash is
  fixed by shrinking a warmup/coverage axis, treat it as a stopgap with a perf
  debt, and before landing any warmup cap diff the capped grid against the live
  shape histogram of the disagg perf-sanity rows above.
- **Generalizes to:** pattern-warmup-coverage-gap in its *cap-shrunk-grid*
  form — a warmup grid can lose coverage without anyone editing the grid, just
  by capping an axis. Carries to any global cap that trades safety at one
  shape against coverage at another, to CUDA-graph capture batch-size caps,
  and to fixes that add a broad protection without a scoped alternative.
