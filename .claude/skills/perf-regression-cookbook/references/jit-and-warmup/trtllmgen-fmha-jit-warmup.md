---
id: case-trtllmgen-fmha-jit-warmup
type: regression-case
family: kernel-and-fusion
module: jit-and-warmup
maturity: full
regression_class: [warmup-jit-gap]
signals: [midrun-stall, itl-increase, throughput-drop]
subsystems: [attention-kernel]
introduced_via: [pre-existing-gap]
phase: [any-phase]
patterns: [pattern-warmup-coverage-gap]
nvbugs: ["6185446", "6193854"]
commits: ["6dee1673737f"]
success_prs: [14851]
failed_prs: [15321]
---

# trtllm-gen FMHA kernels JIT-compile in the middle of serving (round 1: no warmup at all)

> Part of the [JIT & warmup regression cookbook](index.md) · schema: [case-template](../case-template.md)

**Round 1 of 2.** This case is "trtllm-gen FMHA had no JIT warmup"; the
follow-up [densify-grid case](trtllmgen-fmha-densify-grid.md) (#15305) is
"the warmup grid this PR added was too sparse". Same failure mode, two
rounds — read both before proposing a warmup change to this kernel family.

- **Provenance:** nvbugs `6185446` / `6193854` · commit `6dee1673737f` ·
  PR #14851 — "[https://nvbugs/6185446][fix] Add warmup for trtllm-gen fmha
  JIT kernels". Nvbug `6193854` is the tag on the follow-up PR #15321 below,
  whose description states #14851 had already removed the #13505 logic it
  targeted. A different warmup defect — the token-cap revert #14252 — is
  documented in [warmup-token-cap-revert](warmup-token-cap-revert.md) (nvbug
  6185713). Cross-reference the two, do not fold them.
- **Failed attempts:**
  - PR #15321 — repair-bot-authored follow-up on nvbug 6193854 (created
    2026-06-12, four days *after* #14851 merged; closed unmerged 2026-08-19;
    `+1/-1` in `fmhaKernels.h`) adding intermediate seq-len candidates `256` and `3072`
    to `kDefaultWarmupSeqLenQkvCandidates` · never merged and superseded: its
    own body records that "#14851 already removed the bad
    `is_sliding_window`/`mMaxSeqLenKv` logic on `origin/main`", so only the
    two extra candidates were new, and the general problem of a too-sparse
    candidate list was then solved properly by merged #15305
    ([round 2](trtllmgen-fmha-densify-grid.md)), which derives the grid from
    autotuner structure instead of hand-adding points. #15321's own body
    reports a recovery (`total_token_throughput` 1.855e+04 bad / 2.207e+04
    good / 2.157e+04 after fix), so a measured recovery is *not* evidence a
    PR landed — check state before re-proposing this.
- **Symptom:** Multi-second stalls during serving — each trtllm-gen FMHA
  kernel compilation takes 7-9 s (PR #14851) — showing as sporadic slow
  iterations and dropped throughput. PR #14851's table counts up to 8
  runtime JIT compilations without warmup on dsv3-bf16-mtp/dsr1-fp4-mtp.
  Why a one-time cost reads as a mean shift on a CI bar: on a short,
  time-limited bench case a multi-second JIT host bubble is a visible fraction
  of total wall time, not an outlier iteration.
- **Root cause:** trtllm-gen FMHA kernels are NVRTC-JIT-compiled per kernel
  variant on first use; the autotuner picks variants based on
  `batchSize`/`seqLenQ`/`seqLenKv` (via tile counts), so shapes first seen
  during serving trigger compilation on the hot path. No warmup pass
  exercised the JIT-triggering codepath at all — the general-warmup and
  autotuner-warmup passes did not route through it.
- **How introduced:** Pre-existing gap in the trtllm-gen JIT design (no
  warmup existed). An earlier mitigation, PR #13505's shape padding /
  `mMaxSeqLenKv` pinning, is explicitly reverted by PR #14851; PR #15321
  names that same #13505 (`ebf19a49`) as the root cause of its Qwen3
  regression.
- **Fix mechanism:** PR #14851 adds `runJITWarmupGridIfRequested` in
  `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h`, driven by
  warmup forward passes from `model_engine.py` (`_run_attention_warmup`),
  compiling every kernel the grid reaches before serving; it also logs any
  suspicious runtime compile.
- **Detection signal:** a mid-run multi-second gap in nsys with no kernel
  running, plus the runtime-JIT log line added by the fix:
  `grep "Possible JIT Cache Missing" <serve log>`; warmup activity shows as
  `grep "TRTLLM-Gen FMHA JIT warmup" <serve log>`. On a short bench case look
  for a `gpu_time` / total-wall mean shift rather than an outlier iteration.
- **Prevention/guard:** PR #14851 added the `Possible JIT Cache Missing`
  TLLM_LOG_WARNING for any runtime `generateAndCompileKernel` taking over
  1000 ms, and a check that JIT warmup never runs during CUDA graph
  capture. Treat "kernel is JIT-compiled" as a first-class warmup attribute:
  any kernel that calls into a runtime JIT must ship a warmup entry in the
  same PR. Gap: no automated test asserts that every runtime-selectable
  kernel variant is reachable from the warmup grid — which is exactly how
  [round 2](trtllmgen-fmha-densify-grid.md) happened.
- **Generalizes to:** pattern-warmup-coverage-gap — carries to any
  TorchInductor / Triton / DeepGEMM / cutlass-JIT kernel added without a
  warmup entry, to CUDA-graph capture lists missing a runtime batch size
  (eager fallback), to autotuner caches that tune-on-first-use inside
  serving, and to backends whose first-touch cost is seconds.
