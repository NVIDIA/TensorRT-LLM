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

- **Provenance:** nvbug `6185446` · commit `6dee1673737f` · PR #14851 —
  Add warmup for trtllm-gen fmha JIT kernels. The same PR also closed
  nvbug `6193854` (a Qwen3 regression of 5 %–153 % against 1.3.0rc13 across
  B200, GB200, B300 and GB300): #14851's title cites 6185446 only, so a
  PR→bug lookup never links it — the link lives only in 6193854 itself, which
  ties the regression to the same missing JIT warmup, names #14851 as its fix,
  and is closed as fixed and verified.
  Caution on 6193854: it is a **multi-root-cause QA sweep** over ~20 Qwen3
  rows, not a single defect. Earlier in its history a batch of rows was
  attributed to a different culprit and closed against the revert in #14252,
  after which perf recovered for some Qwen3 rows but not for the rest — the
  surviving rows are the ones re-triaged to #13505 and fixed here. #14252 is
  a *different* defect, documented in
  [warmup-token-cap-revert](warmup-token-cap-revert.md) (nvbug 6185713);
  whether 6193854 duplicates 6185713 was asked early on that bug and never
  answered. Cross-reference the two, do not fold them.
- **Failed attempts:**
  - PR #15321 — repair-bot-authored follow-up on nvbug 6193854 (`OPEN`,
    created 2026-06-12, four days *after* #14851 merged; `+1/-1` in
    `fmhaKernels.h`) adding intermediate seq-len candidates `256` and `3072`
    to `kDefaultWarmupSeqLenQkvCandidates` · never merged and superseded: its
    own body records that "#14851 already removed the bad
    `is_sliding_window`/`mMaxSeqLenKv` logic on `origin/main`", so only the
    two extra candidates were new, and the general problem of a too-sparse
    candidate list was then solved properly by merged #15305
    ([round 2](trtllmgen-fmha-densify-grid.md)), which derives the grid from
    autotuner structure instead of hand-adding points. Per nvbug 6193854,
    #15321 was nonetheless measured as a successful perf fix (B300,
    `qwen3_235b_a22b_fp8-…-con:256-ep:8-gpus:8`, 18545.36 → 21568.87
    total_token_throughput vs a 20309.79 threshold), so a measured recovery
    is *not* evidence a PR landed — check state before re-proposing this.
- **Symptom:** Multi-second stalls during serving — each trtllm-gen FMHA
  kernel compilation takes 7-9 s (PR #14851) — showing as sporadic slow
  iterations and dropped throughput. PR #14851's table counts up to 8
  runtime JIT compilations without warmup on dsv3-bf16-mtp/dsr1-fp4-mtp.
  nvbug 6193854 is the same gap seen on the CI bar rather than in a serve
  log: Qwen3 against 1.3.0rc13, `gpu_time` up 12.14 % (fp8, B300), 15.83 %
  (fp4, B300), 15.97 % (fp4, GB200-OCI), 28.99 % (fp4, B200) and 32.06 %
  (fp8, B200) on the `qwen3_235b_a22b_fp{4,8}-bench-pytorch-*-maxbs:512-maxnt:2048-input_output_len:1000,2000`
  cases (5 %–153 % across the whole bug). Why a one-time cost reads as a
  mean shift, per that NVBug: a one-time JIT host bubble only matters to
  time-limited tests, and some tests are short enough that its impact
  becomes obvious — so on a short bench case the JIT bubble is a fraction of
  total wall time, not an outlier iteration. Caution from the same bug: one
  of its rows (`disagg-e2e-gb200_qwen3-235b-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_con1_ccb-NIXL`,
  reported at 50.06 %) was later reclassified as *instability*, not
  regression, and triaged to cross-rack node variance — a Qwen3 row in this
  family is not automatically this defect.
- **Root cause:** trtllm-gen FMHA kernels are NVRTC-JIT-compiled per kernel
  variant on first use; the autotuner picks variants based on
  `batchSize`/`seqLenQ`/`seqLenKv` (via tile counts), so shapes first seen
  during serving trigger compilation on the hot path. No warmup pass
  exercised the JIT-triggering codepath at all — the general-warmup and
  autotuner-warmup passes did not route through it.
- **How introduced:** Pre-existing gap in the trtllm-gen JIT design (no
  warmup existed). An earlier mitigation, PR #13505's shape padding /
  `mMaxSeqLenKv` pinning, is explicitly reverted by PR #14851; nvbug 6193854
  bisected its Qwen3 regression to that same #13505 / `ebf19a496e1d`.
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
