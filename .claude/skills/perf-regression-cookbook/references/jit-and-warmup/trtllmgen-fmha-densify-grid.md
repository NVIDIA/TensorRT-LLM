---
id: case-trtllmgen-fmha-densify-grid
type: regression-case
family: kernel-and-fusion
module: jit-and-warmup
maturity: full
regression_class: [warmup-jit-gap]
signals: [midrun-stall, throughput-drop, ttft-increase]
subsystems: [attention-kernel]
introduced_via: [incomplete-coverage]
phase: [any-phase]
patterns: [pattern-warmup-coverage-gap]
nvbugs: ["6293823", "6315845"]
commits: ["7e243650e8dd"]
success_prs: [15305]
failed_prs: [15279, 15472]
---

# trtllm-gen FMHA warmup grid too sparse for a narrow seqlen band (round 2)

> Part of the [JIT & warmup regression cookbook](index.md) · schema: [case-template](../case-template.md)

**Round 2 of 2.** Round 1 —
[trtllm-gen FMHA JIT warmup](trtllmgen-fmha-jit-warmup.md) (#14851) — added
the warmup pass. This case is the residual coverage hole *in the grid that PR
added*. The rejected-direction record below is the important half: two
separate attempts (one closed, one still open) tried to hide the hole by
re-pinning `mMaxSeqLenKv`, and that direction was rejected (per nvbug 6293823).

- **Provenance:** nvbug `6293823` and its NVBugs-marked dupe `6315845` ·
  commit `7e243650e8dd` · PR #15305 — Densify trtllm-gen fmha warmup grid to
  catch missing kernels. Both bugs report a B200 DSR1 FP8 aggregated TP8
  regression on `aggr_upload-deepseek_r1_fp8_blackwell-r1_fp8_tp8_mtp3_8k1k`,
  both bisected to the same `b4d44d33baea` → `15d06c0923b6` pair (i.e. PR
  #14851) and both closed by #15305 (per nvbug 6293823).
- **Failed attempts:**
  - PR #15279 (nvbug 6293823) — instead of widening the warmup grid, it
    restored PR #13505's one-line MLA-generation override, re-pinning
    `mMaxSeqLenKv` to `generation_params.max_attention_window_size` in
    `mlaGeneration()` (`cpp/tensorrt_llm/common/attentionOp.cpp`, +9/−1, the
    only file touched) so runtime kv-len variability collapses onto the single
    shape warmup already covers, keeping all of #14851's warmup framework and
    cache-miss warning intact; it names the uncovered variant as
    (HVPerCta256, MultiCtasKv-noCga) selected at kv-len ≈ 8199 — the same
    narrow band #15305 cites — costing ~20 s per mid-benchmark compile and
    38.2 % of `output_token_throughput` (615.3 bad / 992 good / 990.2 after
    fix) · **closed unmerged**, with no reason on the GitHub thread (the only
    PR comment is CodeRabbit's, no human review); the reason is recorded on
    nvbug 6293823 instead — #15279 would resolve the issue but is not
    recommended, because the pin bypasses kernel selection for the specific
    seqlen and can cause problems on smaller seqlens, which is why #14851
    removed it in the first place; the bug was then fixed by densifying the
    grid (#15305).
  - PR #15472 (nvbug 6315845) — **still OPEN on GitHub, but superseded**: it
    targets the dupe of a bug already closed by the merged #15305, and it
    re-treads exactly the direction rejected above, widened to two call
    sites — pin `mMaxSeqLenKv = max_attention_window_size` in
    `mlaGeneration()` *and* pin
    `(mQkvLayout == PagedKv) ? max_attention_window_size : max_past_kv_length`
    in `XqaDispatcher::runImpl`'s non-spec-dec-tree branch
    (`cpp/tensorrt_llm/common/attentionOp.cpp`,
    `cpp/tensorrt_llm/kernels/xqaDispatcher.cpp`, +18/−2, C++-only, no
    warmup-grid change). Its own framing is that #14851 removed "protective
    code" in both locations and that "prior attempts addressed only
    `mlaGeneration` (or only the warmup grid)". It reports TTFT P99 882 ms →
    25 s with TPOT/ITL within 1 %, and `output_token_throughput`
    609 bad / 992.5 good / 988.9 after fix. Authored by repair-bot.
  - **Read the pair as one lesson:** pinning `mMaxSeqLenKv` back to the cache
    capacity hides the grid hole at the cost of the kernel choice for *small*
    seqlens. Widening warmup coverage is the accepted direction; re-proposing
    the pin — at one call site or at two — is a re-tread. An agent that
    rediscovers "#14851 removed protective code, restore it" has rediscovered
    #15279, not a new fix.
- **Symptom:** a kernel variant selected only when seqLenKv lands in the
  narrow band `8193-9316` was absent from #14851's warmup grid, so live
  traffic hitting that band JIT-compiled inline — ~20 s per compile. On a
  deterministic perf-sanity case the hole reads as a plain mean shift rather
  than variance, which is how these two bugs were filed: B200 post-merge
  `r1_fp8_tp8_mtp3_8k1k-con4_iter10_8k1k`, `output_token_throughput`
  996.6 → 824.9 (a stated 15 % / −17.23 % gap) and *staying there*, because
  every rep serves the same uncovered kv-len. In a latency view the same hole
  is a TTFT P99 blowup with decode untouched (#15472: 882 ms → 25 s, TPOT
  within 1 %).
- **Root cause:** the autotuner selects a trtllm-gen kernel from
  `batchSize`/`seqLenQ`/`seqLenKv` via *tile counts*; selection is sensitive
  when the tile count is in the 1-24 range, so a small seqLenKv band can map
  to a different tile shape than either neighbouring band. #14851's grid
  loosely sampled seqlen and skipped the 8193-9316 band, so the variant was
  reachable at runtime but not at warmup — the selector's decision boundaries
  are finer than the warmup sampling grid.
- **How introduced:** by PR #14851 itself (see
  [round 1](trtllmgen-fmha-jit-warmup.md)) — the grid it shipped enumerated
  the likely-common buckets; the narrow band is a real code path outside the
  enumerated set.
- **Fix mechanism:** PR #15305 rebuilds the grid from autotuner structure
  rather than from hardcoded shape tuples: dense tile counts 1-24 plus sparse
  26-256, multiplied by tile sizes {128, 256, 512}, in
  `kDefaultWarmupSeqLenKvCandidates` and denser
  `kDefaultWarmupBatchSizeCandidates` (~1 s measured added tuning overhead).
- **Detection signal:** same mid-run stall signature as round 1, but the JIT
  lands at iter N > 0 when the offending shape first appears:
  `grep "Possible JIT Cache Missing" <serve log>`. Compare grid coverage
  against the seqlen distribution actually served —
  `grep -n "kDefaultWarmupSeqLenKvCandidates" cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h`
  against the seqlen histogram from the bench log.
- **Prevention/guard:** derive the warmup grid *from the autotuner's own
  selection function* rather than hardcoding shape tuples; any addition to the
  autotuner's kernel-selection space must extend warmup coverage in the same
  PR. Gap: still no automated test asserts that every runtime-selectable
  variant is reachable from the warmup grid — round 2 was found on one
  particular DSR1 case (per PR #15305), not by a coverage test.
- **Generalizes to:** pattern-warmup-coverage-gap — carries to autotuner-backed
  MoE variant selection, TorchInductor tuning of matmul epilogues, cutlass-JIT
  with shape-conditional kernel choice, and Triton kernels whose best-tile
  choice varies non-monotonically with shape. Whenever a *selector* is finer
  grained than the *warmup sampler*, expect a hole.
