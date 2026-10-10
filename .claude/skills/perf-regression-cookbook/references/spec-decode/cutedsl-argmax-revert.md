---
id: case-cutedsl-argmax-revert
type: regression-case
family: kernel-and-fusion
module: spec-decode
maturity: full
regression_class: [kernel-swap-regressed]
signals: [slower-kernel-in-trace, throughput-drop]
subsystems: [spec-decode]
introduced_via: [kernel-change]
phase: [decode]
patterns: [pattern-kernel-swap-regressed]
nvbugs: ["5853720", "5853556"]
commits: ["eac56b793ea4"]
success_prs: [11403]
failed_prs: []
---

# CuTe-DSL argmax kernel regressed spec-decode sampling vs torch.argmax

> Part of the [Speculative decoding regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbugs `5853720` + `5853556` · commit `eac56b793ea4` ·
  PR #11403 — Disable cutedsl argmax kernel to fix perf regression.
  `5853556` (a performance bug against DSR1 FP4 DEP aggregated serving, now
  closed as verified) is the same defect on DSR1 rather than DSV32 and was
  marked as a duplicate of 5853720 because it shares the same root cause.
  One root cause, one fix PR — hence one case.
- **Symptom:** perf regression in speculative decoding after the draft-token
  sampling path switched from `torch.argmax` to a CuTe-DSL argmax kernel
  (per the PR: revert "to investigate perf regression from commit
  df8be0c50"). The PR names no metric, model or hardware; nvbug 5853720
  does — three post-merge rows against regression commit
  `80dd6e70c689` (Jan 30) on DSV3.2 FP4 DEP + MTP1:
  `v32_fp4_dep4_mtp1_1k1k-con1024_iter10_1k1k` on GB200 −17.52 %
  d_token_throughput (11709.65 vs 14196.49),
  `v32_fp4_dep4_mtp1_8k1k-con256_iter10_8k1k` on GB200 −9.54 %, and
  `v32_fp4_dep8_mtp1_8k1k-con256_iter10_8k1k` on B200 −10.50 % — all against
  a 5 % threshold.
- **Root cause:** the `cute_argmax` path in `SpecWorkerBase`
  (`tensorrt_llm/_torch/speculative/interface.py`) was slower end-to-end
  than the `torch.argmax(logits, dim=-1)` it replaced; the CuTe path also
  returns an `(M, 2)` value/index tensor that needs a `[:, 1].long()`
  slice-and-cast to extract token ids. The PR itself reverts "to
  investigate", but the mechanism *was* pinned down on the bug afterwards:
  the CuTe-DSL argmax kernel only works for fp32 input, and in MTP
  spec-decode the two call sites differ —
  `_sample_tokens_for_batch` (target) sees `torch.float32` while
  `_draft_sampler_greedy` (draft) sees `torch.bfloat16`. So the draft-side
  swap is the one that hurt, and the proposed narrow fix on the bug is a
  dtype gate (`if logits.dtype == torch.float32:` → cutedsl, else
  `torch.argmax`) rather than a blanket revert.
- **How introduced:** commit `df8be0c50c` / PR #10476
  "[TRTLLM-10276][feat] Integrate cutedsl argmax kernel" added
  `tensorrt_llm/_torch/cute_dsl_kernels/argmax.py` and swapped both
  `torch.argmax` call sites in `SpecWorkerBase` (draft-token sampling and
  the greedy branch of target sampling) to `cute_argmax`.
- **Fix mechanism:** exact revert of the two call sites back to
  `torch.argmax(logits, dim=-1)` and removal of the `cute_argmax` import;
  the kernel itself (`cute_dsl_kernels/argmax.py` and its unit test)
  stays in-tree, just unused. Two caveats recorded on the bug. (1) The
  revert did not recover every row: after it, the GB200 1K/1K regression
  did not recover while the 8K/1K case improved (even beyond the Jan 27
  baseline) — the 1k1k GB200 row had a second contributor, and the bug was
  closed once later post-merge data showed perf recovered. (2) The
  dtype-gated re-enable is **still not on `main`**: PR #11466
  "[https://nvbugs/5853720] [Fix] use cutedsl argmax only for fp32 dtype
  input" (`+9/-2` in `speculative/interface.py`) has been `OPEN` since
  2026-02-12 with two approvals and never merged. It is *not* listed in
  `failed_prs` because it is an open attempt that may still land — but its
  own measurement is why nobody pushed it: DeepSeek-R1-0528-FP4-v2 on B200
  TP4/EP4 MTP1 8k/1k con256 moved 4277.31 → 4287.24 output tok/s,
  **+0.23 %**, i.e. neutral. Re-enabling this kernel needs a workload where
  fp32 argmax is actually hot, not another gate.
- **Detection signal:** nsys decode-step trace shows a CuTe-DSL argmax
  kernel (plus the extra slice/cast ops) where a native torch argmax kernel
  ran in the good build; confirm which path is wired with
  `grep -n "cute_argmax\|torch.argmax" tensorrt_llm/_torch/speculative/interface.py`.
- **Prevention/guard:** the fix adds no guard. The introducing PR shipped
  both a correctness unit test and a standalone CUDA-event microbenchmark
  vs `torch.max` (`test_argmax_performance` in
  `cute_dsl_kernels/test_argmax.py`); the gap is that the isolated kernel
  microbenchmark did not capture the end-to-end serving path (per-step
  launch/wrapper overhead plus the extra `[:, 1].long()` slice/cast), and
  no e2e spec-decode perf bar gated the merge.
- **Generalizes to:** `pattern-kernel-swap-regressed` — a replacement
  kernel is itself slower than the op it displaces; carries to other
  DSL-generated kernels (CuTe/Triton) substituted for tuned torch/cuBLAS
  ops, swaps whose new output layout forces extra slice/cast/copy ops
  around the kernel, sampling-path micro-ops where per-step launch and
  wrapper overhead dominates, and any "integrate new kernel" feature PR
  that lands without a perf comparison on the exact serving path.
