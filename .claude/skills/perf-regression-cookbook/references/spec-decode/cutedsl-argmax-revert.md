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

- **Provenance:** nvbugs `5853720` / `5853556` · commit `eac56b793ea4` ·
  PR #11403 — "[https://nvbugs/5853720][fix] Disable cutedsl argmax kernel to
  fix perf regression". One fix PR — hence one case.
- **Symptom:** perf regression in speculative decoding after the draft-token
  sampling path switched from `torch.argmax` to a CuTe-DSL argmax kernel
  (per the PR: revert "to investigate perf regression from commit
  df8be0c50"). The PR names no metric, model or hardware; do not attach a
  number to it.
- **Root cause:** the `cute_argmax` path in `SpecWorkerBase`
  (`tensorrt_llm/_torch/speculative/interface.py`) was slower end-to-end
  than the `torch.argmax(logits, dim=-1)` it replaced; the CuTe path also
  returns an `(M, 2)` value/index tensor that needs a `[:, 1].long()`
  slice-and-cast to extract token ids. PR #11403 itself reverts "to
  investigate"; the follow-up PR #11466 pins the mechanism in its diff and
  thread: the kernel is meant for fp32 logits (its author: fp32 is "the only
  precision this kernel can support"), and "for DeepSeek, the MTP draft uses
  BF16, which was causing problems". So the draft-side swap
  (`_draft_sampler_greedy`) is the one that hurt, and #11466's narrow fix is a
  dtype gate at both call sites (`if logits.dtype == torch.float32:` →
  cutedsl, else `torch.argmax`) rather than a blanket revert.
- **How introduced:** commit `df8be0c50c` / PR #10476
  "[TRTLLM-10276][feat] Integrate cutedsl argmax kernel" added
  `tensorrt_llm/_torch/cute_dsl_kernels/argmax.py` and swapped both
  `torch.argmax` call sites in `SpecWorkerBase` (draft-token sampling and
  the greedy branch of target sampling) to `cute_argmax`.
- **Fix mechanism:** exact revert of the two call sites back to
  `torch.argmax(logits, dim=-1)` and removal of the `cute_argmax` import;
  the kernel itself (`cute_dsl_kernels/argmax.py` and its unit test)
  stays in-tree, just unused. Caveat: the dtype-gated re-enable is **still
  not on `main`**: PR #11466
  "[https://nvbugs/5853720] [Fix] use cutedsl argmax only for fp32 dtype
  input" (`+9/-2` in `speculative/interface.py`) has been `OPEN` since
  2026-02-12 with approvals and never merged. It is *not* listed in
  `failed_prs` because it is an open attempt that may still land — but its
  own measurement in the PR thread is why nobody pushed it: DeepSeek-R1-0528-FP4-v2 on B200
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
