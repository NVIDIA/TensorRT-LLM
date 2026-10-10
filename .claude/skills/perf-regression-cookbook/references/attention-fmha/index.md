# Regression Cookbook — Attention & FMHA

This module is the attention path and the code that picks a kernel for it:
the trtllm-gen FMHA kernels and their dispatcher, the MLA decode kernels, and
the DeepSeek DSA sparse-attention indexer (`tensorrt_llm/_torch/modules/sparse/`,
`cpp/.../fmha*`). Regressions here arrive in two shapes. Either the *dispatcher*
picks a different kernel than it used to — a support guard was relaxed or a
specialization stopped matching, so a capable-but-slower variant runs with no
error anywhere — or the *host* side of the attention path grows: per-layer
`.item()` syncs, Python view bookkeeping, driver queries per dispatch, or an
auxiliary computation repeated on every TP rank. First thing to check: diff the
attention kernel names in an nsys trace against a good build, and if the kernel
inventory is unchanged, compare the host spans instead.

## Recurring patterns in this module

- **Fast path silently fell back** — an eligibility/specialization check in the
  FMHA dispatcher stops matching (a new tile, a new dtype, a shape outside the
  instantiated set) and a generic kernel runs, correct but slower. The tell is a
  kernel *name* missing from the trace rather than a kernel getting slower.
  _(Instance: DS-V3.2 FP4 sparse context attention lost its 2-CTA kernel.)_
- **Variant misroute** — support tables are perf policy. Removing a waive or
  widening a capability predicate makes shapes newly *eligible* for a kernel
  that is capable but not fastest for them; re-measure every shape an "unwaive"
  admits. _(Instances: the unwaived TRTLLM-Gen MLA decode kernel; the 2-CTA
  fallback, which is also a dispatch-eligibility defect.)_
- **Host work on the hot path** — attention dispatch runs per step, so anything
  synchronous inside it multiplies: a CUDA-driver SM-count query per dispatch, or
  the DSA indexer's Python-side view/index work.
  _(Instances: FMHA dispatcher SM-count query; DSA indexer host overhead.)_
- **Per-step sync added** — the DSA indexer took `.item()`-style host reads on
  the per-iteration path, so the GPU idles waiting on the host with kernel
  durations unchanged. _(Instance: DSA indexer host overhead.)_
- **Redundant work across ranks** — the indexer's prefill computation was run
  in full on every attention-TP rank instead of split, so TTFT scaled the wrong
  way with TP. _(Instance: DSA indexer TP-duplicated prefill.)_
- **Kernel swap regressed / below an alternative** — the flip side of the
  misroute: the newly-selected MLA decode kernel was genuinely functional, just
  slower than the one it displaced on those shapes, so the fix is to restore the
  waive rather than to repair anything. Compare the two candidate kernels at the
  affected shape before assuming the new one is a bug.
  _(Instance: the unwaived TRTLLM-Gen MLA decode kernel.)_

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [DSA indexer host overhead and per-step syncs](dsa-indexer-host-overhead.md) | ~65% of per-iteration GPU time in DSA-launched kernels; 2× 12–15 ms `cudaStreamSynchronize` per context step | host-work-added, sync-introduced |
| [DSA indexer prefill duplicated on every TP rank](dsa-indexer-tp-duplicated-prefill.md) | long-context chunked-prefill TTFT; up to 2.3× TTFT improvement at 150k ISL, TEP8, 8×B200 | redundant-cross-rank-work |
| [DS-V3.2 FP4 sparse context attention stopped selecting the 2-CTA FMHA kernel](dsmla-sparse-ctx-2cta-fallback.md) | ctx-only throughput 16793 → 14533 tok/s (−13.4%) on a GB200 <cluster> | fast-path-fallback, kernel-selection-regression |
| [FMHA dispatcher queries SM count from the driver every step](fmha-dispatcher-sm-count-query.md) | host overhead on `llama70b_fp4_tp4_512_32-con512_iter10_512_32` | host-work-added |
| [Unwaiving a "now supported" TRTLLM-Gen MLA decode kernel](trtllmgen-mla-decode-unwaived-slower.md) | `r1_fp4_v2_tep8_mtp3-con32_iter12_1k1k` on GB200: 5641.5 → 5184.46 tok/s (−8.10%) | kernel-selection-regression |
