---
id: case-trtllmgen-mla-decode-unwaived-slower
type: regression-case
family: kernel-and-fusion
module: attention-fmha
maturity: full
regression_class: [kernel-selection-regression]
signals: [slower-kernel-in-trace, throughput-drop, perf-ci-bar-failure]
subsystems: [attention-kernel, runtime-python]
introduced_via: [config-default-change]
phase: [decode]
patterns: [pattern-variant-misroute, pattern-kernel-swap-regressed]
nvbugs: ["6430674"]
commits: ["9fe5d0f735a4"]
success_prs: [16167]
failed_prs: []
---

# Unwaiving a "now supported" TRTLLM-Gen MLA decode kernel routed R1 FP8-KV decode to a slower path

> Part of the [Attention & FMHA regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6430674` · commit `9fe5d0f735a4` · PR #16167 —
  "Surgical revert of PR #15838's dispatch guard removal: re-add
  MISSING_MLA_GENERATION_KERNELS and restore tokens_per_block".
- **Symptom:** `output_token_throughput` (higher-is-better) on the
  aggregated perf-sanity case `r1_fp4_v2_tep8_mtp3-con32_iter12_1k1k`,
  DeepSeek-R1 FP4 TEP8 MTP3 1k/1k con32 on GB200: good `045705139d`
  5641.5 tok/s → bad `8c1b230c81` 5184.46 tok/s, −8.10% (3-rep medians,
  cross-node CV 1.0%; per the NVBug). Single-commit source-built
  confirmation put the step at −6.73% (parent `b50a10983c` 5712.18 →
  culprit 5327.78). Surfaced as a `main`-branch perf-sanity CI regression
  on GB200. The NVBug lists five sibling cases with the same shape
  signature — GB300, B200 and GB200, DeepSeek-R1-FP4 and
  Kimi-K2.5-thinking-FP4, aggregated plus disagg `ctx_only` / `gen_only`
  — and the fix recovered all six, from +4.74% to +37.55% (re-measured on
  all six cases after the fix, per the NVBug).
- **Root cause:** the MLA decode support predicate stopped keying on
  `tokens_per_block`, so a shape that had been deliberately excluded
  became eligible. PR #15838 removed the
  `MISSING_MLA_GENERATION_KERNELS = {(576, 512, 32)}` class attribute and
  dropped the `tokens_per_block` parameter from
  `FlashInferTrtllmGenFmha._check_mla_generation_support`
  (`tensorrt_llm/_torch/attention_backend/fmha/flashinfer_trtllm_gen.py`).
  Post-PR the check returns `True` for DeepSeek-family MLA
  (`head_dim_qk=576`, `head_dim_v=512`, `tokens_per_block=32`) on SM100,
  so `FlashInferTrtllmGenFmha.run_mla_generation` is dispatched — the
  test's `kv_cache_config.dtype: fp8` taking the FP8-KV branch the same
  PR added. That TRTLLM-Gen decode kernel is capable on this shape but
  ~6.7% slower end-to-end than the FMHA path that previously served it.
- **How introduced:** culprit commit `4860595da3` / PR #15838
  "[None][fix] Unwaive supported attention backend cases" (merged
  2026-07-08). The waiver was removed on *capability* evidence, not perf
  evidence — the author re-enabled it because flashinfer had gained
  support for it, without measuring it.
- **Fix mechanism:** 13 added lines in one file — re-adds the shape gate
  as `SLOWER_MLA_GENERATION_KERNELS = {(576, 512, 32)}` and restores the
  `tokens_per_block` parameter on `_check_mla_generation_support` and on
  its caller in `_is_supported_with_reason`, so the
  `(head_dim_qk, head_dim_v, tokens_per_block)` triple reports
  unsupported with the reason "[Generation][MLA] slower TRTLLM-GEN decode
  kernel for headDimQk=…, headDimV=…, tokens_per_block=…" and MLA decode
  falls back to FMHA. Every other runtime change from PR #15838 is left
  intact. The set was proposed as `MISSING_MLA_GENERATION_KERNELS` and
  renamed during review at the culprit PR author's request ("This kernel
  is not missing, just slower"), which is why the landed name and reason
  string differ from the PR description.
- **Detection signal:** the guard is data, so check whether it is present
  and whether the predicate still keys on block size:
  `grep -n "SLOWER_MLA_GENERATION_KERNELS\|tokens_per_block" tensorrt_llm/_torch/attention_backend/fmha/flashinfer_trtllm_gen.py`
  — an absent set, or a `_check_mla_generation_support` signature without
  `tokens_per_block`, means every R1-shaped MLA decode is going to
  TRTLLM-Gen. Confirm the dispatch empirically from the kernel names in
  the JIT-warmup log: with the guard active, MLA decode runs
  `fmhaSm(100|103)aKernel_QkvE4m3OBfloat16HQk576HV512…ForGen` (the FMHA
  fallback); without it only the TRTLLM-Gen path appears (smoke test, per
  the NVBug). Metric shape is a clean step down
  that stays down, and it moves the whole shape family at once — every
  case combining MLA `head_dim_qk=576` / `head_dim_v=512`,
  `tokens_per_block=32`, `attn_backend=TRTLLM` and
  `kv_cache_config.dtype=fp8` on Blackwell regresses together.
- **Prevention/guard:** the fix *is* the guard, and that is the weakness —
  a set of blocked shape triples in the dispatcher can be deleted in one
  line with nothing failing. PR #16167 adds no unit test or perf bar
  pinning `(576, 512, 32)` off the TRTLLM-Gen decode path, and the
  reviewing bot's suggestion of an inline comment naming the regression
  and PR #15838 was not taken (the reason string is the only in-code
  explanation of why the entry exists). The durable half of the fix is
  the rename: a gate called `MISSING_…` invites removal as soon as the
  kernel exists, while `SLOWER_…` states the real, still-valid reason.
  Remaining gap: an "unwaive now-supported cases" PR is not required to
  run the affected perf-sanity stages — here the GB200 2-node PerfSanity
  stages were run only on the fix PR, after the regression had shipped.
- **Generalizes to:** `pattern-variant-misroute` and
  `pattern-kernel-swap-regressed` — a support/eligibility predicate is
  perf policy as much as capability policy, so relaxing it silently
  re-routes a shape to a capable-but-slower implementation. Carries to:
  "unwaive / re-enable now-supported" cleanups of backend support tables
  (attention backends, MoE backends, quantization paths), where the
  upstream library gaining support says nothing about which path is
  faster; support predicates that lose a parameter in a refactor (here
  `tokens_per_block`) so a coarser key matches more shapes than the
  original exclusion covered; newly added dtype branches (FP8 KV) that
  widen an existing dispatch decision onto shapes nobody re-measured;
  and any guard whose name describes capability while its real reason is
  performance.
