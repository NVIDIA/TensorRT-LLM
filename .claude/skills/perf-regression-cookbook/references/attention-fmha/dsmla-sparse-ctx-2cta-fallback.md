---
id: case-dsmla-sparse-ctx-2cta-fallback
type: regression-case
family: kernel-and-fusion
module: attention-fmha
maturity: full
regression_class: [fast-path-fallback, kernel-selection-regression]
signals: [slower-kernel-in-trace, throughput-drop, perf-ci-bar-failure]
subsystems: [attention-kernel, autotuner]
introduced_via: [new-feature, prior-fix-side-effect]
phase: [prefill]
patterns: [pattern-fast-path-silent-fallback, pattern-variant-misroute]
nvbugs: ["6106687"]
commits: ["80d157ae3581"]
success_prs: [13652]
failed_prs: []
---

# DeepSeek-V3.2 FP4 sparse context attention stopped selecting the 2-CTA FMHA kernel

> Part of the [Attention & FMHA regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6106687` · commit `80d157ae3581` · PR #13652 —
  `[None][feat] Add DeepSeekV4 attention kernels` (merged 2026-05-09). #13652 is
  the only PR linked on the bug, and the bug closed as fixed and verified on
  2026-05-11, two days later. The PR carries **no NVBug tag and an empty
  Description section**, so a PR→bug lookup never finds this regression: the
  link exists only in the NVBug, which also records that an upstream
  trtllm-gen autotuner logic change had been merged and was expected to fix
  the regression once integrated.
- **Symptom:** output/total token throughput down on `main` for two ctx-only
  DeepSeek-V3.2 FP4 cases on a GB200 <cluster>,
  `ctx_only-gb200_deepseek-v32-fp4_32k4k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb`
  (both UCX and NIXL) and
  `ctx_only-gb200_deepseek-v32-fp4_8k1k_con4096_ctx1_dep4_gen1_dep32_eplb256_mtp0_ccb-UCX`.
  Per the NVBug, the gap is 15 % between good `4e69c14f732a` and bad
  `7a8bd87f6a08` (249 commits apart); a bisect measured 16756.89 → 14616.82
  tok/s (−12.8 %) and a later run of the same case 16793 → 14533 tok/s
  (−13.4 %, pass threshold ≥ 16289 = baseline × 0.97). Surfaced as a post-merge
  perf-tracking regression on the GB200 sweep, not from a profile.
- **Root cause:** trtllm-gen FMHA kernel *selection* for DeepSeek MLA
  token-sparse attention stopped choosing the 2-CTA cooperative kernel
  (`HVPerCta=256` + 2-CTA MMA) and ran the single-CTA variant
  (`HVPerCta=128`) instead. The nsys kernel name on the bug changes from
  `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta256PagedKvSparseP1VarSeqQ64Kv128Persistent2CtaKeepsAbForGen`
  (good) to
  `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvStaticTokenSparseP1VarSeqQ64Kv128PersistentKeepsAbForGen`
  (bad) — same MLA head dims, no 2-CTA cluster. Two changes compose: the FMHA
  dispatch predicate was broadened so DeepSeek-V3.2 *context* attention runs
  generation-style sparse kernels, and a crash workaround in
  `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h` then forces
  `MultiCtasKvMode::GmemReduction` back to `Disabled` whenever
  `params.multiCtasKvScratchPtr` or `params.multiCtasKvCounterPtr` is null —
  which is exactly the context path, since `FmhaDispatcher` never allocated
  those two buffers while `XqaDispatcher` did. Per the NVBug, the guard alone
  does not explain it: the upstream trtllm-gen autotuner change `659177c0` is
  what moved selection off the 2-CTA kernel — that change likely overlooked
  that DeepSeek-V3.2 can run generation kernels for prefill, so the autotuner
  may need adjusting — with the null-scratch guard turning what it selected
  into the slow path.
- **How introduced:** PR #12470 (`cf9963f19550`,
  `[None][feat] Support sparse mqa/gqa attention`, merged 2026-04-19) changed
  FMHA dispatch to the broader `useTllmGenSparseAttention()` predicate instead
  of `useSparseMLA()`, so GB200 DeepSeek-V3.2 context attention began using
  generation-style sparse kernels — first crashing. #13206 (`27ba2035fb92`,
  missing aarch64 lib) and #13104 (`0d2bea7c3c99`, KV-cache-manager-V2 /
  scheduler-V2 fixes) cleared the crash but left throughput regressed.
  PR #13379 (`9ddef037016c`,
  `[https://nvbugs/6098442][fix] WAR IMA on DS V3.2 and update trtllm-gen
  cubin, lib and src`, merged 2026-04-24) then shipped both the refreshed
  precompiled kernel library and the null-scratch `HOTFIX` guard quoted above,
  with its own `TODO: add scratch allocation and re-enable GmemReduction`.
- **Fix mechanism:** #13652 integrates the refreshed trtllm-gen FMHA kernel
  library — of its **3058** changed files (`+9952/-9152`) exactly **five** are
  source files and the rest are precompiled `*.cubin.tar.zst` blobs (the
  GitHub files API caps its listing at 3000 entries: 2663 modified, 168 added
  and 164 removed cubins are visible there). Among them is a 1:1 rename of the
  DS-MLA `HQk576HV512` sparse family from `PagedKvStaticTokenSparse…` to
  `PagedKvDenseStaticTokenSparse…` — 24 removed names against 24 added ones
  differing only by the inserted `Dense`, covering
  `HVPerCta128`/`HVPerCta256`/no-`HVPerCta`, 2-CTA and non-2-CTA, Bfloat16 and
  E4m3. The five source files switch MLA
  generation to `TrtllmGenAttentionMaskType::Dense` ("MLA generation kernels
  use dense mask. For multi-token generation, TRTLLM-Gen applies causality by
  shrinking each token's effective KV length") and add `DynamicTokenSparse`
  selection in `attentionOp.cpp` / `fmhaDispatcher.cpp`. Two earlier attempts
  on the same defect did not close it, and both are worth knowing before
  re-proposing them: PR #13410
  (`[None][fix] Add support for context multiCtaKv sparse fmha`, merged
  2026-05-06) added `AttentionOp::getFmhaMultiCtasKvScratchSize()`, grew the
  context workspace from 26 to 27 buffers and passed
  `multiCtasKvScratchPtr` / `multiCtasKvCounterPtr` into the sparse context
  path — i.e. exactly the allocate-the-buffers fix that makes the guard a
  no-op — yet it measured 14660.22 tok/s per the NVBug, still ~10 % below the
  16320 good baseline, so it does not fix the regression; and a patch that
  hard-overrode `options.mHeadDimPerCtaV = 256; options.mClusterDimX = 2` for
  DS MLA sparse attention in `fmhaKernels.h` measured 17373 tok/s (+3.5 % over
  baseline) but was never filed as a PR, judged viable but somewhat broad and
  liable to disturb other cases.
- **Detection signal:** diff FMHA kernel *names* between a good and a bad
  build — the trait substring is the whole signal:
  `nsys stats --report cuda_gpu_kern_sum <rep>.nsys-rep | grep -E 'HQk576HV512.*(HVPerCta256|2Cta)'`;
  a good run carries `HVPerCta256…Persistent2Cta`, the regressed run has
  neither. For the guard itself, the disable is logged (at debug level only):
  `grep -n "forcing MultiCtasKvMode to Disabled" <trtllm log>`, and the guard
  is `grep -n "HOTFIX" cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h`.
- **Prevention/guard:** the fallback announces itself only via
  `TLLM_LOG_DEBUG("MultiCtasKvScratchPtr/MultiCtasKvCounterPtr is null,
  forcing MultiCtasKvMode to Disabled")` — invisible at default verbosity —
  and shipped with a `TODO: add scratch allocation and re-enable
  GmemReduction`, i.e. a knowingly perf-degrading workaround with no warning
  and nothing tracking the cost; #13410 later did that allocation, three weeks
  after the drop landed. Gaps: a capability guard that swaps to a slower
  kernel should WARN at default level (the sibling
  [trtllm-gen JIT warmup case](../jit-and-warmup/trtllmgen-fmha-jit-warmup.md) added
  exactly such a warning in the same header), and no test asserts that a
  precompiled-kernel-library bump preserves the selected variant for a
  bar-tracked case.
- **Generalizes to:** `pattern-fast-path-silent-fallback` (a null-buffer
  capability guard added to stop a crash quietly demotes the fast kernel) plus
  `pattern-variant-misroute` (a dispatch predicate broadened for a new feature
  routes one model's *phase* onto a foreign kernel family); carries to any
  `if (ptr == nullptr) disable <fast mode>` workaround on a dispatch path where
  only one of several callers allocates the buffer; context-vs-generation
  kernel sharing where the two dispatchers set up different workspaces;
  precompiled-cubin or JIT selection where a library refresh changes the trait
  key a heuristic matches on (the mask type here); and upstream-owned
  autotuner heuristics that reach TensorRT-LLM as an opaque library bump, so
  the selection change is invisible in the TensorRT-LLM diff.
