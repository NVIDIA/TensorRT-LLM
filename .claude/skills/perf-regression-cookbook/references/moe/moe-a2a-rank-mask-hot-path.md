---
id: case-moe-a2a-rank-mask-hot-path
type: regression-case
family: communication
module: moe
maturity: full
regression_class: [kernel-swap-regressed, communication-regression]
signals: [itl-increase, perf-ci-bar-failure, throughput-drop]
subsystems: [communication, moe]
introduced_via: [new-feature]
phase: [any-phase]
patterns: [pattern-inactive-feature-guard-on-hot-path]
nvbugs: ["6430702"]
commits: ["9aae1b8a4488"]
success_prs: [16200]
failed_prs: [16025]
---

# WideEP fault-tolerance rank-mask checks left in the NVLink one-sided MoE AllToAll hot path

> Part of the [MoE regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6430702` · commit `9aae1b8a4488` · PR #16200 —
  "[https://nvbugs/6430702][perf] Restore NVLink one-sided A2A fast path"
  (base `main`, merged 2026-07-14). #15524 is the culprit; #16025 is a
  rejected earlier fix attempt (below).
- **Failed attempts:** PR **#16025** "[None][perf] template NVLink A2A fault
  tolerance checks" (opened 2026-07-07) — CLOSED unmerged
  2026-08-03 with "Superseded by
  https://github.com/NVIDIA/TensorRT-LLM/pull/16200". It touched the same
  seven files as #16200 and proposed the *same* core mechanism (an
  `ENABLE_FAULT_TOLERANCE` template parameter with the `is_rank_active`
  checks under `if constexpr`), but it also deleted the fault-tolerance
  route and payload checks outright. A reviewer requested changes
  2026-07-09: "Compile-time specialization is a valid improvement and
  addresses concern (b). However, the removal of FT route and payload checks
  assumes concern (a) is already an enforced runtime invariant. It is not:
  #15525 supplies only the EPLB reconfiguration primitive… #15895 covers
  only in-flight abort", plus "the mask remains captured by value and no
  CUDA-graph lifecycle is changed", asking instead to "keep the compile-time
  FT/non-FT specialization, but remove the FT-specialized route/payload
  checks _for now_" and that "FT must explicitly disable or reject CUDA
  graphs". #16200 is that narrower version by the reviewer — which is why it
  keeps mask handling for dispatch routing, peer counters, EPLB stats and
  completion sync, and adds `reject_rank_mask_cuda_graph_capture`. Note
  #16025 does not carry the bug id in its title, so a
  `gh pr list --search "6430702"` lookup misses it entirely. Lesson for a
  future agent: the templating idea is right, but do not bundle removal of
  the FT-mode checks with it while the FT MVP is still landing.
- **Symptom:** Mean Gen Worker Per-Iter Device Step Time (lower is better) on
  `gen_only-gb200_deepseek-r1-fp4_1k1k_con1024_ctx1_dep4_gen1_dep32_eplb0_mtp0_ccb-NIXL-con1024_iter1_isl1024_osl1024`
  (GB200, DeepSeek-R1-FP4, DEP32) went `12.88 ms` → `14.43 ms` between
  post-merge jobs 2825 and 2826 — good commit
  `7c8dde830bac813e23605d47a1d27c92d5437a92`, bad commit
  `045705139d125dbfd0614b369096f7bb5ddcbebf` (numbers from PR #16200's Root
  cause section). Surfaced as a post-merge perf-CI regression bar.
- **Root cause:** The WideEP fault-tolerance active-rank mask was implemented
  as *runtime data* rather than a compile-time mode. Two
  `is_rank_active(ptrs.active_rank_mask, target_rank)` checks sat inside
  `vectorized_combine_impl` in
  `cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu` — one
  in the payload-load pass and one in the conversion pass — so they executed
  in the token × top-k × vectorized-payload hot loop of every MoE combine,
  including the overwhelmingly common case where every rank is active and no
  fault tolerance is configured. The same unconditional `is_rank_active`
  guards were also spread through the dispatch route loop, peer counters,
  EPLB stat gather, and completion-flag store/wait loops. Because the
  feature's "off" state was expressed as an all-ones mask
  (`active_rank_mask[] = {~uint64_t{0}, ~uint64_t{0}}`, documented as
  "backwards-compatible no-masking behavior"), the branch was still emitted
  and still evaluated — off-by-data, not off-by-compilation.
- **How introduced:** PR #15524 / commit `a0c406ff88`
  ("[TRTLLM-12557][feat] WideEP FT: add AlltoAll watchdog (1a.3 + 1a.4)"), a
  fault-tolerance feature; #16200 lists it as related, and its diff is the one
  that adds the two `is_rank_active` checks to `moeAlltoAllKernels.cu`.
- **Fix mechanism:** Make the mask a compile-time specialization instead of a
  runtime value. `moeA2ADispatchKernel` and `moeA2ACombineKernel` gain a
  `bool ENABLE_RANK_MASK` template parameter, every `is_rank_active` check
  moves under `if constexpr (ENABLE_RANK_MASK)`, and the launch paths add a
  `SWITCH_BOOL(params.enable_rank_mask, ENABLE_RANK_MASK, ...)` around the
  existing dtype/top-k switches — so the non-FT instantiation contains no
  mask code at all. The two combine-payload checks are deleted outright
  (the loop is restored to the `dst_idx < 0` sentinel test only, matching good
  commit `7c8dde83`); mask handling is kept for dispatch routing, peer
  counters, EPLB stats and completion sync in FT mode only. `enable_rank_mask`
  is threaded explicitly through `moe_a2a_dispatch` / `moe_a2a_combine`
  (real + fake schemas) and is fixed at communicator construction from
  `ep_group_health is not None`, so a run cannot flip specializations
  mid-flight. Rank-mask mode additionally rejects CUDA graphs via
  `reject_rank_mask_cuda_graph_capture` in
  `tensorrt_llm/_torch/alltoall_watchdog.py`, because the mask is captured by
  value in kernel arguments.
- **Detection signal:** A per-iteration device-time step regression on a
  high-EP MoE case (here DEP32) with no kernel *replacement* in the range —
  the same A2A kernels, just slower, and slower for every config rather than
  one dtype/shape. Audit with
  `grep -n "is_rank_active" cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu`
  and confirm each hit is inside an `if constexpr (ENABLE_RANK_MASK)` block
  and not in a per-token / per-top-k / per-payload-element loop; then check
  the launch site actually specializes, i.e. `SWITCH_BOOL(params.enable_rank_mask, ...)`
  wraps the kernel selection. Cross-check the feature is really off by
  construction: `grep -n "_rank_mask_enabled" tensorrt_llm/_torch/distributed/moe_alltoall.py tensorrt_llm/_torch/modules/fused_moe/communication/nvlink_one_sided.py`.
- **Prevention/guard:** PR #16200 added mode-consistency asserts that make the
  off state unambiguous — `TORCH_CHECK(!hasActiveRankMask(activeRankMask),
  "active_rank_mask requires enable_rank_mask=True")` in
  `cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp`, the matching Python
  `ValueError("active_rank_mask requires committed EP group health")`, and
  reworked multi-GPU coverage in
  `tests/unittest/_torch/modules/moe/test_moe_comm.py` (+101/−162) plus
  `tests/unittest/_torch/modules/test_alltoall_watchdog.py` (+26). It also
  records a reviewable structural check in its Validation section: "Source
  comparison confirms the combine payload loop matches good commit
  `7c8dde83`". All of these are correctness/structure guards — the gap is
  that nothing perf-tests the default (non-FT) A2A instantiation, and PR
  #16200 explicitly could not run benchmarks locally ("Not run locally: CUDA
  build, MPI multi-GPU tests, or performance benchmarks"), so both the
  regression and its fix depended entirely on post-merge CI. Review rule: an
  optional feature must be off at *compile* time on the kernel hot path, not
  off by feeding it a neutral runtime value.
- **Generalizes to:** `pattern-inactive-feature-guard-on-hot-path` — a guard
  for an optional, usually-inactive feature is evaluated unconditionally
  inside a hot device loop, so every deployment pays for a feature nobody
  enabled. Carries to: fault-tolerance / watchdog / health-mask work landing
  in collective kernels (dispatch-route, completion-flag and peer-counter
  loops multiply a scalar check by ranks × tokens × top-k); "backwards
  compatible" defaults expressed as neutral data (all-ones masks, identity
  scales, null-object pointers) that keep the branch alive instead of
  removing it; debug/validation predicates added to kernels behind a runtime
  flag rather than a template parameter or `if constexpr`; and any new kernel
  parameter passed as a runtime `bool` where the existing launch path already
  has a `SWITCH_BOOL`-style specialization macro available.
