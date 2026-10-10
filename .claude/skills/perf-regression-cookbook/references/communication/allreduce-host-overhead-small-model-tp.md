---
id: case-allreduce-host-overhead-small-model-tp
type: regression-case
family: communication
module: communication
maturity: full
regression_class: [host-work-added, measurement-artifact]
signals: [throughput-drop, perf-ci-bar-failure, host-time-increase, gpu-idle-between-steps]
subsystems: [communication, autotuner, runtime-cpp]
introduced_via: [config-default-change, pre-existing-gap]
phase: [any-phase]
patterns: [pattern-host-work-on-hot-path, pattern-measurement-not-product]
nvbugs: ["6262973"]
commits: ["c0e6d795c8d4"]
success_prs: [16902]
failed_prs: [15157]
---

# AllReduce host overhead on small-model TP, behind a 9.65% Llama headline that was 2.42% on one node

> Part of the [Communication regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6262973` · commit `c0e6d795c8d4` · PR #16902 —
  "[https://nvbugs/6262973][perf] Move AllReduce autotuner dispatch to C++"
  (merged 2026-08-11). Culprit for the visible step: PR #14472 ·
  commit `3d56a4e3996c` — "[TRTLLM-10004][chore] Enable NCCL symmetric
  zero-copy by default" (merged 2026-05-29). Only two PRs ever cite this
  bug (#15157 and #16902).
- **Failed attempts:** PR **#15157** (bot-filed by `tensorrt-cicd`,
  2026-06-09) — still **OPEN** but **superseded by the merged #16902**
  (untouched since 2026-06-22, no human review at all, only a CodeRabbit
  comment; the code it wanted to revert has since been restructured into
  C++). It attacked the same host overhead by *reverting* rather than
  relocating: (1) flip `_NCCL_SYMMETRIC_ZERO_COPY` back from `"1"` to `"0"`
  in `tensorrt_llm/_torch/distributed/ops.py`; (2) drop
  `inputs_pre_hook_register_nccl_symmetric_memory_window` from
  `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` so the AutoTuner
  benchmarks each tactic on the real production input instead of a
  pre-copied window buffer; (3) change the cache-miss fallback from
  `NCCL_SYMMETRIC.value` to `NCCL.value`. Its own diagnosis is worth
  keeping — the post-fix profile still showed "322 ncclSymk calls @ 473us
  = 152ms" against "Lamport one-shot (4347 calls @ 7.3us = 31.6ms)", i.e.
  the `inputs_pre_hook` "hides NCCL_SYMMETRIC's in-kernel
  cudaMemcpyAsync cost" so the tuner picks it for shapes where
  ONESHOT/TWOSHOT would win — and it reported `total_token_throughput`
  `1.785e+04` bad → `1.863e+04` after fix against a `1.926e+04` good, i.e.
  it recovered only part of the gap. Two claims (11/−25 and 6/−3 lines) are
  the whole diff; nothing was ever measured by a human. Lesson: reverting
  the zero-copy default was tried and not taken — the accepted fix moved
  the tactic dispatch to C++ and left the default alone.
- **Symptom:** rc16 → rc17 QA perf sweep reported 5–11% on three Llama
  families. Worst row, from the NVBug's own table:
  `llama_v3.3_70b_instruct_fp4-bench-pytorch-float4-maxbs:512-maxnt:2048-input_output_len:512,32-gpus:4`
  on GB300 — Inference Time `13664.328 → 15124.540` (+10.69%), Total Token
  Throughput `20383.585 → 18415.634` (−9.65%); the same test on GB200-OCI
  reported +5.51% / −5.22%, and `llama_v3.1_8b` BF16 and
  `llama_v3.3_nemotron_super_49b` were listed too. Window: `4517988c`
  (good, 1.3.0rc16) → `a422420d` (bad, 1.3.0rc17). **The headline was
  mostly measurement.** Two independent re-measurements shrank it: a GB200
  reproduction gave 70B NVFP4 TP4 512/32 `17,961.57 → 17,610.34` = −1.955%
  (reproduced, smaller magnitude) and 8B BF16 TP1 512/32
  `67,600.14 → 68,070.98` = +0.697% (not reproduced); a re-bisect with
  **all** commits pinned to one GB300 node (3 reps/commit) collapsed the
  good→bad gap to **2.42%** with a non-monotonic per-commit curve (no clean
  step). Both figures are from the NVBug.
- **Root cause:** host-side, not kernel-side. Two contributions, established
  in the NVBug:
  (1) making the GEMM output window-resident for zero-copy NCCL_SYMMETRIC
  moves the lifetime management of registered NCCL buffers back and forth
  across the C++/Python boundary, and that CPU cost shows up as rank skew
  **in eager mode only** — confirmed from eager-mode rank skew, with graph
  mode unaffected; and (2) a pre-existing tax the investigation surfaced as
  a side finding — AutoTuner CPU overhead of about 5% or more — because
  under `AllReduceStrategy.AUTO` the per-call bucket lookup
  and tactic dispatch ran in Python (`torch.ops.trtllm.tunable_allreduce`)
  on every AllReduce. Both only matter where the model is small enough that
  the host is the critical path; PR #16902 states it "benefits small,
  CPU-overhead-sensitive models".
- **How introduced:** the visible rc16→rc17 step came from a
  **default flip**: PR #14472 changed `TLLM_NCCL_SYMMETRIC_ZERO_COPY` from
  off-unless-set to on-by-default, justified by internal E2E sweeps that
  improved dense FP8 Llama-3.3-70B / 405B and showed no regression on
  Qwen2.5-72B, Mixtral-8x22B and DeepSeek-R1 — i.e. validated on models
  where the win is real and the host is not the bottleneck. The same-node
  A/B isolates it: idx106 `c7683f2f` with ZC=0 → `18904.6` vs idx107
  `3d56a4e3` with ZC=1 → `18616.9`, **the same wheel with the flag off vs
  on** (≈1.5%). The Python AUTO dispatch cost (2) was pre-existing and not
  introduced by this window.
- **Fix mechanism:** PR #16902 moves the hot-path AllReduce autotuner bucket
  lookup and tactic dispatch out of Python into C++. It adds
  `autotuned_allreduce` plus a thread-safe native tactic cache in
  `cpp/tensorrt_llm/thop/allreduceOp.cpp` (14 fixed token-bucket cutoffs;
  key = tensor group, fusion-op id, NCCL-window vs non-window mode, input
  scalar dtype, static input shape excluding `size(0)`), with
  `register_allreduce_tactic`, `validate_allreduce_tuning_buckets` and
  `clear_allreduce_tactic_cache` as native control ops. Python still tunes
  and registers tactics; the AUTO branch in
  `tensorrt_llm/_torch/distributed/ops.py` now picks `tunable_allreduce`
  (Python) while `_ALLREDUCE_AUTOTUNER_TUNING_MODE` is set or the native
  path is disabled, and `autotuned_allreduce` (C++) otherwise. Note this
  lands prong **1** of the two-prong plan recorded in the NVBug; prong 2
  (the Python-side lifetime handling and pool management of registered NCCL
  buffers), which targets the zero-copy buffer cost directly, had not landed
  as of this case.
- **Detection signal:** a broad multi-model throughput drop with a host
  signature rather than a kernel one — no kernel replacement in the range,
  the drop concentrated on *small* models / low token counts, present in
  eager mode and absent under CUDA graphs, and (per the NVBug) GPU power
  down 4.5% while throughput was down 7.3%. Three
  checks, cheapest first: A/B the culprit knob on **one node with one
  wheel**, `TLLM_NCCL_SYMMETRIC_ZERO_COPY=0` vs `=1`; bound the AUTO
  dispatch share with `TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1` (pre-existing
  knob, read in `ops.py`'s AUTO branch); and confirm the native path is
  actually live with
  `grep -n "autotuned_allreduce\|_ALLREDUCE_NATIVE_AUTOTUNER_ENABLED" tensorrt_llm/_torch/distributed/ops.py`
  plus a log grep for `Disabling native AllReduce autotuner` (emitted once
  by `disable_native_allreduce_autotuner()` when the Python/C++ bucket
  contract fails and it silently reverts to the Python autotuner —
  a fixed-but-quiet path is how this tax comes back).
- **Prevention/guard:** PR #16902 added a runtime contract guard —
  `validate_allreduce_tuning_buckets` requires the Python bucket list to
  match `kAllReduceTuningBuckets` exactly, and a mismatch logs once and
  falls back instead of dispatching wrong tactics — plus unit coverage in
  `tests/unittest/_torch/multi_gpu/test_allreduce.py` (clears the
  `AutoTuner` cache, then compares both the in-`autotune()`-context output
  and the native output against their references) and
  `test_user_buffers.py` (UB pass generation now expected on
  `autotuned_allreduce`). Gaps, both worth carrying: nothing perf-tests the
  CPU-bound small-model AllReduce path, so this class is only visible
  post-merge; and a default flip validated on host-insensitive models had
  no eager-mode / small-model gate. **Measurement guard:** for a
  sub-3% GB300-class bar, pin *every* probe in the window to one node and
  run ≥3 reps before bisecting — cross-node variance here manufactured
  ~7 of the 9.65 reported points, and single-run probes scattered across
  nodes are what stalled the first bisect (compounded by an unrelated
  crash-inducing commit, `50ca49f8c5` / PR #14378, nulling the middle probes — that is
  **round 1** of the IPC-HMAC-fd defect, now recorded as
  [IPC HMAC key by fd deadlocks the benchmark launcher](../measurement-and-test/ipc-hmac-key-via-fd-breaks-bench.md);
  its two broken windows null probes on *any* bisect that crosses them, so
  test for them before trusting a sparse probe series). Also note
  the residual real effect was small enough to be declined once — a fix PR
  was judged not worth filing for a 1.5% gap — and was ultimately fixed on
  its merits, not on the reported number.
- **Generalizes to:** `pattern-host-work-on-hot-path` (Python-side lookup
  and cross-boundary bookkeeping executed per collective call) and
  `pattern-measurement-not-product` (most of the reported delta was
  cross-node variance). Carries to: any per-op tactic/bucket lookup left in
  Python on a hot path while the op itself is C++; buffer
  registration/lifetime management that crosses the Python↔C++ boundary
  once per step; defaults flipped on the strength of sweeps run only in the
  regime that benefits (graph mode, large dense models) where another
  regime pays in host time; and any sub-3% headline on a shared multi-node
  perf cluster, where the first move is a same-node A/B of the suspected
  knob rather than a bisect.
