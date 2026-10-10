---
id: case-beam-search-host-handoff
type: regression-case
family: execution-and-graph
module: sampler
maturity: full
regression_class: [host-work-added]
signals: [ttft-increase, host-time-increase, many-small-kernels]
subsystems: [sampler]
introduced_via: [pre-existing-gap]
phase: [any-phase]
patterns: [pattern-host-work-on-hot-path]
nvbugs: ["5615248"]
commits: ["755d38849563", "371c12662eca"]
success_prs: [13748, 13799]
failed_prs: []
---

# Beam-search sampler host cost: handoff kernel storm + per-step D2H copies

> Part of the [Sampler regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5615248` · commit `755d38849563` · PR #13748 —
  Reduce beam-search prefill->decode handoff cost;
  related: nvbug `5615248` · commit `371c12662eca` · PR #13799 —
  Beam history copies only on terminal steps. (Split from umbrella nvbug
  5615248: the two beam-search host-cost commits only.)
- **Failed attempts:** none as PRs (`failed_prs: []` — no PR was ever opened
  and closed against this bug). Per the NVBug, two kernel-side experiments were
  tried on personal branches and abandoned before any PR: a GEMM-tuning branch
  identified and adopted the faster GEMM kernel TRT picks for `o_proj`, but it
  made no visible difference in E2E latency because GPU work was not the
  bottleneck; and a megakernel branch was not yet performant and needed more
  tuning. Both are evidence for the diagnosis rather than against it: on this
  workload the bottleneck is host-side, so kernel-level tuning is the wrong
  lever and should not be re-proposed.
- **Symptom:** Elevated TTFT/E2E for beam-search serving (ITL impact
  negligible in the PRs' measurements). Measured on
  TinyLlama-1.1B-Chat-v1.0, L40S, ISL=100/OSL=20, beam_width=10,
  max_batch_size=1, piecewise CUDA graphs (PR #13799 workload); PR #13748
  reports TTFT -2.13% mean after fix, PR #13799 reports TTFT -0.092 ms and
  E2E -0.448 ms median. Surfaced via nvbug perf investigation with a repro
  bench on a personal branch.
- **Root cause:** Two host-cost sources in the TorchSampler beam-search path:
  (1) the prefill->decode handoff and per-step sampling launched many
  redundant small kernels — indexed assigns for seven beam-buffer resets in
  `_prepare_beam_search` (21 kernels/handoff), per-scatter indexed assigns in
  `FinishReasonsHandler.update_for_new_request`, and per-step re-allocation
  of `seq_offsets`/`beam_idx_arange` in `beam_search_sampling_batch`;
  (2) `_prepare_beam_history` issued D2H copies of beam history on *every*
  decode step, blocking the main stream, plus 2 redundant `seq_slots`/
  `seq_lens` H2D launches per step.
- **How introduced:** unknown — not stated in the PRs; neither fix names a
  regressing commit. The beam-search sampler path carried this host cost by
  construction (pre-existing gap), exposed once the surrounding path was
  otherwise optimized (PR #13748 builds "on top of" the piecewise-CUDA-graph
  coverage fix, PR #13574).
- **Fix mechanism:** PR #13748 replaces indexed assigns with
  `Tensor.index_fill_`/`Tensor.index_copy_` (21 -> 8 kernels per handoff,
  2 -> 1 per scatter), hoists the int64 `seq_slots` cast into
  `setup_sampler_step`, and caches `seq_offsets`/`beam_idx_arange` at
  `BeamSearchStore`/`BeamSearchMetadata` construction (4 fewer kernels/step).
  PR #13799 skips beam-history D2H copies on non-terminal steps via a
  host-side predictor, runs potentially-terminal copies on a private side
  stream recorded into `SamplerEvent` (predictor miss falls back to a
  synchronous `.cpu()`), gated by the opt-in prototype knob
  `enable_speculative_beam_history_d2h` (default False); `seq_slots`/
  `seq_lens` are cast once per step at the top of `_process_requests`.
- **Detection signal:** nsys shows per-decode-step DtoH memcpys and streams of
  tiny fill/copy kernels between sampler steps for beam-search runs; check
  the knob with `grep -n "enable_speculative_beam_history_d2h"
  tensorrt_llm/llmapi/llm_args.py` and the copy site with
  `grep -n "_prepare_beam_history" tensorrt_llm/_torch/pyexecutor/sampler.py`.
- **Prevention/guard:** PR #13799 added
  `tests/unittest/_torch/sampler/test_beam_search_speculative_d2h.py` plus a
  `validate_speculative_beam_history_d2h` config validator (rejects the knob
  with `sampler_force_async_worker=True`); these guard correctness of the
  fix, not the perf itself — no beam-search perf CI bar exists (gap).
- **Generalizes to:** pattern-host-work-on-hot-path — per-step host
  bookkeeping and eager tensor ops in the sampler serialize with GPU steps;
  carries to per-step D2H reads in any sampler/stop-criteria path, per-request
  Python loops issuing indexed assigns instead of batched `index_*_` ops,
  repeated per-step allocation of index tensors that could be cached at
  construction, and redundant per-step H2D casts of metadata already resident
  on device.
