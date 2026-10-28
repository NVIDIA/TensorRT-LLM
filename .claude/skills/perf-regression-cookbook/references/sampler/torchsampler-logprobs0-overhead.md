---
id: case-torchsampler-logprobs0-overhead
type: regression-case
family: execution-and-graph
module: sampler
maturity: full
regression_class: [host-work-added]
signals: [host-time-increase, gpu-idle-between-steps, itl-increase, throughput-drop]
subsystems: [sampler]
introduced_via: [pre-existing-gap]
phase: [decode]
patterns: [pattern-host-work-on-hot-path]
nvbugs: ["5708901"]
commits: ["2bc2acda4f0f", "7443b7f02ba7"]
success_prs: [11983, 16958]
failed_prs: []
---

# TorchSampler pays full logprobs machinery for `logprobs=0` requests

> Part of the [Sampler regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5708901` · commit `2bc2acda4f0f` · PR #11983 —
  "[perf] reduce logprobs=0 overhead in TorchSampler". **Two merged fixes
  carry this bug id, five months apart.** The follow-on is PR #16958 · commit
  `7443b7f02ba7` · base `main`, merged 2026-07-31 —
  `[https://nvbugs/5708901][perf] avoid logits copies when computing logprobs`
  (`sampler/sampler.py` +442/-251, `sampler/ops/vanilla.py` +12/-28,
  `tests/.../test_logits_logprobs.py` +604/-0, `test_torch_sampler.py` +4/-4),
  which avoids snapshotting logits for later logprobs computation and simplifies
  the processed-logprobs indexing. Cite #11983 for the `logprobs=0`
  specialization below; cite #16958 for the copy-elimination rework of the same
  path.
- **Symptom:** With `logprobs=0` at batch size 1000 (Llama-3.2-1B-Instruct,
  L40S), TorchSampler spent 2.4 ms GPU time and 13.8 ms host time per step on
  sampling — far above `TRTLLMSampler` (5.9 ms host). The fix cut this to
  1.0 ms GPU / 11.3 ms host (numbers from the PR description).
- **Root cause:** The logprobs path did not specialize for the trivial
  `logprobs=0` case. On the host, `_store_logprobs_list_to_request` (called
  from `handle_logprobs`) ran the full per-step top-k dict-merge loop even
  when `num_topk_logprobs == 0` (only the sampled token's logprob is
  needed). On the GPU, the path ran unfused eager ops:
  an index-select materialized before `F.log_softmax` over
  `(batch, vocab)`, an in-place `greater_` + `sum` pair for sampled-rank
  with extra memory passes, and a gather-then-scatter copy of raw logits.
- **How introduced:** unknown — not stated in the PR. The PR benchmarks
  against `TRTLLMSampler` as the reference, framing this as a pre-existing
  efficiency gap in the Torch sampling path rather than a break from a
  previously faster TorchSampler state.
- **Fix mechanism:** Adds a dedicated `num_topk_logprobs == 0` branch in
  `TorchSampler._store_logprobs_list_to_request` that builds only the
  sampled-token `Logprob` dict, and introduces a `_Fusions` class in
  `tensorrt_llm/_torch/pyexecutor/sampling_utils.py` with `torch.compile`d
  helpers — `gather_log_softmax` (fuses index-select into log_softmax,
  `online_softmax=True`), `determine_sampled_rank`, and `gather_scatter` —
  so intermediates are not materialized.
- **Detection signal:** nsys shows a long sampling span between decode
  forward steps (host-side dict building plus a stream of small eager
  sampling kernels around `log_softmax`); overhead scales with batch size
  even when requests set `logprobs=0`. Compare the sampling span against a
  `TRTLLMSampler` run; confirm the fused path exists with
  `grep -n "_Fusions" tensorrt_llm/_torch/pyexecutor/sampling_utils.py`.
- **Prevention/guard:** The PR adapts the existing
  `tests/unittest/_torch/sampler/test_logits_logprobs.py` tests (rank
  tie-break ordering, `assert_close` tolerances) so pre-existing
  `logprobs=0` cases pass with the fused path; the PR itself lists Test
  Coverage as n/a. The follow-on #16958 does add substantial coverage to the
  same file (+604 lines in `test_logits_logprobs.py`), but again for
  correctness of the reworked path. That guards correctness only — no perf bar
  pins TorchSampler sampling overhead vs batch size, so a reintroduced slow path
  would ship silently. Review checklist: any per-step sampler code must
  special-case the feature-off configuration.
- **Generalizes to:** `pattern-host-work-on-hot-path` — a feature's cost is
  paid even when the feature is disabled or trivial (`logprobs=0`). Carries
  to: per-step Python loops over the batch in any sampler/postprocess hook;
  eager gather + softmax chains that materialize `(batch, vocab)`
  intermediates; stop-criteria or penalty code that runs for requests not
  using them; logits post-processors invoked unconditionally per iteration.
