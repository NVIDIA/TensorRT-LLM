---
id: case-overlap-scheduler-first-token-delay
type: regression-case
family: execution-and-graph
module: scheduler-and-executor
maturity: full
regression_class: [scheduler-batching-regression]
signals: [ttft-increase, host-time-increase]
subsystems: [scheduler-executor]
introduced_via: [pre-existing-gap]
phase: [prefill]
patterns: [pattern-delayed-first-token-emission]
nvbugs: ["5615248"]
commits: ["cf87a8beaa8e"]
success_prs: [14061]
failed_prs: []
---

# Overlap scheduler delays first-token emission behind the next sample step

> Part of the [Scheduler & executor regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5615248` · commit `cf87a8beaa8e` · PR #14061 —
  "[perf] Early emission of first token with overlap scheduling". The same bug
  id also appears on the beam-search handoff and piecewise-cudagraph PRs
  (#13748, #13799, #13574); this case covers only the overlap-scheduling
  first-token emission commit.
- **Failed attempts:** none as PRs (`failed_prs: []` — `gh pr list --search
  5615248 --state all` returns only merged fixes).
- **Symptom:** Elevated TTFT under the overlap scheduler: the first-token
  response is enqueued only after `sample_async` of the *next* step, so the
  first token waits behind host-side sampling dispatch. Per the PR, this is
  "particularly useful for small language models with tight TTFT
  expectations" — i.e. the delay is most visible where each step is short.
- **Root cause:** With overlap scheduling, `PyExecutor` processed the
  previous batch's responses (including the iteration-1 first-token response
  in `_handle_responses`) only after issuing `sample_async` for the current
  step, delaying first-token emission behind the next step's host-side
  sampling dispatch (magnitude not quantified in the PR; most visible when
  steps are short).
- **How introduced:** unknown — not stated in the PR; inherent to the
  overlap-scheduler pipeline structure (a design trade of per-step latency
  for throughput), not attributed to a specific regressing commit.
- **Fix mechanism:** Opt-in prototype knob
  `enable_early_first_token_response` (`TorchLlmArgs`, default `False`). When
  set, a new `_emit_first_token_responses` pass in `py_executor.py` enqueues
  non-terminal iteration-1 responses right after `_update_requests` /
  `_send_kv_async` and before the next `sample_async`; `_handle_responses`
  gains an `emit_first_iter` flag to suppress the duplicate iter-1 emission.
  A validator warns and disables the knob when the overlap scheduler is off
  or a disaggregated `cache_transceiver_config` is configured.
- **Detection signal:** TTFT noticeably worse with the overlap scheduler
  enabled than with `disable_overlap_scheduler: true` at equal batch, while
  ITL is fine; in an nsys timeline the first-token response enqueue lands
  after the next step's sampling dispatch. Check the knob with
  `grep -rn "enable_early_first_token_response" tensorrt_llm/llmapi/llm_args.py`
  and look for the NVTX range `_emit_first_token_responses` in a trace.
- **Prevention/guard:** PR adds streaming-parity tests
  (`tests/unittest/_torch/sampler/test_logits_logprobs.py`, plus accuracy
  and l0_h100 list entries) covering the early-emission path. Gap: no perf
  bar compares TTFT overlap-vs-non-overlap, so the latency cost of overlap
  pipelining on first tokens is not tracked automatically.
- **Generalizes to:** `pattern-delayed-first-token-emission` — pipelining or
  overlap machinery postpones when the first-token response leaves the
  executor. Carries to: response batching/stream-interval logic holding the
  iteration-1 response; disaggregated serving where ctx→gen handoff delays
  first-token return; speculative-decoding loops that emit responses only
  after draft+verify of the following step; any async sampler where response
  enqueue is ordered after next-step dispatch instead of before it.
