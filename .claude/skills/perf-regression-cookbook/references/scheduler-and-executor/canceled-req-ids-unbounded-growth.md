---
id: case-canceled-req-ids-unbounded-growth
type: regression-case
family: execution-and-graph
module: scheduler-and-executor
maturity: full
regression_class: [host-work-added]
signals: [host-time-increase, throughput-drop]
subsystems: [scheduler-executor, runtime-python]
introduced_via: [pre-existing-gap]
phase: [any-phase]
patterns: [pattern-host-work-on-hot-path]
nvbugs: ["5508267"]
commits: ["f9380581c507"]
success_prs: [9280]
failed_prs: []
---

# Cancelled-request id list never pruned — per-step O(n) scan grows for the life of the server

> Part of the [Scheduler & executor regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5508267` · commit `f9380581c507` · PR #9280 —
  "[https://nvbugs/5508267][fix] Proper handling of inactive canceled requests".
- **Symptom:** a **gradual** slowdown of a long-running server under normal
  operation, root-caused per PR #9280 to failed requests: each failed request
  was added to the list of canceled request ids and never removed from it, so
  the list grew longer with every failed request. **No metric number, model
  config or hardware is stated in the PR** — do not cite a magnitude for this
  case.
- **Root cause:** `canceled_req_ids` is a plain Python **list**, and the only
  removal path lived inside the `for request in self.active_requests` loop and
  fired only when `_try_cancel_request` succeeded. An id belonging to a request
  that was no longer in `active_requests` — i.e. every *failed* request — could
  therefore never be removed and leaked permanently. The unbounded list was then
  re-scanned linearly twice per iteration: the `req_id not in
  get_canceled_req_ids()` membership test, run once per active request
  (O(n_active × n_canceled) per call), and the same `not in` test inside
  `update_waiting_queue()`. There *was* an unconditional
  `clear_canceled_req_ids()` — but only under `if self.enable_attention_dp:`, so
  for every non-attention-DP configuration the leak was structural.
- **How introduced:** unknown — the PR names no culprit commit. This is
  as-shipped behaviour of the cancellation bookkeeping, not a regression from a
  previously faster state; it is a *degrade-over-time* bug, which is why it
  reached a customer rather than a perf bar (see Prevention/guard).
- **Fix mechanism:** stop removing entries and instead **rebuild** the list every
  call — collect the ids whose `_try_cancel_request` returned False into
  `still_pending_canceled_ids`, then `clear()` + `extend(...)`
  unconditionally, deleting the `enable_attention_dp`-only special case (and its
  "TODO: revisit the cancel logic of attention dp"). Separately the membership
  test is materialized once per call as a `set(...)`, so the residual scan is
  O(1) per request rather than O(n_canceled).
- **Detection signal:** the leak is visible as *drift*, not as a level, so
  compare a fresh server against the same server hours later at the same request
  rate. `_handle_canceled_requests` is already `@nvtx_range`-annotated in
  `tensorrt_llm/_torch/pyexecutor/py_executor.py`, so in an nsys capture that
  NVTX range's duration growing monotonically over wall-clock at flat request
  rate *is* the leaked list being rescanned. Static probe on a suspect commit:
  `grep -n 'canceled_req_ids' tensorrt_llm/_torch/pyexecutor/py_executor.py
  tensorrt_llm/_torch/pyexecutor/executor_request_queue.py` — a `clear` reachable
  only under an `enable_attention_dp` branch is the defect.
- **Prevention/guard:** **no guard added** — the PR ships no test and no bound or
  warning on `len(canceled_req_ids)`. That is the gap worth naming: any
  per-iteration container keyed by request lifetime should either be a `set` with
  an unconditional prune or carry a size assert/warning, because an unbounded
  one produces a *slope*, and perf CI measures *levels* in a fresh process — this
  class is invisible to every bar the project runs.
- **Generalizes to:** `pattern-host-work-on-hot-path`; carries to any
  per-step linear scan over a list whose pruning is conditional on a config flag,
  request-id bookkeeping in disagg/cancellation paths, degrade-over-time
  customer reports with no bisectable commit (look for monotonic growth, not a
  step), and `x in list` on the hot path where a `set` was intended.
