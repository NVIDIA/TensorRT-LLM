# Disaggregated KV transfer: session lifecycle and cancel contract

> **Status:** beta — applies to disaggregated serving with the C++ cache
> transceiver (NIXL / UCX backends).

This note describes the contract between the Python executor and the C++
cache transceiver for cancellation, timeout, quarantine, and process
health. It supersedes the per-request fail-close behaviour explored in
PR&nbsp;13706 and the C++-only lifetime hardening in PRs&nbsp;13713 / 13728:
both versions still hung in the field because neither isolated the
"backend is wedged but the executor thread is blocked" failure mode.

The fix direction is documented in
`disagg-kv-transfer-hang-restart-analysis.md`. This file describes the
mechanism that this codebase actually implements.

## Three responsibilities, separated

| Responsibility    | Owner                | Mechanism                                 |
| ----------------- | -------------------- | ----------------------------------------- |
| Request failure   | Python executor      | Surface a request-level error to the user when C++ reports a final transfer state. |
| Resource quiescence | C++ transceiver    | Worker future stays pinned until ready / error / quarantine. |
| Process health    | C++ transceiver      | Quarantine budget + global progress deadline → `isHealthy() == false`. |

## Status polling is non-blocking by default

`BaseCacheTransceiver::checkContextTransferStatus` and
`checkGenTransferStatus` poll worker futures with `wait_for(0ms)` and
**never** call `future.get()` on an unready entry. When
`atLeastRequestNum > 0` is requested, the polling code admits additional
ready futures but skips unready ones rather than blocking the executor
thread.

For shutdown / drain there are explicit blocking variants:
`drainContextTransferStatus` and `drainGenTransferStatus`. These are the
only paths permitted to block on a worker future.

## Structured cancellation

`cancelRequest(LlmRequest*)` returns a backward-compatible bool;
`cancelRequestStructured(LlmRequest*)` returns a
`TransferCancelResult` enum:

| Value                       | Caller may release request resources? |
| --------------------------- | -------------------------------------- |
| `NotFound`                  | yes — C++ already reached a final state |
| `AlreadyComplete`           | yes — worker future is ready |
| `CancelledBeforeAdvertise`  | yes — no buffer was advertised to the peer |
| `CancelRequestedInFlight`   | **no** — worker may still touch buffers |
| `BackendUnhealthy`          | **no** — orchestration must restart    |
| `NotCancellable`            | **no** — retry later                   |

Only the first three states allow Python to free the request's KV /
transfer-buffer resources. The remaining states keep the request owned
by C++; the Python executor adds the request id to
`_inflight_cancel_requested_ids`, which `_can_terminate_request_now`
checks before any `_do_terminate_request` call.

## Bounded quarantine, explicit health signal

When a tracked transfer exceeds its per-request `kvTransferTimeoutMs`
deadline and the worker future is still not ready, the entry is flipped
to *quarantined*: an error is surfaced to the caller, but the future
stays in the tracking vector. A counter (`mQuarantinedTransferCount`)
increments. If the counter exceeds `mQuarantineBudget`, **or** no
worker has reached a final state for longer than
`mGlobalProgressDeadlineMs`, the transceiver flips
`isHealthy()` to false. Orchestration is expected to restart the
worker on this signal — neither C++ nor Python frees pinned memory
locally.

`getHealth()` returns a `TransceiverHealth` snapshot exposing
`is_healthy`, `quarantined_transfer_count`, `quarantine_budget`,
`seconds_since_last_progress`, and `global_progress_deadline_seconds`
for metrics / observability.

## Python policy

The `PyExecutor` keeps these responsibilities only:

* mark transfer start time (`py_kv_transfer_start_time`);
* mark a request timed out after `kv_transfer_timeout_ms`
  (`py_kv_transfer_timed_out`);
* call `cancel_request_structured()` once for a timed-out transfer;
* defer `_terminate_request()` while the request is still in a
  disaggregated transfer state — see `_can_terminate_request_now`;
* report a request-level error only when the structured cancel result
  is one of the safe-to-release outcomes.

The PR&nbsp;13706 anti-pattern of clearing `active_requests`,
`waiting_queue`, `request_accumulated`, and setting `is_shutdown=True`
on a single in-flight transfer timeout is **not** present and must not
be added back. Per-request timeouts are recoverable; only an unhealthy
transceiver should drive a process restart, and that restart is
orchestration's responsibility.

## ADP response flush

When attention data parallelism (`enable_attention_dp=True`) is
enabled, `_enqueue_responses` runs a `tp_gather` collective. Calling it
inside `_end_transfer_and_maybe_terminate` deadlocks because only the
DP rank that owns the request reaches that point. The executor instead
buffers transfer-completion responses in `_pending_transfer_responses`
and flushes the buffer at synchronised loop points where every DP rank
participates in the collective:

* before `_handle_canceled_requests` in the PP path;
* before `_handle_canceled_requests` in the standard executor loop;
* unconditionally in the overlap loop (so ranks with differing
  `should_process_previous_batch` still participate).

`_flush_pending_transfer_responses` always calls `_enqueue_responses`
when ADP is enabled, even if the local buffer is empty, so the
collective is symmetric across ranks.

## Test coverage

* `tests/unittest/disaggregated/test_kv_transfer_session_lifecycle.py`
  — unit tests for the deferred-terminate guard, the structured-cancel
  decision tree, and the ADP flush symmetry. These tests intentionally
  do not require GPU or MPI; they replay the policy decisions as plain
  Python so failures fail-fast in pre-merge CI.
