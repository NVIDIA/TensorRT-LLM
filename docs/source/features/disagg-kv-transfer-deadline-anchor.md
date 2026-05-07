# Anchoring `kv_transfer_timeout_ms` to actual transfer start

> Companion to
> [`disagg-kv-transfer-session-lifecycle.md`](disagg-kv-transfer-session-lifecycle.md).
> Documents the bookkeeping change introduced in PR #13830 (this PR),
> stacked on PRs #13796 and #13818.

## Background — the bug we're fixing

PR #13796 captured the per-request deadline at admit time:

```cpp
// cacheTransceiver.cpp::requestAndReceiveAsync (pre-fix)
auto future = mCacheReceiver->receiveAsync(llmRequest);
auto const deadline = computeTrackedFutureDeadline(
    /*requestStart=*/std::chrono::steady_clock::now());
mRequesterFutures.push_back(TrackedFuture{..., deadline, ...});
```

The actual transfer doesn't start until much later inside the worker
thread — it has to wait for the worker to dequeue the request, then
for `assignBufferIndexFor{Send,Recv}` to acquire a slot, then for
`session.{recv,send}` to exchange data with the peer.

Under burst load with a small staged-buffer pool (default
`recvBufferCount=1` and `TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM=1`),
the deadline burns down while the request is queued, not while it's
transferring. Empirically — see the d224/d227 testing under
`dynamo-gb200-test/docs/disagg-wedge-pr13818-findings-2026-05-06.md`
— a 60 s deadline on TCP at ~3 Gbps with a conc=128 burst caused
~80 % of requests to be quarantined for **queue starvation** rather
than for any actual NIXL/UCX wedge. They never even started
transferring; they just sat waiting for their turn.

## What this PR changes

`kv_transfer_timeout_ms` is now anchored to **actual transfer
start** — the moment the worker has acquired a slot and is about to
begin data movement.

### New `LlmRequest` field

```cpp
mutable TimePoint mKvCacheActualTransferStart{};

void setKvCacheActualTransferStart(TimePoint time) const;       // idempotent
TimePoint getKvCacheActualTransferStart() const noexcept;
bool hasKvCacheActualTransferStart() const noexcept;
```

This is **distinct from** the existing
`mPerfMetrics.timingMetrics.kvCacheTransferStart` (set at
admit / worker-dequeue time and exposed as
`req.kv_cache_transfer_time_ms`). The perf metric semantics are
preserved — downstream consumers / dashboards continue to see the
same timing model they did before.

### Formatters set the field after slot acquisition

| Path | File:line | Slot |
|---|---|---|
| Sender, primary KV | [`cacheFormatter.cpp:497`](https://github.com/NVIDIA/TensorRT-LLM/blob/fix/disagg-kv-transfer-deadline-fix/cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp#L497) | `assignBufferIndexForSend` |
| Receiver, primary KV | [`cacheFormatter.cpp:822`](https://github.com/NVIDIA/TensorRT-LLM/blob/fix/disagg-kv-transfer-deadline-fix/cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp#L822) | `assignBufferIndexForRecv` (or pre-assigned id) |
| MLA indexer-K, sender | [`mlaCacheFormatter.cpp:256`](https://github.com/NVIDIA/TensorRT-LLM/blob/fix/disagg-kv-transfer-deadline-fix/cpp/tensorrt_llm/batch_manager/mlaCacheFormatter.cpp#L256) | indexer-K send slot |
| MLA indexer-K, receiver | [`mlaCacheFormatter.cpp:496`](https://github.com/NVIDIA/TensorRT-LLM/blob/fix/disagg-kv-transfer-deadline-fix/cpp/tensorrt_llm/batch_manager/mlaCacheFormatter.cpp#L496) | indexer-K recv slot |
| RNN/Mamba, sender | [`rnnCacheFormatter.cpp:151`](https://github.com/NVIDIA/TensorRT-LLM/blob/fix/disagg-kv-transfer-deadline-fix/cpp/tensorrt_llm/batch_manager/rnnCacheFormatter.cpp#L151) | RNN-state send slot |
| RNN/Mamba, receiver | [`rnnCacheFormatter.cpp:324`](https://github.com/NVIDIA/TensorRT-LLM/blob/fix/disagg-kv-transfer-deadline-fix/cpp/tensorrt_llm/batch_manager/rnnCacheFormatter.cpp#L324) | RNN-state recv slot |

The setter is idempotent — for hybrid (KV + indexer-K, KV + RNN-state)
models that acquire multiple pool slots in sequence, the field
captures the **first** slot acquisition only. That's the point at
which the worker can be considered "in transfer."

### `maybeQuarantineLocked` consults the new field

```cpp
// cacheTransceiver.cpp::maybeQuarantineLocked (post-fix)
if (!entry.request->hasKvCacheActualTransferStart())
{
    return false;   // worker still queued / waiting for slot — no deadline
}
auto const transferStart = entry.request->getKvCacheActualTransferStart();
auto const deadline = transferStart + std::chrono::milliseconds(timeoutMs.value());
if (now < deadline) return false;
// quarantine
```

Requests that haven't yet reached the formatter's slot-acquire path
are not eligible for quarantine — they're not "stuck" in any
transport-relevant sense.

### Dropped: admit-time deadline plumbing

`TrackedFuture::deadline` is removed. `CacheTransceiver::computeTrackedFutureDeadline`
is removed. The three call sites in `respondAndSendAsync`,
`respondAndSendLayerWise`, and `requestAndReceiveAsync` no longer
capture admit-time deadlines.

## Behavioural impact

### Intended

| Scenario | Before | After |
|---|---|---|
| TCP, conc=128, 60 s deadline, 1-slot serialization | ~80 % of burst quarantined for queue wait | only transfers whose actual transfer time exceeds 60 s are quarantined (rare on functional TCP) |
| Healthy transfer in queue for 30 s, transfers in 5 s | quarantined at T+60 s mid-transfer | not quarantined; user sees normal completion |
| Genuinely stuck mid-transfer (peer crash after slot acquired, NIXL hung) | quarantined at T + 60 s after admit | quarantined at T + 60 s after slot acquisition |
| Request never gets a slot (queue depth ≫ capacity, pool wedged elsewhere) | quarantined at admit + 60 s | not quarantined for queue wait. Orchestrator's `req_timeout_secs=180` is the user-visible backstop. |

### Side effects

- **Perf metric `kv_cache_transfer_time_ms`** is unchanged. Still
  measured from admit to `kvCacheTransferEnd` via the existing
  `mPerfMetrics.timingMetrics.kvCacheTransferStart` — this PR adds
  a separate field, doesn't repurpose the existing one.
- **`_check_transceiver_health` (Python side)** is unchanged. It
  flips `is_shutdown` on `is_healthy=false` after the grace period.
  The trigger for `is_healthy=false` (no progress for >
  `mGlobalProgressDeadlineMs` OR quarantine count > budget) is
  unchanged at the C++ level. With this fix in place, false-positive
  quarantines from queue starvation no longer drive the budget side
  of that condition.
- **Wedge-trace heartbeats from PR #13818** are unchanged — they
  fire inside the worker thread regardless of where the deadline
  anchor is.
- **Structured cancel result enum / deferred resource release /
  `_inflight_cancel_requested_ids`** are unchanged.

### Risks and mitigations

1. **Unbounded queue wait.** A request in the queue without a slot
   never gets its deadline to start ticking. Mitigations:
   - The disagg orchestrator's HTTP-level `req_timeout_secs=180`
     fires a client-visible 504/408 at 180 s.
   - The transceiver's no-progress detector (`mLastProgressTime`
     vs `mGlobalProgressDeadlineMs`) flips `is_healthy=false` if
     the queue head stops draining for > 60 s with at least one
     tracked entry, triggering health-driven shutdown via the
     PR #13796 mechanism.
   - For deployments that want a hard ceiling on queue wait, a
     follow-up could add a separate `kv_queue_wait_timeout_ms`
     knob that fires at admit + N seconds if `hasKvCacheActualTransferStart()`
     is still false. Not needed for the d224/d227 scenario.

2. **Pre-existing behavior change for users who set
   `kv_transfer_timeout_ms` thinking "end-to-end including queue."**
   The new semantics are "actual transfer time only." For most
   deployments this is the more useful interpretation. Users who
   wanted the old admit-to-completion bound should consider using
   the orchestrator's `req_timeout_secs` as a wrapper budget.

3. **MLA / RNN / hybrid-pool ordering.** If a hybrid model acquires
   the indexer-K slot before the primary KV slot, the deadline
   anchors at the earlier acquisition. That's correct — it's the
   first moment the worker is doing real transfer work.

## Empirical validation plan

Re-run the d224 / d227 burst suite:

| Repro | Pre-fix | Post-fix expectation |
|---|---|---|
| d224 r1 (TCP 3 Gbps, conc=128, no canary) | ~80 % quarantined; `is_shutdown` at T+4.4 min | quarantines drop to near zero (only true mid-transfer stalls); `is_shutdown` does not fire; bursts drain over time |
| d227 r4 (Pattern C, canary on, peer crash) | both sides quarantine + canary-driven restart | mid-transfer wedges still get quarantined → still trigger restart path; verdict unchanged |
| Single-request smoke | unchanged | unchanged |

If d224 r1 still reaches `is_shutdown` after this fix, look at the
no-progress side of `updateHealthLocked` — that's the next layer
of the diagnosis (covered separately in
`disagg-deadline-vs-quarantine-proposal.md`).

## Test plan

The existing 19 Python unit tests in
`tests/unittest/disaggregated/test_kv_transfer_session_lifecycle.py`
continue to pass without modification — they exercise the Python
policy layer, which is unchanged by this PR.

C++-side coverage is via the existing multi-GPU integration tests in
`cpp/tests/unit_tests/multi_gpu/cacheTransceiverTest.cpp`. No
adjustments needed; that suite uses the public API and the change
is internal.

For empirical validation of the deadline-anchor behaviour itself,
the d224/d227 reruns above are the canonical signal.

## Sequencing relative to the broader recovery work

This PR is **independent of** the deadline-vs-quarantine proposal
in `claude-swe/disagg-deadline-vs-quarantine-proposal.md`. They're
complementary:

- This PR (deadline anchor): "deadline only fires for actual
  in-transfer time."
- The proposal (deadline-vs-quarantine): "deadline-firing should
  surface a user error but not drive the unhealthy budget."

The proposal can be applied on top of this PR if the team agrees
with the two-layer model. Either order works; this PR alone fixes
the immediate false-positive issue under the d224 scenario,
provided no_progress alone is the only path to `is_shutdown` (which
empirically it is on the prefill side under d224 — the second
UNHEALTHY flip is driven by the time-since-progress check, not by
quarantine count).
