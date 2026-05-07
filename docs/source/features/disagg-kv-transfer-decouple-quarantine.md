# Decoupling deadline-error from transceiver-health quarantine

> Companion to
> [`disagg-kv-transfer-session-lifecycle.md`](disagg-kv-transfer-session-lifecycle.md)
> and to the design proposal in
> `claude-swe/disagg-deadline-vs-quarantine-proposal.md`. Documents
> the bookkeeping change introduced in PR #13831 (this PR), stacked
> on PR #13818. **Sibling** (not stacked) of PR #13824.

## Background

PR #13796 conflated three different failure modes under a single
"quarantined" label, all of which fed `mQuarantinedTransferCount`
and therefore drove `isHealthy()`:

| Underlying state | Example | Actually completes? | Should it count toward unhealthy? |
|---|---|---|---|
| Slow-but-functional transport | TCP at 3 Gbps, 1-slot serialized, conc=128 burst | yes (eventually) | **no** |
| NIXL/UCX endpoint genuinely dead | peer pod crashed mid-transfer | no | yes |
| Sender's done-notif lost | NIXL Q4.1 territory | maybe | yes after confirmation |

The d224/d227 burst tests showed that under sustained TCP-bound
traffic, deadline-quarantine count grew until it crossed
`mQuarantineBudget` (default 16) and flipped the transceiver
unhealthy — even though completions kept arriving (the transport
was functional, just slow). The decode-side asymmetry (only one
UNHEALTHY flip on decode, never reaching `is_shutdown` because
`mLastProgressTime` got reset by unrelated completing transfers)
exposed the same problem in the opposite direction: a single
global "last progress" timer can't represent per-entry stuck-ness.

## What this PR changes

Three coordinated edits:

### (A) `mQuarantinedTransferCount` is now observability-only

`maybeQuarantineLocked` still increments the counter (for dashboards,
metrics, `getHealth()` consumers), but `updateHealthLocked` no
longer reads it. Deadline-only quarantines are no longer wedge
signals.

```cpp
// updateHealthLocked, before
bool overBudget = mQuarantinedTransferCount > mQuarantineBudget;
bool nextHealthy = !overBudget && !wedged;

// updateHealthLocked, after
// (no overBudget check — quarantine count is observability only)
bool nextHealthy = !wedged;
```

### (C) Per-entry `quarantinedAt` timestamps

`TrackedFuture` gains a `quarantinedAt` field. `maybeQuarantineLocked`
sets it to `now` when transitioning the entry to `quarantined=true`.

`updateHealthLocked` flips unhealthy when ANY tracked entry has been
quarantined for longer than `mGlobalProgressDeadlineMs`:

```cpp
auto const isStuckPastDeadline = [&](std::vector<TrackedFuture> const& vec) {
    for (auto const& e : vec) {
        if (e.quarantined
            && e.quarantinedAt != std::chrono::steady_clock::time_point{}
            && (now - e.quarantinedAt) > mGlobalProgressDeadlineMs) {
            return true;
        }
    }
    return false;
};
bool const perEntryStuck = isStuckPastDeadline(mSenderFutures)
                        || isStuckPastDeadline(mRequesterFutures);
```

This per-entry check fixes the asymmetry seen on d224's decode side
— individual stuck transfers surface as wedged regardless of
unrelated completions resetting `mLastProgressTime`.

### Health is now a two-condition OR

```
wedged = noProgressTimeout                            (B2 — global no-progress)
       || perEntryStuck                               (C — per-entry stuck-past-deadline)
```

`noProgressTimeout` (the existing `mLastProgressTime` aging check)
is preserved as a fast-path detector for "all transfers stopped
completing." `perEntryStuck` adds the asymmetry-resistant per-entry
detector.

The new UNHEALTHY log line carries both signals:

```
CacheTransceiver flipping to UNHEALTHY: noProgressTimeout=1 perEntryStuck=0
quarantined=11 sinceProgressMs=60005 deadlineMs=60000
```

vs the pre-fix:

```
CacheTransceiver flipping to UNHEALTHY: quarantined=11 budget=16
sinceProgressMs=60005 deadlineMs=60000
```

The `noProgressTimeout=` and `perEntryStuck=` flags tell you which
of the two health conditions tripped, useful for diagnosis.

## What does NOT change

- `mQuarantinedTransferCount` is still incremented — surfaced via
  `TransceiverHealth.quarantined_transfer_count` for observability.
- `mQuarantineBudget` is preserved as a field for backward compat;
  no longer consulted. Surfaced via
  `TransceiverHealth.quarantine_budget`.
- `_check_transceiver_health` (Python) is unchanged. It still
  watches `is_healthy()` and sets `is_shutdown` after the grace
  period. The trigger conditions for `is_healthy=false` are what
  this PR adjusts at the C++ layer.
- PR #13818's `[wedge-trace]` heartbeats are unchanged.
- PR #13796's structured cancel result enum, deferred resource
  release path, and Python `_inflight_cancel_requested_ids` are
  unchanged.
- The user-visible deadline-error response still fires on
  `kvTransferTimeoutMs`. Quarantine still puts the request in
  `errorRequestIds` and surfaces `Request X timed out` to the
  client. **The user contract is preserved.** Only the internal
  health bookkeeping changes.

## Behavioural impact

| Scenario | Pre-fix | Post-fix expected |
|---|---|---|
| Slow TCP burst, conc=128, transfers complete in 30–80 s | quarantine count grows past budget (16) → `is_healthy=false` → `is_shutdown` after grace → permanent wedge for canary-OFF deployments | quarantine count still grows (observability), but health stays True as long as completions arrive within `mGlobalProgressDeadlineMs` of each other AND each individual entry clears within the same window. d224 r1 should resolve without restart. |
| Genuine wedge: peer crash mid-transfer, no completions | `mLastProgressTime` ages out → `noProgressTimeout=true` → unhealthy | unchanged: `noProgressTimeout` still fires |
| d224 decode-side asymmetry: 6 wedged receives + intermittent unrelated completions | `mLastProgressTime` reset by intermittent completions → never reaches `is_shutdown` | each of the 6 wedged entries gets `quarantinedAt` set when their deadline fires; 60 s later, `perEntryStuck=true` → unhealthy → `is_shutdown` after grace |
| Single-request smoke | unchanged | unchanged |

## Sequencing relative to PR #13824

PR #13824 (deadline anchored to actual-transfer-start) and this PR
(decouple deadline from quarantine) are **independent** and
**complementary**:

- **#13824 alone**: deadline-quarantines fire less often (only for
  transfers that actually took too long after slot acquisition).
  Health bookkeeping unchanged. d224 r1 might still flip unhealthy
  if slot wait pushes some transfers past the deadline at the
  formatter level (rare).
- **This PR alone**: deadline-quarantines may still fire too eagerly
  (admit-time anchor under burst), but they don't drive health.
  Slow-but-completing TCP no longer flips unhealthy. d224 r1 should
  resolve.
- **Both applied**: deadline-quarantines fire only on real mid-flight
  stalls AND don't drive health on edge cases. Cleanest end state.

The PRs can be merged in either order. They don't conflict (this PR
touches `cacheTransceiver.{h,cpp}` only; #13824 touches that plus
the formatters and `llmRequest.h`). After both are merged a small
trivial conflict in `cacheTransceiver.cpp` (the same lines in
`maybeQuarantineLocked` and `updateHealthLocked`) will need
resolution; both edits are compatible.

## Test plan

The 19 existing Python unit tests in
`tests/unittest/disaggregated/test_kv_transfer_session_lifecycle.py`
pass without modification — they exercise the Python policy layer,
which this PR doesn't touch. Confirmed locally.

C++ multi-GPU integration tests in
`cpp/tests/unit_tests/multi_gpu/cacheTransceiverTest.cpp` use the
public API only; no test changes needed.

### Empirical validation (post-build)

Re-run the d224 / d227 burst suite from
[`disagg-wedge-pr13818-findings-2026-05-06.md`](https://github.com/yifjiang/dynamo-gb200-test/blob/main/docs/disagg-wedge-pr13818-findings-2026-05-06.md):

| Repro | Pre-fix | Post-fix expectation |
|---|---|---|
| d224 r1 (TCP, conc=128, no canary) | `is_shutdown` at T+~3 min, permanent wedge | quarantine count climbs (observability) but health stays True; bursts drain; `is_shutdown` does NOT fire |
| d224 decode-side asymmetry | only 1 UNHEALTHY flip, never `is_shutdown` | per-entry stuck check fires; reaches `is_shutdown` if entries genuinely stuck |
| d227 r4 (Pattern C, canary on) | recovers via canary | unchanged: peer-crash scenarios still trigger no-progress timeout AND per-entry stuck (whichever first) |

### What to look for in the new UNHEALTHY log line

Old format:
```
CacheTransceiver flipping to UNHEALTHY: quarantined=N budget=M sinceProgressMs=X deadlineMs=Y
```

New format:
```
CacheTransceiver flipping to UNHEALTHY: noProgressTimeout=B perEntryStuck=B quarantined=N sinceProgressMs=X deadlineMs=Y
```

The two booleans tell you which condition tripped. `quarantined=N`
is now informational only.

## Risks

1. **Quarantine count can grow unboundedly** under sustained burst
   if completions never catch up. Memory cost is small (one
   `TrackedFuture` per pending request), but worth monitoring via
   `getHealth().quarantined_transfer_count` in dashboards.
   Mitigation: combine with PR #13824's deadline-anchor fix to
   reduce false positives at the source.

2. **`mQuarantineBudget` is now dead code at runtime** but kept as
   a public field for backward compat. Future cleanup PR can remove
   once downstream tooling has migrated.

3. **Per-entry `quarantinedAt` adds 16 bytes per `TrackedFuture`.**
   Negligible at typical concurrency (≤ 128).

## Implementation summary

7 LOC across 2 files, plus the new docs:

| File | Change |
|---|---|
| `cpp/include/tensorrt_llm/batch_manager/cacheTransceiver.h` | Add `quarantinedAt` field to `TrackedFuture`; clarify comments on `mQuarantinedTransferCount` and `mQuarantineBudget` (now observability-only). |
| `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` | Set `entry.quarantinedAt = now` in `maybeQuarantineLocked`. Replace `overBudget`-driven health check with `noProgressTimeout || perEntryStuck`. New UNHEALTHY log line surfaces both signals. |
| `docs/source/features/disagg-kv-transfer-decouple-quarantine.md` | This file. |

The 19 existing Python unit tests continue to pass without
modification.
