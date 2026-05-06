# `[wedge-trace]` log lines — quick reference

Companion doc to the verify-wedge runbook (in TRT-LLM PR #13796). When
the disagg KV transfer wedges, the four polling loops where the
process can stall now emit a periodic `[wedge-trace]` log at
`WARNING` level, so an operator can identify the wedge location from
log output alone — no `gdb` attach required.

All four log sites share the same heartbeat interval, controlled by
**`TRTLLM_DISAGG_WEDGE_TRACE_INTERVAL_MS`** (default `5000`). Set to
`0` to disable.

## The four heartbeats

| Log prefix | Source | Wedge it surfaces |
|---|---|---|
| `[wedge-trace] NixlTransferStatus::wait still NIXL_IN_PROG …` | `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.cpp` (inside `NixlTransferStatus::wait`) | **Sender** worker stuck polling `getXferStatus` — the in-flight `ucp_request` is not completing. UCX endpoint or peer wedged. |
| `[wedge-trace] AgentConnectionManager::waitForNotification still pending …` | `cpp/tensorrt_llm/executor/cache_transmission/agent_utils/connection.cpp` (inside `waitForNotification`) | **Receiver** worker stuck polling `getNotifs` — the post-write notification from the sender never arrived. Sender crashed or notification dropped. |
| `[wedge-trace] BaseTransBufferManager::assignBufferIndex blocked …` | `cpp/tensorrt_llm/batch_manager/baseTransBuffer.cpp` (inside `assignBufferIndex`) | A new request can't acquire a slot — slot is held by a wedged worker. **Downstream effect** of one of the above wedges. |
| `[wedge-trace] CacheTransceiver::requestAndReceiveSync executor thread blocked …` | `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` (inside `requestAndReceiveSync`) | Executor thread blocked on `future.get`. **Only on the** `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1` **path.** Pairs with one of the worker-thread heartbeats above. |

## Sample log output during a wedge

```
[2026-05-06 10:42:18.131] WARN  [wedge-trace] NixlTransferStatus::wait still NIXL_IN_PROG after 5023 ms (handle=0x7f1c4c01a3b0, requested timeout_ms=-1). Sender-side getXferStatus poll is not making progress; peer or UCX endpoint may be wedged.
[2026-05-06 10:42:18.155] WARN  [wedge-trace] BaseTransBufferManager::assignBufferIndex blocked waiting for a slot for 5009 ms (concurrence=1 budget=1). The slot is most likely held by a wedged worker thread; check sender/receiver heartbeats and run the stuck-slot diagnosis.
[2026-05-06 10:42:23.131] WARN  [wedge-trace] NixlTransferStatus::wait still NIXL_IN_PROG after 10025 ms (handle=0x7f1c4c01a3b0, requested timeout_ms=-1). …
[2026-05-06 10:42:23.155] WARN  [wedge-trace] BaseTransBufferManager::assignBufferIndex blocked waiting for a slot for 10010 ms (concurrence=1 budget=1). …
```

Reading this:

* The handle pointer `0x7f1c4c01a3b0` identifies which NIXL transfer
  is wedged. Cross-reference with earlier `start recv bufferIdx`
  debug logs (see
  [`disagg-kv-transfer-debug-stuck-slot.md`](disagg-kv-transfer-debug-stuck-slot.md))
  to map the handle to a request id.
* `concurrence=1 budget=1` means the recv-buffer pool has the default
  one slot and it is fully consumed; this is the classic "single
  wedged transfer pins the rank" pattern documented in
  [`disagg-kv-transfer-session-lifecycle.md`](disagg-kv-transfer-session-lifecycle.md).

## Mapping log line → recovery action

| Log line you see | Most likely cause | Action |
|---|---|---|
| Only `NixlTransferStatus::wait` heartbeats fire (sender side) | Sender's `ucp_request` not completing — peer down or UCX endpoint wedge | Health-driven worker restart via PR #13796's `_check_transceiver_health` (default ~3 min total). Ask NIXL Q1.2 — should `getXferStatus` surface UCX endpoint-error as a final state? |
| Only `waitForNotification` heartbeats fire (receiver side) | Sender notification dropped or sender crashed before sending it | Same health-driven restart path. Ask NIXL Q4.1 — are notifications reliable? |
| Both sides heartbeat | Both peers wedged; usually one started it (often a downstream pod restart) | Restart the worker that's hardest to recover. |
| Only `assignBufferIndex` heartbeats fire (no NIXL-level heartbeat) | The thread holding the slot is wedged in something OTHER than NIXL — most likely CUDA, MPI, or a TRT-LLM mutex | `gdb -p` the process to find the slot owner; this is NOT a NIXL wedge. |
| Only `requestAndReceiveSync` heartbeat fires (no worker heartbeat) | Possible. Means the receive worker died without setting the promise. Or `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1` is on AND the worker is healthy but slow | Check for crashed worker thread via `cat /proc/$PID/task/*/comm`. |

## Disabling the heartbeats

```bash
TRTLLM_DISAGG_WEDGE_TRACE_INTERVAL_MS=0
```

…or for finer control:

```bash
# Only fire after 30 s of no progress (suppresses occasional slow transfers)
TRTLLM_DISAGG_WEDGE_TRACE_INTERVAL_MS=30000

# Tight loop, useful in test environments
TRTLLM_DISAGG_WEDGE_TRACE_INTERVAL_MS=1000
```

The heartbeats themselves are cheap (a single `chrono::steady_clock::now()`
plus a `TLLM_LOG_WARNING` per interval per wedged thread). They do
not affect throughput on the happy path.

## Cross-references

* [`disagg-kv-transfer-debug-verify-wedge.md`](disagg-kv-transfer-debug-verify-wedge.md)
  — the gdb / py-spy / NIXL-telemetry / UCX-stats verification recipes.
* [`disagg-kv-transfer-debug-stuck-slot.md`](disagg-kv-transfer-debug-stuck-slot.md)
  — the start-recv-without-finish-recv slot-leak diagnosis.
* [`disagg-kv-transfer-session-lifecycle.md`](disagg-kv-transfer-session-lifecycle.md)
  — recovery model and global progress deadline.
* [`disagg-nixl-open-questions.md`](disagg-nixl-open-questions.md) —
  upstream NIXL questions whose answers would shorten the recovery path.
