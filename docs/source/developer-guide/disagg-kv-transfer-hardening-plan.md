# Disaggregated KV Transfer Hardening Plan

This note describes follow-up hardening work for disaggregated KV cache transfer.
It assumes the conservative request-lifetime patch is already in place: async
transceiver APIs and worker/future tracking structures carry
`std::shared_ptr<LlmRequest>` instead of raw `LlmRequest*`.

That patch fixes one class of bug: stale access to the `LlmRequest` object
itself. It does not, by itself, prove that all resources referenced by a live
`LlmRequest` are still valid. KV cache blocks, sequence slots, transfer buffer
pool entries, and cache-manager request mappings have separate lifetimes and
need their own ownership rules.

## Goals

1. Avoid transfer buffer-index leaks on exceptions and early returns.
2. Avoid double termination after context transfer completion.
3. Improve diagnostics for futures, buffer pools, request IDs, and worker drain.
4. Harden unknown exception handling without pretending the transport is healthy.
5. Prevent `_terminate_request()` while context KV send is still in progress.
6. Define a policy for broad `_handle_errors()` when generation KV receive is in
   flight.

## Non-Goals

- Do not make stuck transport operations magically interruptible.
- Do not erase a C++ future before the worker has reached quiescence.
- Do not report `cancel_request()` success for an already in-flight transfer
  unless the worker is known to have stopped touching request and KV resources.
- Do not keep serving after an unknown transport/backend exception unless the
  backend provides a clear health guarantee.

## Current Implementation Status

This branch has started the plan with the pieces that are local and low risk:

- `BufferIndexHolder` now owns send/recv transfer buffer slots and releases them
  on scope exit.
- `TransferSession` owns AgentConnection pre-assigned receive buffer slots from
  pre-assignment through formatter `unformat()`.
- `CacheFormatter`, `MLACacheFormatter`, and `RnnCacheFormatter` use RAII
  holders instead of manual free calls.
- `_handle_responses()` now checks
  `request.is_disagg_context_transmission_state` before the partial-reuse
  cleanup branch.
- `_terminate_request()` now defers termination while context KV send is in
  progress, including the disaggregated PP termination handler path.
- Broad `_handle_errors(requests=None)` now fails closed when generation KV
  receive is in flight: it emits client error responses where possible, clears
  local scheduling queues, marks the executor shut down, and intentionally does
  not free active request resources in-process.
- C++ worker/future paths now have catch-all handling for unknown exceptions and
  convert them into promise/future errors.

Still remaining:

- Add focused unit/fault-injection tests.
- Add an explicit process/transceiver unhealthy flag if we want C++ to expose
  health directly rather than surfacing errors through Python futures.
- If graceful in-process recovery is required later, add real generation receive
  transfer tracking instead of relying on fail-closed restart.

## 1. RAII BufferIndexHolder

### Problem

Several formatter paths manually pair:

- `assignBufferIndexForSend()` with `freeBufferIndexForSend(...)`
- `assignBufferIndexForRecv()` with `freeBufferIndexForRecv(...)`

Important sites:

- `cpp/tensorrt_llm/batch_manager/cacheFormatter.cpp`
  - send path around `assignBufferIndexForSend()`
  - receive path around `assignBufferIndexForRecv()`
- `cpp/tensorrt_llm/batch_manager/mlaCacheFormatter.cpp`
  - send path for MLA/indexer cache
  - receive path for MLA/indexer cache
- `cpp/tensorrt_llm/batch_manager/rnnCacheFormatter.cpp`
  - send path
  - receive path
- `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp`
  - Agent receive pre-assignment path that calls `assignBufferIndexForRecv()`
    before sending request/buffer info to peers.

Manual free is fragile. Any exception, `TLLM_CHECK`, or early return between
assignment and free can leave a pool slot marked in use. That can wedge later
transfers even if the request-level logic is otherwise correct.

### Proposed API

Add a small move-only RAII holder near `BaseTransBufferManager`, for example in
`cpp/tensorrt_llm/batch_manager/baseTransBuffer.h`.

```cpp
class BufferIndexHolder
{
public:
    enum class Direction
    {
        kSend,
        kRecv,
    };

    static BufferIndexHolder assignSend(BaseTransBufferManager& manager);
    static BufferIndexHolder assignRecv(BaseTransBufferManager& manager);

    // For pre-assigned ids that are owned by a caller but consumed later, allow
    // construction with an existing id and explicit ownership.
    BufferIndexHolder(BaseTransBufferManager& manager, Direction direction,
        std::optional<int> id, bool owns) noexcept;

    BufferIndexHolder(BufferIndexHolder const&) = delete;
    BufferIndexHolder& operator=(BufferIndexHolder const&) = delete;
    BufferIndexHolder(BufferIndexHolder&& other) noexcept;
    BufferIndexHolder& operator=(BufferIndexHolder&& other) noexcept;

    ~BufferIndexHolder() noexcept;

    std::optional<int> get() const noexcept;
    std::optional<int> release() noexcept;
    void reset() noexcept;

private:
    BaseTransBufferManager* mManager{nullptr};
    Direction mDirection{Direction::kSend};
    std::optional<int> mId{std::nullopt};
    bool mOwns{false};
};
```

Destructor behavior:

- If `mOwns == false`, do nothing.
- If `mId == std::nullopt`, do nothing.
- If direction is send, call `freeBufferIndexForSend(mId)`.
- If direction is recv, call `freeBufferIndexForRecv(mId)`.
- Destructors must be `noexcept`. If the free path can throw today, normalize it
  first or catch/log in the holder destructor.

### Usage Pattern

For a locally assigned send buffer index:

```cpp
auto bufferId = BufferIndexHolder::assignSend(*mCacheTransBufferManager);
auto result = mCacheTransBufferManager->getOrAllocateSendBuffers(
    bufferId.get(), targetNum, bufferSizes, bufferManager);

// Any exception before function exit frees the slot.
// Normal exit also frees the slot automatically.
```

For a locally assigned receive buffer index:

```cpp
auto bufferId = BufferIndexHolder::assignRecv(*mCacheTransBufferManager);
auto result = mCacheTransBufferManager->getOrAllocateRecvBuffers(
    bufferId.get(), targetNum, bufferSizes, bufferManager);
```

For pre-assigned AgentConnection ids, ownership is trickier:

- If the id is assigned in the same formatter scope, use a holder directly.
- If `dataTransceiver.cpp` assigns ids and passes them through connection
  metadata for a later formatter to consume, do not let a stack holder free the
  id at the end of `sendRequestInfo()`. That would be too early.
- Prefer storing recv buffer-index holders in `TransferSession`, or an
  equivalent per-request transfer object, so ownership spans from pre-assignment
  through `CacheFormatter::unformat()` / `MLACacheFormatter::unformat()` /
  `RnnCacheFormatter::unformat()`.
- Once the formatter takes ownership, avoid a second manual free.

### Implementation Steps

1. Add `BufferIndexHolder` and unit tests for move, release, reset, send free,
   and recv free.
2. Replace manual send/recv buffer-index frees in `cacheFormatter.cpp`.
3. Replace manual send/recv buffer-index frees in `mlaCacheFormatter.cpp`.
4. Replace manual send/recv buffer-index frees in `rnnCacheFormatter.cpp`.
5. For AgentConnection pre-assigned recv ids in `dataTransceiver.cpp`, decide
   and implement the ownership handoff:
   - either store holders in `TransferSession`, or
   - store them in a request-scoped object owned by the receive worker until
     `unformat()` completes.
6. Add debug logs on assign/free with request id, direction, buffer id, and
   manager/buffer kind.
7. Add a stress test that injects exceptions after assignment and verifies the
   next transfer can still acquire a slot.

## 2. `is_disagg_context_complete_state` Guard

This guard is already present in the conservative PR. The intended shape in
`_handle_responses()` is:

```python
if request.is_disagg_context_complete_state:
    # Transfer completion already handled cleanup for this request.
    pass
elif request.is_disagg_context_transmission_state:
    # Do not terminate while KV send is still in flight.
    pass
elif self.enable_partial_reuse_for_disagg and not self.kv_cache_manager.is_vswa and self.dist.pp_size == 1:
    requests_to_terminate.append(request)
else:
    requests_to_terminate.append(request)
```

The current PR includes the complete-state guard. The next hardening pass should
also make the in-transmission guard explicit and place it before the partial
reuse branch, so partial reuse cannot bypass the transmission check.

## 3. Prevent `_terminate_request()` During Context Transmission

### Problem

For context/prefill send, `AsyncTransferManager.start_transfer()` is the object
that tracks in-flight context transfers. It stores the request in
`_requests_in_transfer`, optionally pins blocks with
`kv_cache_manager.store_blocks_for_reuse(request, True)`, and only unpins in
`end_transfer()`.

Calling `_terminate_request()` while `request.is_disagg_context_transmission_state`
is true can fight that ownership model. Even if `shared_ptr<LlmRequest>` keeps
the request object alive, `_terminate_request()` may call
`resource_manager.free_resources(request)`, and resource managers can remove
sequence/block mappings that the sender still needs.

### Proposed Python Guard

Add a helper:

```python
def _can_terminate_request_now(self, request: LlmRequest) -> bool:
    if self.kv_cache_transceiver is None:
        return True
    if request.is_disagg_context_transmission_state:
        return False
    return True
```

Use it in all normal termination paths that are not fatal broad-error cleanup:

- `_handle_responses()`
- `_end_transfer_and_maybe_terminate()`
- request-scoped `_handle_errors(..., requests=[...])`
- cancellation handling after `finish_by_reason(...)`

For `_handle_responses()`, prefer the explicit ordering:

```python
if request.is_disagg_context_complete_state:
    pass
elif request.is_disagg_context_transmission_state:
    pass
elif partial_reuse_cleanup_condition:
    requests_to_terminate.append(request)
else:
    requests_to_terminate.append(request)
```

This means a context request can leave `active_requests` but remain owned by
`AsyncTransferManager.requests_in_transfer()`. When
`_check_disagg_ctx_cache_transfer_status()` later sees completion, it calls
`_end_transfer_and_maybe_terminate()`, which calls
`async_transfer_manager.end_transfer(request)` and only then terminates.

### Tests

Add a Python unit test with a fake transceiver / fake transfer manager:

1. Put a context-only request in `DISAGG_CONTEXT_TRANS_IN_PROGRESS`.
2. Force `_handle_responses()` to see `request_done`.
3. Assert `_terminate_request()` is not called.
4. Simulate `_check_disagg_ctx_cache_transfer_status()` completion.
5. Assert termination happens after `end_transfer()`.

## 4. Broad `_handle_errors()` Policy for In-Flight Generation Transfers

### Problem

Generation receive is different from context send:

- It is tracked by C++ `mRequesterFutures`.
- The receiver worker can be actively writing destination KV blocks.
- There is no Python `AsyncTransferManager` equivalent that pins/unpins the
  destination KV resources for receive.

The broad error path:

```python
def _handle_errors(self, error_msg=None, *, requests=None):
    failed_requests = requests if requests is not None else self.active_requests
    if requests is None:
        self.active_requests.clear()
    ...
    for request in failed_requests:
        self._terminate_request(request)
```

does not filter out `DISAGG_GENERATION_TRANS_IN_PROGRESS`. If it frees all
active requests while a generation receive worker is still in flight,
`shared_ptr<LlmRequest>` keeps the request object alive, but it does not pin the
destination KV blocks. The worker may still be writing to resources that the
resource manager has removed or reused.

### Recommended Policy

Treat broad `_handle_errors(requests=None)` as fatal if any active request is in
disaggregated transfer:

```python
inflight_transfer_requests = [
    request for request in self.active_requests
    if request.is_disagg_context_transmission_state
    or request.is_disagg_generation_transmission_in_progress
]

if requests is None and inflight_transfer_requests:
    self._mark_unhealthy_for_restart(...)
    self.should_stop_processing = True
    self.shutdown_event.set()
    # Do not call _terminate_request() on in-flight transfer requests.
    # Let process restart release GPU and transport resources.
```

Reasoning:

- Broad `_handle_errors()` callers include hang detector, decode exceptions,
  forward exceptions, sampling/setup/update exceptions.
- These are not narrow request-local cleanup paths.
- If a worker is inside UCX/NIXL/MPI or writing KV blocks, we do not have a
  proven safe in-process recovery protocol.
- The deployment already has canary/health checking that can restart an
  unhealthy pod.

The fatal path can still enqueue client error responses if that is safe and
non-blocking, but it should not free in-flight KV transfer resources and then
continue serving.

### Alternative: Graceful Deferred Termination

If we later need graceful recovery instead of process restart, add real
generation-transfer tracking:

1. Have `check_gen_transfer_status()` return completed/error request ids, like
   context status does.
2. Add a Python `generation_transfer_manager` that records in-flight generation
   receives and blocks termination while they are active.
3. Add `_pending_termination_after_transfer[request_id]`.
4. On request-local error/cancel while generation transfer is in flight:
   - mark pending error/cancel,
   - keep resources alive,
   - do not free the request,
   - let `check_gen_transfer_status()` complete the future,
   - then terminate and free resources.
5. If the future never becomes ready, rely on unhealthy-process restart.

Do not erase the C++ future merely because Python wants cleanup. Erasing the
future is only safe after the worker has completed or the worker has a proven
cancel/quiescence handshake.

## 5. Better Diagnostics

Add diagnostics that are actionable but not noisy at normal INFO level.

### Future Tracking

In `CacheTransceiver`:

- On insertion into `mSenderFutures` / `mRequesterFutures`, log request id,
  pointer address, state, vector size, and whether overlap is enabled.
- On status poll, log at DEBUG:
  - number of tracked sender/requester futures,
  - number ready,
  - number selected for completion,
  - oldest transfer age.
- On timeout or repeated non-ready status, log at WARNING with request id,
  elapsed time, transfer start time, and vector size.
- On erase, log request id, final state, and vector size after erase.

### Buffer Pools

In `BaseTransBufferManager`:

- Log assign/free with direction, buffer id, buffer kind, configured count, and
  dynamic-buffer mode.
- Expose a debug-only method to count outstanding send/recv slots.
- On destructor or shutdown, warn if any non-dynamic buffer slot is still
  marked in use.

### Worker Drain

In `CacheSender::Impl` and `CacheReceiver::Impl`:

- Log worker start/stop and request ids currently queued.
- On destructor/drain, log queue sizes and number of futures being joined.
- When a worker sets a promise exception, include request id and context request
  id when available.

### Python Error Paths

In `_handle_errors()`:

- Log whether `requests` is `None` or request-scoped.
- Log request ids grouped by state.
- If broad error cleanup sees in-flight transfer requests, emit a single
  high-severity log explaining that the process is being marked unhealthy and
  in-flight transfer resources are intentionally not freed in-process.

## 6. `catch (...)` Hardening

### Current Gap

Several worker paths catch `std::exception`, set promise exceptions, and keep
going. Unknown non-`std::exception` failures may bypass those handlers. Also,
continuing to serve after unknown transport/backend exceptions is risky because
the connection manager, CUDA stream, or transport backend may be in an unknown
state.

### Proposed C++ Pattern

Add catch-all blocks immediately after existing `catch (std::exception const&)`
blocks in worker/future completion paths:

```cpp
catch (...)
{
    auto error = std::runtime_error("Unknown exception in KV cache transfer worker");
    TLLM_LOG_ERROR("%s request id: %ld", error.what(), requestId);
    markUnhealthy(error.what());
    promise.set_exception(std::make_exception_ptr(error));
}
```

Candidate locations:

- `CacheSender::Impl::sendAndRemoveResponse`
- `CacheSender::Impl::response`
- `CacheReceiver::Impl::request`
- `CacheTransceiver::checkContextTransferStatus`
- `CacheTransceiver::checkGenTransferStatus`
- async send/receive helper lambdas launched with `std::async`

### Health Semantics

Prefer fail-closed:

- Convert the unknown exception to `std::runtime_error` for the future/promise.
- Mark the cache transceiver or process unhealthy.
- Surface the failure to Python.
- Let Python stop accepting work and rely on canary/pod restart.

Avoid catching unknown exceptions and continuing to serve unless the specific
transport backend guarantees it is still valid after that exception class.

## 7. Validation Plan

### Unit Tests

- RAII holder frees send/recv ids on normal scope exit.
- RAII holder frees send/recv ids on exception.
- Move construction and move assignment transfer ownership exactly once.
- Pre-assigned/borrowed ids are not double-freed.
- `_handle_responses()` does not terminate a context request in
  `DISAGG_CONTEXT_TRANS_IN_PROGRESS`.
- `_handle_errors(requests=None)` with in-flight generation transfer marks
  unhealthy and avoids `_terminate_request()` for those requests.

### Integration / Fault Injection

- Inject exception after send buffer assignment before manual free location.
  Verify subsequent transfer can acquire the slot.
- Inject exception after recv buffer assignment before concat/free. Verify slot
  is released.
- Simulate a generation receive future that never becomes ready, then trigger
  broad `_handle_errors()`. Verify process is marked unhealthy and no
  in-process free happens for the in-flight receive request.
- Simulate context send in progress and force `_handle_responses()` to emit the
  logical `LlmResponse`. Verify resource free waits until
  `AsyncTransferManager.end_transfer()`.

### Observability Checks

- Logs include request id for future insertion, poll, completion, erase, and
  exception.
- Buffer pool logs show balanced assign/free counts under fault injection.
- Broad-error fatal path emits one clear health log, not per-iteration spam.

## Suggested Implementation Order

1. Add `_can_terminate_request_now()` and context transmission guard in Python.
2. Decide and implement broad `_handle_errors()` policy for in-flight generation
   transfer. Prefer fail-closed/unhealthy restart first.
3. Add `BufferIndexHolder` for straightforward local send/recv formatter sites.
4. Handle AgentConnection pre-assigned recv ids with a request/session-scoped
   holder.
5. Add diagnostics.
6. Add `catch (...)` hardening and unhealthy marking.
7. Add fault-injection tests.
