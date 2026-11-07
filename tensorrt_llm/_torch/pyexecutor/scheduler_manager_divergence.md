# KVCacheV2Scheduler ↔ KVCacheManagerV2: assumptions and divergence points

## Summary

`KVCacheV2Scheduler` treats `KVCacheManagerV2` as a try-it-and-see allocator: it calls manager APIs and trusts the boolean/void result, without a page-capacity pre-check or a way to reconcile manager state afterward. `BudgetTracker` is a separate limit on per-iteration tokens, request count, and PEFT pages — it does not track KV pages.

## Manager API assumptions

| API | Scheduler assumes | Divergence risk |
| --- | --- | --- |
| `prepare_context` | Success means the request's cache is ready for scheduling. | It may create or resume a cache and apply prefix reuse before admission is actually decided. |
| `resize_context` | Success means context capacity is ready. | A later failure can leave the enlarged cache sitting unused. |
| `try_allocate_generation` | Success means the cache is active with one more generation step of capacity. | A failed call may already have resumed the cache and recorded draft-length state before the resize step fails. |
| `_suspend_request` | Suspension is atomic and can't fail. | It clears runtime state, then suspends the primary manager, then the draft manager, one after another, with no rollback. Each manager's own `suspend_request` is a no-op if that cache is absent or already inactive. |
| `prepare_disagg_gen_init` | Prompt and draft capacity are ready for disaggregated generation. | It prepares/resumes before resizing, so a later failure can leave earlier changes in place. |

## Generation allocation and eviction

When `try_allocate_generation()` fails, the scheduler tries to evict another request (`_try_evict_for_gen()`) and retries; if that doesn't free enough capacity, it evicts the current request instead.

`_try_evict_for_gen()` does the eviction in this order: suspend the victim, add it to `evicted`, advance the scheduling frontier, then retry allocation.

This assumes `_suspend_request()` always succeeds. If suspending the primary cache works but suspending the draft cache raises an exception:

- The primary cache is suspended.
- The draft cache is left as-is, possibly still active.
- `req.py_batch_idx` has already been cleared.
- The victim is never added to `evicted`.
- The scheduling frontier is never advanced.
- The exception stops scheduling for this iteration; nothing is rolled back or reconciled.

Self-eviction works the same way: the request is only added to `evicted` after suspension returns.

Because of this, the primary and draft caches can end up disagreeing if the second suspend call raises. They can also drift silently when one manager has no cache, or an already-suspended cache, since suspension is a no-op in those cases. The native `suspend()` call does have a known throw site, but whether it's actually reachable in a correct call sequence is unconfirmed.

## Context scheduling and rollback

`_try_schedule_context_chunked()` calls `prepare_context()`, computes and stores chunk state, calls `resize_context()`, then does cross-context scheduling.

The manager can already be modified before that last step fails:

- `prepare_context()` may have created or resumed the cache and applied prefix reuse.
- `resize_context()` may have grown capacity and set `py_ctx_pre_resize_cap`.
- If cross-context scheduling fails afterward, the scheduler suspends the cache at its enlarged size — it does not free it.

Resize failure behaves differently depending on the chunk:

- First chunk: the manager suspends the cache before returning `False`.
- Later chunks: the cache stays active at its previous capacity.

Either way, the scheduler just treats the request as `SKIP`.

`revert_allocate_context()` is only used for disaggregated-generation-init requests that get deferred by transfer-buffer admission control. It normally shrinks and suspends the cache, but frees it entirely if history has moved past the rollback point. It is not a general rollback for every discarded context request: the `can_queue=False` path only reverts generation allocations, so an already-resized context cache can stay allocated.

## BudgetTracker versus manager pages

`BudgetTracker` and the manager track different things, so their views can drift apart:

- The tracker counts logical forward tokens and scheduled requests; the manager allocates page-rounded physical capacity across pools and tiers.
- The tracker is rebuilt every scheduling iteration; manager state persists across iterations.
- Manager changes from failed or later-rejected scheduling attempts don't show up in tracker commits.
- Prefix reuse, `num_extra_kv_tokens`, page rounding, tier residency, and suspended allocations all use manager capacity but aren't counted as tracker tokens.
- A failed generation allocation can still resume a cache or record draft-length metadata, even though the tracker records nothing for it.
- At disaggregated completion, the manager may reserve draft capacity while the scheduler's budget for it is zero.
- Draft-manager reserve padding is internal to the manager and invisible to the scheduler's budget.
- PEFT accounting is a separate pool; suspending a KV cache does not release PEFT residency.
- The request-count budget and the manager's `IndexMapper`/page capacity are independent limits on each other.
