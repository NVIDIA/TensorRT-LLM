# Implementation Plan for Python Native Disaggregated KV Transfer Ownership

[< Back to ownership contract](README.md)

| Field | Value |
|---|---|
| **Owner** | Chien-Chun Hung |
| **Status** | Draft implementation plan |
| **Created** | 2026-07-08 |
| **Last updated** | 2026-07-08 |

## Performance and Capacity Contract

Ownership safety must not undo the performance purpose of bounce transfer.

- Direct transfer adds no data copy and no new CUDA synchronization.
- Context lookup and result processing are O(1) average; metadata is O(writers)
  per slice.
- Locks are per context or sharded; no global lock is held across NIXL or CUDA
  work.
- Executor polling remains bounded and non-blocking.
- Source KV is released at its earliest safe per-path boundary, enabling chunked
  early release rather than pinning the whole request until all slices finish.
- Bounce arena sizing remains configuration-driven and opt-in.

Fail-closed retention can consume KV or bounce capacity. This is intentional
under uncertainty and must be visible operationally. It is preferable to an
invisible cross-request overwrite. Bounded reclamation is a follow-up enabled by
stronger transport/protocol evidence.

## Observability

At minimum, expose counters or structured logs for:

- active send and receive contexts;
- contexts logically terminal but physically draining;
- in-doubt context count, bytes, age, and reason;
- active source/destination KV leases;
- active send/receive bounce leases;
- direct, bounce, and fallback writers;
- duplicate, unexpected, contradictory, and stale-generation results;
- partial-publication failures;
- shutdown drain duration and non-drained context count;
- admission failures caused by safely retained capacity.

Logs identify transfer ID, slice, generation, writer identity, path, logical
outcome, physical state, and held leases without logging raw data contents.

## Implementation Plan

The work should land as a chain of focused PRs rather than one large refactor.
Size estimates are directional and include only production code unless noted.

### Phase 1 — Receiver safety foundation

**Size:** Medium-large; approximately 7–10 production files, 600–1,000
production lines, and 700–1,200 test lines. Generation negotiation or a new V1
lease binding can increase this estimate.

- Add `TransferRegistry` and `RecvTransferContext` above the bounce package.
- Split logical outcome, exposure, access, target, and data state.
- Create the exact writer-operation plan and range authorization before
  publication.
- Move the publication boundary before message send.
- Route all results through the registry before session lookup.
- Drain late sibling results after logical failure/session removal.
- Replace timed bounce reuse with indefinite in-doubt retention.
- Rename or narrow the current bounce-only `TransferContext`.
- Add a minimum allocator-enforced destination-KV lease before resolving or
  publishing destination pointers; Phase 1 cannot claim direct-path safety
  without it.
- Treat any failure lacking documented quiescence evidence as in doubt.
- Preselect the exact ADP writer cohort or exclude that topology from the first
  rollout.
- Add safe interim shutdown behavior: no unmap if contexts remain in doubt.
- Meet the hard identity gate: add generation/versioning or prove and enforce
  non-reuse across late results and worker recreation.

Likely files:

- `tensorrt_llm/_torch/disaggregation/native/transfer.py`
- `tensorrt_llm/_torch/disaggregation/transceiver.py`
- `tensorrt_llm/_torch/disaggregation/native/bounce/core.py`
- `tensorrt_llm/_torch/disaggregation/native/bounce/impl.py`
- a new native transfer-context/registry module
- unit and fault-injection tests

Phase 1 is a receiver-safety milestone, not completion of the comprehensive
sender-and-receiver contract.

### Phase 2 — Complete KV leases and sender context

**Size:** Large; approximately 8–12 files, 500–1,000 production lines, and
600–1,000 test lines. A small C++/nanobind extension may be required.

- Complete the common source/destination `KVTransferLease` contract for both KV
  managers.
- Support atomic pending-free and overlapping reference-counted slice leases.
- Acquire source leases before gather or direct submission.
- Add `SendTransferContext` and track gather, NIXL, send-bounce, descriptor, and
  result-delivery state independently.
- Integrate with or generalize `AsyncTransferManager` rather than creating a
  second request-retention mechanism.
- Permit per-chunk source release at the earliest safe boundary without forcing
  request-wide pinning; PR #15803 provides C++-side prior art for the lease
  shape.
- Cover direct, bounce, and mixed-mode fan-in.

### Phase 3 — Two-phase shutdown and backend quiescence

**Size:** Medium; approximately 300–600 production lines and 400–700 test lines.

- Add backend quiesce/drain and reorder shutdown before
  deregistration/unmapping.
- Keep non-drained workers retryable; prohibit destructor fallback unmapping.
- Include independent auxiliary-transfer owners in the shared-registration
  shutdown barrier.
- Preserve independent earliest-safe release boundaries and retain only
  lightweight result-delivery/tombstone state after GPU leases retire.
- Add the required lifecycle metrics.

### Phase 4 — Protocol and liveness hardening

**Size:** Medium; approximately 200–500 production lines plus protocol tests.

- Add result acknowledgement/retry or receiver query for bounded recovery from
  message loss.
- Add a richer per-operation query/abort API beyond the minimum quiescence
  classification required by Phase 1.
- Consider per-operation abort only if the backend can guarantee no later memory
  access after acknowledgement.

The core effort is approximately 1,500–3,000 production lines plus substantial
concurrency and integration tests. The C++ `CacheTransceiver` remains untouched;
cross-language work is limited to shared KV-manager or NIXL binding contracts
that the Python runtime consumes.

## Validation Plan

### Deterministic unit and fault-injection tests

- Cancel immediately before, during, and after address publication.
- Writer A fails, the consumer/session closes, and writer B reports later.
- Mixed direct/bounce fan-in where one writer fails.
- Partial multi-writer publication where only a subset may have seen addresses.
- Timeout followed by late success and late failure.
- Duplicate, unexpected, contradictory, and stale-generation results.
- Gather launch, descriptor construction, NIXL submission, result-send, and
  scatter exceptions.
- Consumer callback exception during physical finalization.
- Attempted source KV reuse while direct NIXL read is blocked.
- Attempted destination KV reuse while direct write or scatter is blocked.
- Attempted send/receive bounce reuse while NIXL or scatter is active.
- Shutdown with blocked NIXL status, lost quiescence result, and queued/running
  scatter; assert that deregistration and unmapping do not occur.
- Exactly-once release under concurrent cancel/result and reordered duplicates.
- Lease acquisition racing request `free_resources()`.
- Overlapping slice leases on the same KV allocation generation.
- Malformed, overflowing, misaligned, or out-of-range NIXL and scatter metadata.
- Synchronous pre-submit failure versus ambiguous NIXL submission failure.
- Quiesced `NO_REMOTE_ACCESS` and quiesced-with-unknown-data outcomes.
- Global NIXL quiescence while scatter remains queued or active.
- KV registry drained while an independent auxiliary transfer remains active.
- Non-drained shutdown followed by a successful retry.
- Duplicate context identity insertion.
- Gather/scatter worker death or asynchronous CUDA failure.
- Sender GPU lease retirement while result delivery remains pending.

Tests must use barriers/hooks rather than probabilistic sleeps for the critical
publication and teardown races.

### Configuration matrix

- bounce disabled and enabled;
- Python KV manager V2 and C++-backed V1;
- single writer and multi-writer TP/ADP fan-in;
- direct, bounce, and per-writer fallback;
- single slice and chunked/multi-slice transfer;
- PyTorch executor and AutoDeploy with Python transceiver;
- bounce-ineligible/non-attention layer-group fallback;
- normal completion, cancellation, timeout, peer failure, and shutdown.

### Performance validation

- Direct-path throughput and CPU overhead with ownership enabled.
- Bounce throughput and gather/scatter overlap.
- Context-registry lock contention under high concurrency.
- Source KV early-release timing for chunked transfer.
- Memory high-water mark with normal drain and deliberately in-doubt contexts.
- Shutdown drain latency.

### Acceptance criteria

The design is implemented only when all of the following are true:

1. No address can be published without a registered context and live leases.
2. Session/request removal cannot drop a physical quiescence result.
3. KV and bounce allocators cannot reuse leased generations.
4. Every release path is idempotent and tied to quiescence evidence.
5. Direct, bounce, and mixed transfers share the same ownership path.
6. Timers alone never authorize reuse.
7. Shutdown cannot deregister or unmap memory while a context may access it.
8. New identity fields are negotiated or mixed versions are rejected before
   DMA addresses are exchanged.
9. The executor remains responsive while contexts drain.
10. In-doubt capacity and shutdown state are observable.
11. Lease acquisition and `free_resources()` cannot race into block reuse.
12. Every NIXL/gather/scatter range is authorized by a live generation lease.
13. Every local CUDA/NIXL accessor is independently tracked through quiescence.
14. Phase 1 either provides generation-safe identity or enforces non-reuse
    across worker recreation and all possible late results.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Lock-order deadlock across registry, session, allocator, CUDA, and callbacks | Per-context serialization; no blocking operations or callbacks under lifecycle locks; lock-order tests. |
| KV over-pinning or leaked leases | Exactly-once release, active-lease metrics, oldest-context diagnostics, and fault-injection coverage. |
| Capacity exhaustion from indefinite retention | Fail admission visibly; expose bytes/age/reason; add reliable result recovery or backend quiescence before bounded reclaim. |
| Treating failure as quiescence incorrectly | Track quiescence separately from data outcome and preserve an explicit in-doubt state. |
| Mixed direct/bounce accounting error | Record actual mode per exact writer and retain both candidate leases until mode is known. |
| Stale result advances a new transfer | Generation and endpoint epoch, tombstones, exact writer validation, and negotiated protocol. |
| Main-loop latency regression | Preserve bounded polling; perform draining on transfer/completion workers; benchmark registry contention. |
| Shutdown hangs indefinitely | Use an explicit deadline and return non-drained status, but never convert deadline expiry into unsafe unmap/reuse. |
| Rolling-upgrade incompatibility | Capability/version negotiation or pre-DMA rejection; no silent downgrade of safety fields. |
| Behavioral divergence between KV managers | One conformance test suite for the common lease API; implementation-specific adapters only below it. |
| Scope expands into cancellation consensus | Keep logical consensus in the adjacent cancellation design; consume its outcome through `TransferHandle`. |

## Alternatives Considered

### Expand the current bounce `TransferContext`

Rejected as the general architecture. That object is naturally scoped to a
receive bounce slot. Making it own direct writers, source KV, sender status,
request aggregation, and KV-manager leases would invert module boundaries and
make ownership disappear when bounce is disabled.

### Make `LlmRequest` the owner

Rejected. Its logical lifetime may end before physical retirement; it aggregates
multiple slices; and a live object reference does not necessarily prevent
allocator-level `free_resources()`. It remains an associated consumer through a
request-level handle.

### Treat timeout or fixed quarantine as safe reclamation

Rejected. Wall-clock duration is not proof that one-sided RMA has stopped. A
timer may trigger diagnostics, fail admission, or escalate process health; it
cannot authorize memory reuse.

### Solve only the bounce path

Rejected. Sender fallback can mix bounce and direct writers, and destination KV
is the resource at risk when a direct writer outlives logical failure. The
general owner must sit above `BounceTransport`.

### Unify Python and C++ transceivers now

Rejected for this project. Their implementations, flow control, and active
bounce proposals differ. They should share invariants and, where useful,
allocator/protocol primitives, but not be forced into one implementation before
the Python safety gap is closed.

## Open Implementation Questions and Phase Gates

These do not change the safety contract, but they determine phase sequencing.

1. Which exact NIXL states guarantee no later DMA after success, failure, abort,
   and peer loss?
2. Can the advertisement messaging layer prove non-delivery after a send error?
   If not, every attempted send remains `MAY_ACCESS`.
3. Can `unique_rid` be reused across worker recreation or retry? If yes or not
   provably no, generation/version work is a Phase 1 prerequisite.
4. What supplies a stable endpoint epoch across rank/process restart?
5. Can existing V1 block pinning enforce leases through explicit
   `free_resources()`, or is a new nanobind API required?
6. What is the smallest V2 KV-manager lease surface that supports per-slice early
   release without request-wide over-pinning?
7. Does the NIXL wrapper need a separate `quiesce()` operation before final
   shutdown so registrations can remain valid during drain?
8. Is result acknowledgement/retry required for the first production rollout,
   or is observable indefinite retention acceptable for all Python-native
   direct and bounce transfers?
9. Which independent auxiliary-buffer paths can outlive their request/session?
   They need a follow-up audit before the broader transceiver can claim complete
   ownership coverage.
10. Can every claimed TP/ADP topology identify its exact potential writer cohort
    before address fan-out, or which cells must be gated initially?

No answer to these questions may weaken the invariant by interpreting
uncertainty as retirement.
