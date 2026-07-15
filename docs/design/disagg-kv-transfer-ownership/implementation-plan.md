<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Implementation Plan for Python Native Disaggregated KV Transfer Ownership

[< Back to ownership contract](README.md)

| Field | Value |
|---|---|
| **Owner** | Chien-Chun Hung |
| **Status** | Draft implementation plan |
| **Created** | 2026-07-08 |
| **Last updated** | 2026-07-14 |

## Performance and Capacity Contract

Ownership safety must not undo the performance purpose of bounce transfer.

- Direct transfer adds no data copy and no new CUDA synchronization.
- Registry/identity lookup and fixed-operation state transitions are O(1)
  average. Per-segment bounds checks are O(segments), and fixed exact-mapping
  normalization is O(segments log segments) unless canonical wire ordering
  permits a linear scan. Dynamic-stream interval insertion/overlap validation is
  O(log(operations + segments)), and end-of-stream coverage/compaction is
  O(operations + segments), all within negotiated bounds. Metadata is
  O(contexts + writers + operations + segments) per transfer and bounded by
  fixed-manifest or dynamic-stream envelopes plus endpoint-global capacity.
- Locks are per context or sharded; no global lock is held across NIXL or CUDA
  work.
- Executor polling remains bounded and non-blocking.
- Session capability negotiation is amortized, and healthy per-transfer
  admission piggybacks on the existing KV request/target response without a new
  request/ack RTT.
- Source KV is released at its earliest safe per-path boundary, enabling chunked
  early release rather than pinning the whole request until all slices finish.
- Bounce arena sizing remains configuration-driven and opt-in; requested,
  transport-active, and per-transfer-engaged states are distinguished.

Fail-closed retention can consume KV or bounce capacity. This is intentional
under uncertainty and must be visible operationally. It is preferable to an
invisible cross-request overwrite. Bounded reclamation is a follow-up enabled by
stronger transport/protocol evidence.

### Expected metric impact

This is primarily a correctness and capacity-recovery project, not a NIXL
bandwidth optimization. On a healthy transfer, the expected data path is
unchanged. The owner adds host metadata, lookup, and state transitions, but no
new direct-path CUDA copy, kernel, synchronization, or device arena. When the
requested bounce transport is active, the configured send/receive arenas
introduced by PR #15618 remain the only preallocated bounce storage.

For the initial containment implementation specifically, no TTFT, throughput,
or NIXL bandwidth improvement should be claimed. The healthy sender adds an O(1)
operation-ledger lookup plus small state/counter transitions, exact-session
queue binding, and serialized socket access; the receiver adds registry
transitions and bounds checks. Duplicate request delivery is cheaper because it
replays or joins the existing operation instead of launching another transfer.
Lazy destination-manifest construction can reduce host allocation/CPU work when
a bounce slot is unavailable and the operation falls back to direct transfer,
but it does not change transferred bytes. The slice allocates no additional
device arena beyond the opt-in bound buffers from PR #15618. Under cancellation,
missing results, or ambiguous NIXL status, retained Python owners and bounce/KV
capacity can increase memory residence and reduce admission throughput; that is
the intended fail-closed result. The detailed metrics below are the target
instrumentation contract and are not all exported by this first slice.

| Scenario | Expected impact |
|---|---|
| Healthy direct or bounce transfer | With admission piggybacked on the existing request/target response, throughput, TTFT, and transfer-task latency should remain within benchmark noise. CPU time and host memory rise slightly with O(contexts + writer operations + descriptor/scatter segments + lifecycle records + outstanding generations) metadata and state transitions. An implementation that adds a per-transfer RTT must declare and qualify that separate behavior. |
| Chunked bounce send | Unique transfer-leased source-KV byte-seconds/high-water mark can decrease because a slice may release source KV after gather completes rather than at request-wide completion; the pending-free subset shows actual admission capacity recovered after request cleanup. |
| Chunked direct send | Unique transfer-leased source-KV retention can decrease when each slice is released after its NIXL operation becomes definitively quiescent. There is no extra copy. |
| Known-terminal failure that currently strands state | Registry-first routing should recover pending-free KV and unique bounce-slot capacity as soon as the last physical accessor retires, reducing excess post-logical retention, admission loss, and fault-period goodput degradation. |
| Known-terminal direct/mixed failure that currently frees KV too early | The owner intentionally increases lease duration and possibly unique/pending-free high-water marks to the true quiescence boundary. That increase closes unsafe reuse and must not be classified as a regression. |
| Lost result or ambiguous transport failure | Exact unique bounce-slot bytes and unique/pending-free KV bytes remain unavailable, so high-water marks, admission failures, and tail latency can worsen as safe capacity is exhausted. This is an intentional safety tradeoff until bounded recovery exists. |
| Many dynamic streams reserve maximum envelopes but materialize little work | Host allocation may stay low while reserved credits reduce admissible concurrency and goodput. Reservation utilization/slack and reservation-caused rejection determine whether a configured envelope is rollout-safe. |
| Shutdown with in-flight work | Graceful shutdown may take longer or return non-drained. Unsafe deregistration/unmapping events must fall to zero. |

The main capacity metrics are:

```text
lease_token_byte_seconds = sum(token_range_bytes * token_lifetime)
unique_transfer_leased_byte_seconds = integral(union of ranges with transfer_leases > 0)
pending_free_kv_blocked_byte_seconds = integral(bytes with request_owners == 0 and transfer_leases > 0)
reservation_utilization(kind) = materialized_credits(kind) / reserved_credits(kind)
reservation_slack(kind) = reserved_credits(kind) - materialized_credits(kind)
post_logical_retention = max(0, last_lease_release - logical_terminal_time)
early_release_lead(resource) = max(0, logical_terminal_time - lease_release_time)
```

Lease-token byte-seconds intentionally count overlapping tokens and measure
ownership bookkeeping. Unique transfer-leased bytes deduplicate overlapping
ranges by allocation generation. Pending-free blocked KV bytes are the subset
whose reuse is prevented solely by transfer leases; KV capacity-recovery and
admission claims use that subset rather than token sums. Token metrics may be
attributed to each operation path, but deduplicated unique and pending-free
capacity gauges are path-neutral. An optional path breakdown must use mutually
exclusive `direct_only`, `bounce_only`, and `mixed` allocation-generation
classes so its series remain additive. `post_logical_retention`
measures failure/cancellation drain without becoming negative when a source
chunk retires early. Per-resource `early_release_lead` and lease duration
measure that intended early release. Improvements should be claimed only for
the applicable scenario. A normal-path latency or bandwidth improvement is not
an expected outcome of ownership alone.

## Observability

At minimum, expose counters or structured logs for:

- active send and receive contexts;
- active fixed-operation manifests and dynamic writer streams, including open
  streams and their enrolled/high-water-mark operation counts;
- configured, reserved, materialized, and remaining context/slice,
  fixed-manifest operation/descriptor-segment/authorized-byte, and
  dynamic-stream operation/segment/byte capacity, plus reservation utilization,
  unused slack, envelope/global-capacity rejection, and reservation-caused
  rejection;
- contexts logically terminal but physically draining;
- in-doubt context count, bytes, age, and reason;
- pre-cancel tombstone count, oldest age, capacity rejections/backpressure, and
  fence-based retirement;
- configured, used, and remaining canonical lifecycle-record/control-state
  capacity;
- configured, used, and remaining endpoint replay-window capacity, retired-floor
  advancement, sparse generation gaps, oldest gap age, and backpressure;
- active source/destination KV leases;
- active send/receive bounce leases;
- lease-token bytes/byte-seconds by endpoint and operation path; path-neutral
  unique transfer-leased bytes/byte-seconds and pending-free KV
  bytes/byte-seconds blocked solely by transfer leases, optionally split only
  into mutually exclusive `direct_only`, `bounce_only`, and `mixed` classes;
- bounce requested, transport active/inactive, and direct, bounce, or fallback
  writers so configured-but-inactive runs cannot claim bounce coverage;
- gather, NIXL, scatter, registry lookup, and lifecycle-lock wait latency;
- transfer admission latency and bounded capacity-rejection latency;
- healthy piggyback and exceptional skip/close/reject control-message counts and
  round-trip latency, so an accidental normal-path RTT is visible;
- logical completion time, last lease release, post-logical retention, and
  per-resource early-release lead;
- duplicate, unexpected, contradictory, and stale-transfer-generation results;
- partial-publication failures;
- shutdown drain duration and non-drained context count;
- rollback requested, blocked/non-drained, and completed;
- admission failures caused by safely retained capacity.

Aggregate metrics and benchmark records use low-cardinality dimensions for
ownership mode, negotiated protocol/capability bucket, rollout cohort,
KV-manager implementation, and executor. Operation/token metrics also use path;
deduplicated capacity gauges omit it unless they use the mutually exclusive
classes above. Request/transfer identities remain in structured logs rather
than metric labels. Logs identify transfer ID, target slice, transfer
generation, writer/stream/operation identity, path, logical outcome, physical
state, and held leases without logging raw data contents.

The existing sender `transfer_latency_ms` timer begins after bounce gather and
the receive task completes after scatter. Raw NIXL throughput can therefore hide
gather/scatter and lifecycle overhead. Benchmarks must report stage timings and
end-to-end TTFT/goodput in addition to the existing NIXL interval.

## Implementation Plan

The work should land as a chain of focused PRs rather than one large refactor.
Size estimates are directional and include only production code unless noted.

### Phase 1 — Receiver safety foundation

**Size:** Very large; approximately 18–28 production files, 2,000–3,500
production lines, and 2,000–3,500 test lines. Metrics export or a new V1 lease
binding can increase this estimate. Phase 1 should normally be a short stack:
identity/admission/replay and fixed envelopes; receive contexts plus KV leases;
then cancellation, shutdown veto, and minimum metrics. None is independently
rollout-safe until the full Phase 1 gate passes.

- Add `TransferRegistry`, registry-owned canonical `EndpointTransferRecord`,
  durable `EndpointEpochState`, access gate/handle, and `RecvTransferContext`
  above the bounce package. Requests/sessions hold only consumer references;
  contexts retain the shared gate without forming a record-context ownership
  cycle.
- Create a one-shot `TransferAdmissionToken` atomically with the request/session
  state transition that enables transfer scheduling. Implement token-to-registry
  `admit_and_bind()` so local cancellation either prevents token creation,
  aborts pending admission, or observes/aborts the canonical bound handle before
  any peer authorization.
- Make `begin_shutdown()` close registry admission and snapshot all committed
  records in one registry transaction. A racing pending token is
  shutdown-aborted if it loses; a winning record is included, and its later
  peer-authorization boundary is handle-gated and either owned or suppressed.
- Assign one negotiated admission initiator. Allocate its transfer generation
  monotonically within its stable endpoint epoch, committing counter advancement
  only with replay-entry reservation, canonical record creation, and token
  binding after capacity checks. Negotiate the authenticated initial floor and
  maximum outstanding-generation window, and maintain a contiguous retired
  floor plus bounded sparse live/out-of-order-retired window per negotiated
  endpoint-epoch pair. Atomically mark a generation retired before deleting its
  canonical record; backpressure gaps/window exhaustion rather than evicting
  anti-replay state. Convert every committed generation whose normal admission
  control is not sent into an authenticated generation-skip record, and block
  higher-generation admission control until its acknowledgement. A cumulative
  skip watermark cannot cross a locally live/published/unclosed generation. If
  admission control was sent but no address published and the peer rejects,
  reserve/close the peer replay entry first and retire both sides through an
  authenticated creation-close/reject acknowledgement. After address
  publication, require the full operation/stream closure and physical-retirement
  predicate instead.
- Piggyback generation/epoch/fixed-envelope admission on the existing KV
  request and target-advertisement response. Commit the receiver record before
  that response; do not add a separate healthy admission acknowledgement.
- Split logical outcome, exposure, access, target, and data state.
- Key contexts by role, local endpoint epoch, and admission-authority epoch; key
  receive ledger entries by peer epoch, writer rank, and fixed-operation or
  writer-stream ID.
- Create the exact writer cohort, fixed-operation manifest, and range
  authorization, including expected logical source-to-target segment mappings,
  before publication. Dynamic sender-determined chunk streams are excluded from
  Phase 1 rollout and land in Phase 2.
- Negotiate per-transfer context/slice and fixed-manifest
  operation/descriptor-segment/authorized-byte limits. Reserve actual
  operation/authorized-byte credits and the worst-case permitted segment
  envelope atomically against configured endpoint-global capacity before
  publication; reject an oversized plan or exhausted pool without partial
  exposure.
- Move the publication boundary before message send.
- Put target slice and exact fixed writer-operation identity in each
  advertisement. Deduplicate replay before local work so one advertisement can
  launch at most one operation.
- Route every KV writer-operation result through the registry before session
  lookup; keep auxiliary results on their independent ownership path.
- Route remote cancellation and every state-mutating control message through the
  same transfer-generation/endpoint-epoch validation. Replace bare-`unique_rid`
  pre-cancellation state with registry-atomic transfer-generation-aware
  records. Reserve control-state capacity at admission, before the peer may send
  lifecycle messages. An arbitrary unnegotiated control poisons/closes the
  endpoint session in O(1); it does not allocate an unbounded deny entry.
- Add a minimum negotiated generation/epoch-scoped creation-close/cancel-ack
  handshake in Phase 1. The peer acknowledges only after its local producer gate
  prevents later context creation; retry/deduplicate the acknowledgement and use
  it to retire zero-context pre-cancel state. This is not the Phase 3 submission
  fence and cannot retire any published address.
- Retain unmatched pre-cancel state until producer sealing or acknowledged
  creation close, enforce capacity through pre-admission backpressure, and
  export configured/used/remaining capacity plus count/age/rejection metrics.
- Drain late sibling results after logical failure/session removal.
- Replace timed bounce reuse with indefinite in-doubt retention.
- Keep the initial containment's renamed `RecvBounceContext` narrow and local
  to the bounce package; optionally reduce it further to `RecvBounceLease`
  state rather than promoting it into the general owner.
- Add an allocator-atomic `snapshot_and_lease(request, slice_spec)` operation for
  destination KV before copying block IDs, resolving pointers, or publishing
  addresses. Construct the receive context and wire slice from the immutable,
  allocation-generation-bearing lease snapshot; Phase 1 cannot claim
  direct-path safety without it.
- Add a Python-native lifecycle cancellation adapter with a structured
  logical/physical result. Select it explicitly for `KvCacheTransceiverV2`
  without changing the C++ transceiver's boolean contract. Let PyExecutor drop
  destination-KV ownership while the lease makes it pending-free, but preserve
  independent session/auxiliary cleanup gates; retain the sender-side legacy
  gate until Phase 2. Shared manager accounting may veto teardown for a live
  generic owner or release, but the adapter must not retire a C++ transport
  owner or treat local C++ object destruction as quiescence evidence.
- Make receiver context insertion and `TransferHandle` enrollment atomic with
  respect to handle abort/sealing. Gate every access boundary on the durable
  gate before taking the context lock, abort the gate on first failure,
  cancellation, or shutdown before enumerating contexts, and unwind a losing
  preparation before target publication.
- Define one monotonic transfer-attempt generation shared by every slice. Let
  only the receiver producer/session seal its handle, using an immutable
  expected-slice manifest after all fixed contexts are atomically enrolled.
- Treat any failure lacking documented quiescence evidence as in doubt.
- Preselect the exact ADP writer cohort or exclude that topology from the first
  rollout.
- Preserve a separate auxiliary-slot cleanup gate, or exclude generation-first
  configurations that transfer auxiliary data from Phase 1.
- Add the minimum transceiver `ShutdownResult`, caller-retention contract, and
  PyExecutor/creator/KV-manager teardown veto. If a context remains in doubt,
  Phase 1 must keep its worker, registry, manager, registration, and mapping
  alive; it cannot defer that safety boundary to Phase 3.
- Add negotiated transfer generation and both endpoint epochs to every
  lifecycle-mutating advertisement, result, and control message. Reject peers
  with the legacy `unique_rid`-only identity before address publication.
- Bind every admission to the negotiated endpoint-epoch pair. Recreating an
  endpoint without its replay window requires a fresh, non-reused epoch, and
  old-epoch traffic cannot bootstrap a replacement session.
- Ship minimum capacity diagnostics with the fail-closed path: in-doubt
  count/bytes/oldest age/reason, lease-token and unique transfer-leased
  KV/bounce bytes, pending-free KV bytes, post-logical retention, per-resource
  early-release lead, and admission/fallback failures caused by retained
  capacity.

Likely files:

- `tensorrt_llm/_torch/disaggregation/native/transfer.py`
- `tensorrt_llm/_torch/disaggregation/transceiver.py`
- `tensorrt_llm/_torch/disaggregation/native/bounce/core.py`
- `tensorrt_llm/_torch/disaggregation/native/bounce/impl.py`
- `tensorrt_llm/_torch/disaggregation/native/perf_logger.py` or the selected
  serving-metrics adapter
- destination KV-manager adapter/binding and PyExecutor cancellation integration
- `tensorrt_llm/_torch/pyexecutor/py_executor.py`,
  `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py`, and KV-manager teardown
  adapters for the minimum non-drained shutdown veto
- a new native transfer-context/registry module
- unit and fault-injection tests

Phase 1 is a receiver-safety milestone, not completion of the comprehensive
sender-and-receiver contract.

### Phase 2 — Complete KV leases and sender context

**Size:** Large; approximately 12–18 files, 1,000–1,800 production lines, and
1,200–2,200 test lines. A small C++/nanobind extension may be required. Split
source-lease/sender-context work from the dynamic-stream protocol when review or
validation scope would otherwise become too broad.

- Complete the common source/destination `KVTransferLease` contract for both KV
  managers.
- Support atomic pending-free and overlapping reference-counted slice leases.
- Acquire source leases before gather or direct submission.
- Promote the initial in-doubt `SendTransferContext` containment record into a
  canonical registry-owned context, backed by allocator-issued source leases,
  and track gather, NIXL, send-bounce, descriptor, and result-delivery state
  independently through normal as well as ambiguous completion.
- Extend atomic `TransferHandle` enrollment to sender contexts and reject a
  losing preparation before gather or NIXL launch.
- Retain the sender's canonical record/gate independently of its request-facing
  handle reference, and abort it on first failure/cancellation before admitting
  another chunk.
- Add a negotiated bounded dynamic-writer-stream mode for sender-determined
  chunks: advertise one authorized writer/target/range/stream, assign a distinct
  operation sequence per chunk, validate the operation/segment/byte budget, and
  reserve its worst-case metadata envelope before publication, then close
  enrollment with an authenticated end-of-stream high-water mark.
- Use an ordered interval/coverage structure so dynamic operation insertion and
  overlap checks are O(log(operations + segments)), while end-of-stream
  validation is one O(operations + segments) pass bounded by the negotiated
  maximum.
- Track reserved versus materialized operation/segment/byte credits and unused
  slack. Return only unused envelope credits when authenticated close makes
  future materialization impossible and establishes the exact final operation
  set, or after backend quiescence permits fail-closed record retirement; retain
  credits backing materialized entries through their retirement.
- Atomically enroll the final sender context with its final-slice proof before
  sealing the sender handle. Keep the receiver target leased until the stream is
  closed/fenced and every operation through the high-water mark is quiescent.
- Emit end-of-stream only after every covered sequence is already submitted or
  terminal `NO_REMOTE_ACCESS`; atomically close future submission first. A
  failed/cancelled stream uses terminal per-operation decisions or remains open
  until the applicable fence, never normal sealing as a shortcut.
- Deduplicate repeated stream advertisements and operation results, and fail
  closed on missing, conflicting, out-of-range, or out-of-budget sequences.
- Integrate with or generalize `AsyncTransferManager` rather than creating a
  second request-retention mechanism.
- Remove the remaining sender-side safe-to-free cancellation gate once source
  leases make request cleanup pending-free.
- Permit per-chunk source release at the earliest safe boundary without forcing
  request-wide pinning; PR #15803 provides C++-side prior art for the lease
  shape.
- Cover direct, bounce, and mixed-mode fan-in.

### Phase 3 — Submission fencing and safe shutdown

**Size:** Large; approximately 8–14 production files, 700–1,300 production
lines, and 800–1,500 test lines. This phase should normally be split into a
fence-protocol PR and a shutdown-integration PR.

- Add a negotiated peer/session/endpoint submission-fence capability and
  transfer-generation/endpoint-epoch/fixed-operation-or-stream-scoped
  request/ack framing for advertisements without a terminal result. An
  acknowledgement closes future submission against all covered advertisements.
  This strengthens the Phase 1 creation-close handshake after addresses may have
  been exposed.
- Add retry/idempotency handling for lost, delayed, duplicated, contradictory,
  and stale-epoch fence messages. Mixed-version peers that cannot provide the
  required fence are non-drained rather than silently downgraded.
- After fencing, add backend quiesce/drain for submitted or in-doubt work and
  reorder both before deregistration/unmapping.
- Extend the Phase 1 non-drained result and teardown veto with backend-wide
  quiescence, full retry/drain behavior, and process-health escalation.
- Include independent auxiliary-transfer owners in the shared-registration
  shutdown barrier.
- Preserve independent earliest-safe release boundaries and retain only
  lightweight result-delivery/tombstone state after GPU leases retire.
- Add shutdown/quiesce-specific lifecycle metrics beyond the Phase 1 minimum.

Likely files include `tensorrt_llm/_torch/disaggregation/transceiver.py`,
`tensorrt_llm/_torch/disaggregation/native/transfer.py`,
`tensorrt_llm/_torch/disaggregation/base/agent.py`,
`tensorrt_llm/_torch/pyexecutor/py_executor.py`,
`tensorrt_llm/_torch/pyexecutor/py_executor_creator.py`, and KV-manager teardown
adapters in `tensorrt_llm/_torch/pyexecutor/_util.py`.

### Phase 4 — Protocol and liveness hardening

**Size:** Medium; approximately 300–700 production lines plus protocol tests.

- Add result acknowledgement/retry or receiver query for bounded recovery from
  message loss.
- Add a richer per-operation query/abort API beyond the minimum quiescence
  classification required by Phase 1.
- Consider per-operation abort only if the backend can guarantee no later memory
  access after acknowledgement.

The core effort is approximately 4,000–7,300 production lines plus substantial
concurrency and integration tests. The C++ `CacheTransceiver` implementation,
data path, and wire protocol remain outside this scope. Cross-language work is
limited to shared KV-manager or NIXL binding contracts that the Python runtime
consumes. Shared `AsyncTransferManager`/`ResourceManager` accounting may observe
live generic ownership and fail closed, but it does not implement or retire the
C++ transport lifecycle.

## Rollout and Rollback

Ownership selection is an endpoint-session capability decided before context
creation and before any address is published. A temporary internal deployment
gate may select eligible canaries, but `kv_cache_bounce_size_mb` is not that
gate: bounce-disabled direct transfers need the same lifetime safety.

Rollout proceeds by configuration cohort:

1. Land deterministic tests and allocator conformance tests before enabling any
   production cohort.
2. Enable Phase 1 only where the exact writer cohort is known, both peers
   negotiate the ownership protocol, the destination KV manager implements the
   atomic lease contract, Phase 1 creation-close/cancel acknowledgement,
   monotonic generation/replay-window contract, and fixed-manifest/context
   capacity envelopes, and the minimum capacity metrics are available.
   Unsupported ADP cohorts, mixed versions, and generation-first auxiliary paths
   without an independent safe cleanup gate fail before address publication.
   Sender-determined dynamic chunk streams remain excluded until Phase 2.
3. Canary direct and bounce paths separately for each KV manager, then expand to
   fan-in, fallback, AutoDeploy, and chunked configurations as their matrix cells
   pass. A bounce canary requires positive transport-active and per-transfer
   bounce-engaged signals; configuration alone is not evidence.
4. Promote toward default-on only after the safety, capacity-recovery, and
   quantitative performance gates pass for the applicable cells.

Rollback never changes the owner of an active transfer. Stop new admission,
drain or fail closed all contexts negotiated under the new protocol, and only
then select another mode for newly negotiated sessions. If drain is non-terminal,
retain the owners and escalate process health; do not bypass leases to complete
rollback. A legacy path may remain available for explicitly excluded canary
cohorts during rollout, but it is not an in-place recovery mechanism for an
ownership failure. Rollback reports a blocked/non-drained result while any
context cannot retire and increments the corresponding low-cardinality cohort
metric.

## Validation Plan

### Deterministic unit and fault-injection tests

- Cancel locally immediately before transfer-eligibility/token creation and
  before/during `admit_and_bind()` with barriers. Cancellation prevents
  scheduling, aborts the pending token with no peer authorization, or observes
  and aborts the canonical bound handle; no unbound interval is permitted.
  Repeat request cleanup while admission is paused.
- Race `begin_shutdown()` with transfer-eligibility/token creation,
  `admit_and_bind()` before and after registry insertion, and the peer
  authorization boundary. Every run must either reject/shutdown-abort the token
  with no record/authorization or include the canonical record and any possible
  authorization in the shutdown snapshot and drain set.
- Cancel immediately before, during, and after address publication.
- Race cancellation against receiver and sender preparation/handle enrollment
  with barriers; assert that the context is either enrolled and cancelled or
  fully unwound before its publication/local-launch boundary. Reject slice
  creation after handle sealing.
- Close the handle, pause cancellation before it obtains a context lock, and
  race publication/gather/NIXL/scatter boundary entry; assert that the closed
  handle gate rejects every new boundary while previously admitted accessors
  remain owned.
- Drop every request/session-facing handle reference while a context is blocked,
  then race abort with an access boundary. The registry-owned record/gate must
  remain live. After current contexts retire but before producer sealing or a
  creation fence, reject a delayed slice and advertisement replay instead of
  creating a replacement handle.
- Fully retire and delete a canonical record, reuse its former target
  allocation, then deliver delayed duplicate admission and target-advertisement
  messages. Endpoint replay state must reject both before record creation or
  local launch. Retire generations out of order, verify sparse entries block
  replay, then close the gap and verify contiguous-floor compaction.
- Fill the negotiated generation window behind one missing generation and
  verify admission rejects in the same worker iteration without sleep/retry or
  eviction and within the frozen wall-time ceiling. Recreate an endpoint with a
  fresh epoch and prove that old-epoch traffic cannot bootstrap its session. An
  arbitrary unnegotiated control must poison the current session in O(1), after
  which all admission fails until fresh negotiation. Reconnect with the same
  epoch and a peer-claimed floor beyond a locally live/gapped generation;
  reject the inconsistent session without resetting local replay state.
- Repeatedly fail initiating admission before the generation/record/token
  commit and prove that the counter/floor/window do not advance. Then inject
  failures after local commit but before admission-control send, peer-side
  rejection after admission control but before address publication, ambiguous
  admission-control delivery, and abandonment after address publication. Lose,
  delay, or duplicate the generation-skip acknowledgement and prove
  higher-generation admission-control send remains blocked. Ambiguous normal
  admission must retry/deduplicate and creation-close rather than becoming a
  skip. Reject a contradictory cumulative watermark crossing a live or
  published record. After authenticated skip or pre-publication close/reject,
  both replay states advance/compact; post-publication abandonment retains the
  record/resources until full physical retirement.
- Race first logical failure against next-chunk enrollment and gather/NIXL
  launch; the failure must abort the shared gate before callbacks, reject new
  work, and retain already-possible accessors through quiescence.
- Race cancellation with final-slice enrollment and handle sealing. Cover
  missing, duplicate, reordered, and conflicting final chunks; success requires
  the immutable manifest or final-slice proof to name the complete slice set.
- Writer A fails, the consumer/session closes, and writer B reports later.
- Mixed direct/bounce fan-in where one writer fails.
- Partial multi-writer publication where only a subset may have seen addresses.
- Timeout followed by late success and late failure.
- Duplicate, unexpected, contradictory, and stale-transfer-generation results.
- Duplicate and reorder target advertisements. Assert that an identical fixed
  operation or writer-stream advertisement launches at most once and a
  conflicting replay fails closed.
- Exercise a dynamic writer stream with out-of-order results and authenticated
  end-of-stream. Cover a missing sequence, conflicting duplicate, out-of-range
  subrange, operation/segment/byte budget overflow, forged/premature
  end-of-stream, sequence-complete but in-range undercoverage/gaps, and
  a full-coverage source-to-target permutation. Also cover an open stream whose
  currently known operations are all quiescent; none may cause early target
  retirement or logical success.
- Pause an enrolled sequence before NIXL submission and attempt end-of-stream.
  The close must wait for an irreversible `SUBMITTED`/`NO_REMOTE_ACCESS`
  decision or reject later submission. Backend drain before that decision must
  not permit target unmapping. Repeat for failed and cancelled streams.
- Gather launch, descriptor construction, NIXL submission, result-send, and
  scatter exceptions.
- Cancel while gather is active but before NIXL submission; assert that source
  KV and any send-bounce resources remain leased until gather quiesces.
- Consumer callback exception during physical finalization.
- Attempted source KV reuse while direct NIXL read is blocked.
- Attempted destination KV reuse while direct write or scatter is blocked.
- Attempted send/receive bounce reuse while NIXL or scatter is active.
- Shutdown with blocked NIXL status, lost quiescence result, and queued/running
  scatter; assert that deregistration and unmapping do not occur.
- Publish a target, pause the sender before NIXL submission, and start receiver
  shutdown. Assert that a drain of currently submitted operations cannot unmap
  the target; retirement requires a terminal `NO_REMOTE_ACCESS` result or an
  acknowledged submission fence that prevents later use of the advertisement.
- Lose, delay, duplicate, reorder, contradict, and replay a submission-fence
  acknowledgement from a stale endpoint epoch. Only the matching negotiated
  acknowledgement may close future submission; mixed-version or unresolved
  peers keep shutdown non-drained.
- Exercise new/new negotiation independently for Phase 1 identity,
  creation-close, fixed-envelope, and replay-window capabilities; Phase 2
  dynamic-stream/end-of-stream capability; and Phase 3 submission fencing.
  For each phase, reject new/legacy, missing, unknown, and downgraded capability
  combinations before that phase's publication/local-launch boundary.
- Count protocol messages on healthy admission and assert generation/epoch and
  fixed-envelope fields use the existing KV request/target response with no
  standalone admission acknowledgement. Verify skip/close/reject controls occur
  only in their exceptional scenarios.
- Publish a target, then deliver a valid Phase 1 creation-close acknowledgement.
  It may close future context creation but must not retire the published target,
  satisfy a submission fence, or permit shutdown while later sender submission
  remains possible. Only terminal operation evidence or the negotiated Phase 3
  fence plus quiescence may release the target.
- Exactly-once release under concurrent cancel/result and reordered duplicates.
- Lease acquisition racing request `free_resources()`.
- Raw block-ID snapshot racing `free_resources()` and block reuse; assert that
  only the allocator-atomic allocation-generation-bearing snapshot can create a
  context.
- Overlapping slice leases on the same KV allocation generation.
- Overlap direct and bounce-path lease tokens on one allocation generation.
  Per-path token series may count both, while path-neutral unique/pending-free
  gauges count the range once; any `direct_only`/`bounce_only`/`mixed` breakdown
  must sum exactly to the path-neutral gauge.
- Malformed, overflowing, misaligned, or out-of-range NIXL and scatter metadata.
- Fixed-manifest segments that are individually in range but truncate expected
  coverage, leave a gap, duplicate/overlap within or across writers, mismatch
  aggregate bytes, or permute equal-sized source-to-target intervals while
  retaining full target coverage. Cover direct descriptors and wrong
  bounce-source offsets in scatter metadata. All fail logical success and
  suppress usable scatter while physical access still drains. Equivalent
  differently segmented inputs that normalize to the exact expected mapping
  pass.
- Fixed manifests and transfer context/slice sets at, below, and above their
  configured operation/descriptor-segment/authorized-byte/context limits.
  Concurrent admissions must atomically reserve credits without overshoot;
  rejection occurs before publication/local launch, completes in the same
  bounded worker iteration, and transactional unwind returns every credit.
- Open many dynamic streams with maximum envelopes but materialize only a small
  fraction of their operations/segments/bytes. Verify reserved/materialized
  credits, utilization/slack, reservation-caused rejection, and prompt return of
  unused credits after authenticated close establishes the exact operation set,
  without releasing materialized entries early. A future-submission-only fence
  must retain slack until ledger reconciliation or backend quiescence.
- Synchronous pre-submit failure versus ambiguous NIXL submission failure.
- Quiesced `NO_REMOTE_ACCESS` and quiesced-with-unknown-data outcomes.
- Global NIXL quiescence while scatter remains queued or active.
- KV registry drained while an independent auxiliary transfer remains active.
- Non-drained shutdown followed by a successful retry.
- Duplicate context identity insertion.
- Retry identical admission through the same token and receive the same handle;
  attempt the live identity through a different token and reject it without
  attaching cancellation authority or replacing canonical state.
- Partial KV/bounce acquisition, plan-validation failure, and duplicate registry
  insertion before the receiver target-publication or sender local-launch
  boundary; assert endpoint-local transactional unwind with no leaked lease.
  Repeat after each endpoint's boundary and assert fail-closed retention instead
  of rollback.
- Simultaneous local send/receive identities, fan-in writers from different peer
  epochs, and a stale peer epoch targeting a new local transfer generation.
- Keep the local endpoint epoch fixed while the remote admission authority
  restarts with a fresh epoch and repeats a transfer ID/generation value. The
  new full identity may be admitted, while delayed old-authority traffic is
  rejected without colliding with the new record or replay window.
- Delayed cancellation through a stale handle after request ID reuse; assert
  that it cannot change the current transfer generation.
- Remote cancellation arriving before context creation, followed by the matching
  transfer generation; assert that the transfer-generation-aware pre-cancel
  tombstone applies.
- Race remote cancellation against registry/handle/context creation; assert
  that either the live handle closes or the preparation observes the tombstone,
  with no access boundary in between. Fill the tombstone table and assert
  that every already-admitted generation has reserved cancellation state and
  new admission backpressures before overflow. Deliver a valid cancellation at
  full occupancy and verify it is recorded. Lose, delay, duplicate, and replay
  the Phase 1 creation-close acknowledgement; retire zero-context state only
  after the matching acknowledgement or local producer sealing.
- Old-transfer-generation remote cancellation arriving after request ID reuse; assert
  that neither the live context nor its session changes state.
- Logical cancellation followed immediately by request cleanup while the
  physical disposition remains `DRAINING` or `IN_DOUBT`.
- Generation-first cancellation while auxiliary RMA remains active; assert that
  KV may become pending-free but the session and auxiliary slot remain live.
- Gather/scatter worker death or asynchronous CUDA failure.
- Bounce send with NIXL blocked after gather; assert that the slice's source-KV
  lease retires while its send-bounce lease remains live.
- Direct send with NIXL definitively quiescent and result delivery blocked;
  assert that source KV retires before result delivery.
- Multi-slice send with one slice quiescent and a sibling blocked; assert that
  the first slice retires without waiting for sibling or request completion.
- Non-drained transceiver shutdown retained by PyExecutor; assert that executor
  and creator cleanup cannot shut down the KV manager until a retry drains.
- Request rollback with live direct and bounce contexts. Assert that admission
  and mode switching remain blocked and each context keeps its negotiated owner.
  Hold one context `IN_DOUBT` and require a visible blocked/non-drained rollback
  without resource release; after terminal evidence drains every context, only
  newly admitted sessions may negotiate the replacement mode.

Tests must use barriers/hooks rather than probabilistic sleeps for the critical
publication and teardown races.

### Configuration matrix

- bounce not requested, requested but factory-inactive, and arena-active;
- positive bounce-engagement evidence on candidate GB200/GB300 MNNVL fabric-VMM
  cells; the merged config-enabled GB200 test is not proof, and
  configured-but-inactive runs count as direct fallback;
- Python KV manager V2 and C++-backed V1;
- phase-matched new/new negotiation plus new/legacy, missing-capability,
  unknown-capability, and downgrade rejection;
- single writer and multi-writer TP/PP/ADP overlap or fan-in;
- direct, bounce, and per-writer fallback;
- single slice and chunked/multi-slice transfer;
- fixed manifests and dynamic streams at small, midpoint, and negotiated-maximum
  operation-count/descriptor-segment/authorized-byte budgets, including
  out-of-order and end-of-stream load, plus per-transfer context/slice limits;
- high-concurrency maximum-envelope dynamic streams with low actual
  operation/segment/byte materialization;
- empty, midpoint, near-capacity, and full endpoint-global
  context/operation/segment/authorized-byte credit occupancy;
- empty, midpoint, near-capacity, and full lifecycle-record/pre-cancel occupancy;
- sequential and out-of-order retirement at empty, midpoint, near-capacity, and
  full endpoint replay-window occupancy;
- PyTorch executor and AutoDeploy with Python transceiver;
- context-first scheduling, plus generation-first only after its independent
  auxiliary-slot cleanup gate is validated;
- bounce-ineligible/non-attention layer-group fallback;
- normal completion, cancellation, timeout, peer failure, and shutdown.

### Performance validation

Use an A/B build or runtime gate on the same commit and compare against the
merged PR #15618 behavior. Hold model, hardware, NIXL backend, request trace,
arena size, and scheduler configuration constant. Cover direct, bounce, and
fallback paths across representative transfer sizes, concurrency, fan-in, both
KV managers, and PyTorch/AutoDeploy cells. Run warmup plus enough independent
benchmark runs to report 95% confidence intervals, using runs rather than
individual requests as the sampling unit. Freeze workload definitions and gates
before examining implementation results. Record the low-cardinality ownership
mode, protocol bucket, rollout cohort, KV manager, executor, transport-active
state, and observed writer path for every aggregate. Reclassify an intended
bounce cell that lacks positive engagement evidence as direct fallback and do
not use it to approve bounce rollout.

Report:

- request throughput and completed-request goodput;
- p50/p95/p99 TTFT and end-to-end transfer-task latency;
- gather, NIXL, and scatter latency/overlap, with the existing raw NIXL timer
  reported separately;
- host CPU time per completed transfer and registry/lifecycle-lock contention;
- per-transfer healthy admission control message count plus exceptional
  skip/close/reject RTT and retry count;
- active host metadata by context/stream/tombstone/replay-window occupancy,
  retired-floor advancement and gap age, end-of-stream processing time,
  reserved/materialized credits, reservation utilization/slack and caused
  rejection, admission/rejection latency, per-path lease-token bytes,
  path-neutral unique transfer-leased bytes, pending-free KV bytes blocked
  solely by transfer leases, and each corresponding byte-seconds/high-water
  mark;
- post-logical retention, per-resource early-release lead, lease duration, and
  capacity-recovery time;
- admission failures and goodput during known-terminal and deliberately
  ambiguous fault injection;
- shutdown drain latency and non-drained bytes.

Initial go/no-go thresholds are scoped by scenario.

Healthy/no-fault matrix cells:

- throughput/goodput 95% confidence-interval lower bound is no worse than -2%;
- the upper 95% confidence bound of relative p50/p95 TTFT and end-to-end transfer
  latency regression is at most 3%, and the p99 bound is at most 5%;
- the upper 95% confidence bound of host CPU time per completed transfer
  regression is at most 5%;
- the direct path adds zero CUDA copies, kernels, or synchronizations;
- ownership adds no device arena beyond an active PR #15618 send/receive bounce
  arena pair;
- host metadata has frozen measured per-entry budgets for contexts, writer
  operations, descriptor/scatter segments, lifecycle records, and
  live/sparse-retired generation entries; repeated waves, replay gaps, and floor
  compaction stay within the resulting O(contexts + writer operations +
  segments + lifecycle records + outstanding generations) budget;
- across predeclared small/midpoint/maximum fixed-manifest, context-set,
  dynamic-stream, lifecycle-record, and replay-window sizes—including
  out-of-order gaps and floor compaction—host bytes and CPU/close processing fit
  the frozen per-entry complexity budget with no statistically significant
  unexplained superlinear term;
- at each declared supported concurrency, maximum-envelope/low-materialization
  streams have zero reservation-caused rejection; otherwise that envelope or
  concurrency is excluded from the rollout cohort. Report utilization/slack and
  goodput at and beyond the declared boundary;
- before the configured backpressure threshold, the upper 95% confidence bound
  for p95 admission-latency regression at near-capacity lifecycle-record and
  replay-window/metadata-credit occupancy is at most 5%; at capacity, rejection
  performs no sleep/retry, completes in the same admission-worker iteration and
  within one configured bounded-poll budget, and does not overshoot configured
  record/replay/context/operation/segment/byte limits. Report p95/p99 wall time and
  freeze an absolute per-cohort ceiling before implementation results are seen.

Known-terminal fault cells:

- after each fixed-concurrency fault wave drains, active context/lease counts and
  retained bytes return to their pre-wave baseline with no monotonic growth;
- once the final required terminal/completion evidence is observed, leases are
  released in the same state-machine call or the next completion-worker
  iteration, as applicable;
- delayed-result tests recover all leased capacity.

Ambiguous-result fault cells:

- exactly the affected bytes remain retained and observable, with no unsafe
  reuse, unmap, or cross-request corruption;
- fault-period goodput and latency are reported, but the healthy-path
  non-regression thresholds do not apply because safe capacity exhaustion is an
  expected outcome.

If stable benchmark noise is wider than a threshold, calibrate and document a
replacement before implementation measurements are used for approval. Do not
relax a gate after seeing a regression without a separate review.

### Acceptance criteria

The design is implemented only when all of the following are true:

1. No address can be published without a registered context and live leases.
2. Session/request removal cannot drop a physical quiescence result.
3. KV and bounce allocators cannot reuse leased allocation generations.
4. Every release path is idempotent and tied to quiescence evidence.
5. Direct, bounce, and mixed transfers share the same ownership path.
6. Timers alone never authorize reuse.
7. Shutdown cannot deregister or unmap memory while a context may access it.
8. New identity fields are negotiated or mixed versions are rejected before
   DMA addresses are exchanged.
9. The executor remains responsive while contexts drain.
10. In-doubt capacity and shutdown state are observable.
11. Lease acquisition and `free_resources()` cannot race into block reuse.
12. Every NIXL/gather/scatter range is authorized by a live allocation-generation
    lease.
13. Every local CUDA/NIXL accessor is independently tracked through quiescence.
14. Phase 1 negotiates and validates one monotonic, attempt-wide transfer
    generation plus both endpoint epochs on every lifecycle-mutating wire
    message; legacy identity is rejected before address publication.
15. Contexts are built only from allocator-atomic,
    allocation-generation-bearing KV lease snapshots; copied block IDs are
    never used to acquire a later lease.
16. Local role/epoch plus admission-authority epoch identifies the context,
    while exact peer epoch/rank/operation identifies each fan-in ledger entry.
17. Logical cancellation and dropping KV request ownership do not wait for KV
    physical retirement, and pending-free KV cannot be reused while its lease
    remains live.
18. A non-drained shutdown result keeps the transceiver and KV managers alive and
    can be retried without destructor fallback cleanup.
19. The applicable quantitative normal-path and fault-period performance gates
    pass with the required stage and end-to-end metrics.
20. Unsupported topology or mixed-version cohorts are rejected before address
    publication. Rollback never changes an active transfer's owner and reports a
    visible blocked/non-drained state until every old-mode context retires.
21. Preparation failure unwinds every partial lease before that endpoint's
    publication/launch boundary, while failure after its boundary follows
    fail-closed retirement.
22. Cancellation is routed through a transfer-generation-bound handle; a bare
    or reused request/transfer ID cannot mutate lifecycle state.
23. KV cleanup never releases an independently active auxiliary slot; unsupported
    generation-first configurations are rejected before publication.
24. Bounce source KV retires after gather, direct source KV retires after NIXL
    quiescence, and one slice never waits for unrelated sibling completion.
25. Every remote lifecycle-control message is transfer-generation/epoch
    validated before session lookup; pre-cancel tombstones cannot cross transfer
    generations, and an arbitrary unnegotiated control closes the endpoint
    session without allocating per-identity state.
26. First failure, accepted cancellation, or shutdown atomically aborts context
    admission and every later publication/local-launch boundary. Sealing is
    performed only by the local producer/session with a complete immutable
    manifest or atomically enrolled final-chunk proof.
27. Immediate retirement requires every remote exposure and every local
    gather/NIXL/scatter accessor to be terminal; proof that no remote operation
    was submitted is insufficient while a local gather remains active.
28. Shutdown fences future submission against every published address before it
    treats backend quiescence as sufficient to deregister or unmap memory.
29. Every target advertisement identifies its target slice and exact fixed
    operation or bounded writer stream. Duplicate advertisements are
    idempotent. End-of-stream closes future submission only after every covered
    sequence is submitted or terminal `NO_REMOTE_ACCESS`; dynamic streams cannot
    retire until that close/fencing plus operation quiescence through the
    high-water mark, and logical success additionally requires the exact
    normalized required source-to-target mapping.
30. Pre-cancel lookup and context creation serialize in the registry. Capacity
    is reserved at admission for every valid cancellation; unmatched state
    remains until producer sealing or the Phase 1 creation-close acknowledgement
    and uses pre-admission backpressure rather than unsafe eviction.
31. Bounce correctness/performance cells record positive transport-active and
    per-transfer engagement evidence; `kv_cache_bounce_size_mb > 0` alone does
    not count as bounce coverage.
32. The registry strongly retains exactly one canonical endpoint-transfer
    record, handle, and gate after consumer cleanup and until no later local
    context can be created and all physical obligations retire; it atomically
    marks the generation retired before removing that record.
33. Endpoint-epoch replay state rejects delayed admission/advertisement after
    record deletion, bounds out-of-order generations with a retired floor and
    sparse window, backpressures gaps without eviction or admission-worker
    retry/sleep, and requires a fresh non-reused endpoint epoch when that state
    is recreated.
34. Fixed manifests, dynamic streams, and context/slice sets reserve negotiated
    operation/segment/byte metadata credits against endpoint-global limits
    before exposure; overload fails without overshoot or retry/sleep in the same
    bounded admission-worker iteration, and retirement returns credits exactly
    once.
35. Capacity telemetry distinguishes overlapping lease-token bytes, unique
    transfer-leased allocation-generation bytes, and pending-free KV bytes
    blocked solely by transfer leases; KV recovery/admission claims use the last
    measure, while bounce claims use unique leased slot bytes. Deduplicated
    gauges are path-neutral or use only mutually exclusive direct/bounce/mixed
    classes; per-path token metrics are not summed as unique capacity.
36. Local cancellation and initial admission serialize through a one-shot token:
    cancellation either prevents transfer eligibility/token creation, rejects
    pending admission before peer authorization, or aborts the canonical bound
    handle, with no lost-cancel interval.
37. Phase-specific new/new negotiation succeeds, while missing/downgraded/legacy
    capabilities fail before the relevant access boundary; a Phase 1 creation
    close never substitutes for a Phase 3 submission fence after publication.
38. Dynamic-stream telemetry distinguishes reserved from materialized
    operation/segment/byte credits, exposes slack and reservation-caused
    rejection, and proves declared rollout concurrency without slack-induced
    rejection; authenticated close returns only credits that cannot materialize.
39. Shutdown atomically closes registry admission and snapshots committed
    records; a racing pending token is shutdown-aborted or its canonical record
    and any possible peer authorization are included in drain, never omitted.
40. Fixed-mode success requires the exact normalized per-operation and manifest
    source-to-target KV mapping; in-range gaps, undercoverage,
    duplicate/overlap, aggregate mismatch, and full-coverage source permutation
    fail rather than producing corrupted KV.
41. Initiator pre-commit rejection consumes no transfer generation. Every
    committed generation whose admission control is not sent is bilaterally
    acknowledged as skipped before a higher admission control is sent. A
    generation rejected after admission control but before address publication
    is creation-closed; after publication it follows full physical retirement.
    Repeated admission failure cannot create artificial peer replay gaps.
42. Session negotiation is amortized and healthy admission fields piggyback on
    the existing KV request/target response; no new normal-path per-transfer RTT
    is introduced, or the cohort is separately budgeted and qualified.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Lock-order deadlock across admission token, registry, session, allocator, CUDA, and callbacks | Token-to-registry-to-handle-to-context order; no blocking operations or callbacks under lifecycle locks; lock-order tests. |
| Local cancellation races initial handle binding | One-shot admission token with token-to-registry lock order; bind the canonical handle before peer authorization; deterministic race tests. |
| Shutdown races pending admission or peer authorization | Close admission and snapshot under one registry lock; shutdown-abort losing tokens; gate authorization through the bound handle; deterministic boundary races. |
| KV over-pinning or leaked leases | Exactly-once release, active-lease metrics, oldest-context diagnostics, and fault-injection coverage. |
| Capacity exhaustion from indefinite retention | Fail admission visibly; expose bytes/age/reason; add reliable result recovery or backend quiescence before bounded reclaim. |
| Treating failure as quiescence incorrectly | Track quiescence separately from data outcome and preserve an explicit in-doubt state. |
| Mixed direct/bounce accounting error | Record actual mode per exact writer and retain both candidate leases until mode is known. |
| Individually valid fixed segments leave incomplete, overlapping, or permuted KV | Manifest the exact logical source-to-target mapping, normalize delivered pairs, reject gaps/duplicates/overlap/aggregate mismatch/permutation, and test direct plus bounce pairings. |
| Dynamic stream closes early, undercovers or permutes the target, or grows without bound | Authenticate end-of-stream, require irreversible submission decisions before close, enforce sequence/operation/segment/byte budgets, validate the exact required source-to-target mapping independent of the claimed high-water mark, and retain the target while the stream is open. |
| Stale result advances a new transfer | Endpoint epochs, monotonic generation, durable retired floor/replay window, exact writer validation, and negotiated protocol. |
| Retired transfer is recreated by delayed admission | Endpoint-owned retired floor plus bounded sparse replay window; mark retirement before record deletion; reject old epochs before session creation. |
| Replay-window gap exhausts host capacity | Bound outstanding generations, expose floor/gap/window metrics, and backpressure admission without evicting anti-replay state. |
| Failed admission burns generations and fills peer replay capacity | Allocate the generation only in the successful local admission commit; acknowledge a skip before sending a higher admission control; creation-close rejection only before address publication; use full retirement afterward; repeat/lost-ack tests. |
| Admission protocol adds a healthy-path RTT | Piggyback fields on the existing KV request/target response, count normal/exceptional controls, and separately budget any cohort that cannot preserve the exchange. |
| Fixed manifests, descriptor plans, or context fan-out exhaust host metadata | Negotiate per-transfer operation/segment/byte/context envelopes, reserve endpoint-global credits before address publication, reject oversize/over-capacity plans, and test concurrent no-overshoot accounting. |
| Worst-case stream reservations suppress useful concurrency | Export reserved/materialized credits and slack, benchmark maximum-envelope/low-materialization concurrency, tune or gate supported envelopes, and return unused credits only after authenticated close. |
| Pre-cancel state exhausts host capacity | Reserve a lifecycle/control slot per admitted generation, export configured/used/remaining capacity, backpressure before admission, and retire only after producer sealing or Phase 1 creation close. |
| Request cleanup drops the access gate | Registry-own one canonical record/handle/gate through no-future-context proof and physical retirement; contexts retain only the shared gate, which owns no contexts. |
| First failure races another chunk | Atomically abort the shared gate before callbacks and reject later enrollment/access while already-possible work drains. |
| Main-loop latency regression | Preserve bounded polling; perform draining on transfer/completion workers; benchmark registry contention. |
| Shutdown hangs indefinitely | Use an explicit deadline and return non-drained status, but never convert deadline expiry into unsafe unmap/reuse. |
| Rolling-upgrade incompatibility | Capability/version negotiation or pre-DMA rejection; no silent downgrade of safety fields. |
| Creation-close acknowledgement is mistaken for a submission fence | Distinct message/capability types and predicates; after-publication tests prove Phase 1 close cannot release a target or authorize teardown. |
| Behavioral divergence between KV managers | One conformance test suite for the common lease API; implementation-specific adapters only below it. |
| Scope expands into cancellation consensus | Keep logical consensus in the adjacent cancellation design; consume its outcome through `TransferHandle`. |
| Metrics add hot-path overhead | Use O(1) counters/histograms, batch export, and benchmark instrumentation-on overhead; never scan the full registry per transfer. |
| Bounce configuration silently exercises direct fallback | Require transport-active and per-transfer path signals in tests, benchmarks, and canary telemetry. |
| A caller drops a non-drained worker or shuts down its KV manager | Structured `ShutdownResult`, strong caller retention, retry tests, and a hard teardown precondition on zero live leases. |
| Rollback bypasses an active owner | Negotiate mode before publication, stop admission, and drain existing contexts before selecting a mode for new sessions. |

## Alternatives Considered

### Expand the PR #15618 bounce `TransferContext`

Rejected as the general architecture. The initial containment has since renamed
that object to `RecvBounceContext`, but it remains naturally scoped to a receive
bounce slot. Making it own direct writers, source KV, sender status, request
aggregation, and KV-manager leases would invert module boundaries and make
ownership disappear when bounce is disabled.

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
3. Which endpoint-owned counter/state implementation provides monotonic transfer
   generation allocation and the bounded replay window across session/worker
   recreation, and what maximum outstanding-generation window supports expected
   concurrency without excessive reservation?
4. What allocates a unique endpoint restart epoch, keeps it stable for one
   endpoint lifetime, and prevents its reuse after rank/process restart?
5. Can existing V1 block pinning enforce leases through explicit
   `free_resources()`, or is a new nanobind API required?
6. What is the smallest V2 KV-manager lease surface that supports per-slice early
   release without request-wide over-pinning?
7. Does the NIXL wrapper need a separate `quiesce()` operation before final
   shutdown so registrations can remain valid during drain?
8. Which peer/session/endpoint primitive can acknowledge a submission fence and
   guarantee that an old target advertisement cannot be used afterward?
9. Is result acknowledgement/retry required for the first production rollout,
   or is observable indefinite retention acceptable for all Python-native
   direct and bounce transfers?
10. Which independent auxiliary-buffer paths can outlive their request/session?
   They need a follow-up audit before the broader transceiver can claim complete
   ownership coverage.
11. Can every claimed TP/PP/ADP topology identify its exact potential writer cohort
    before address fan-out, or which cells must be gated initially?
12. Which existing serving metrics surface should export lifecycle histograms and
    retained-capacity gauges without adding per-transfer registry scans?
13. How should the internal rollout gate and negotiated capability be represented
    without making the safety contract depend on the bounce-size option?
14. Which `SliceSetProof` form does each endpoint use: an immutable manifest or
    a final-chunk proof atomically enrolled with the final context? For PR
    #15727-style scheduling, how is the final marker authenticated and tied to
    the sender's operation-sequence high-water mark?
15. What conservative operation-count, descriptor-segment, and byte budget can
    be authorized before opening a dynamic writer stream without constraining
    valid chunk schedules?
16. Which non-GB200/GB300 environments can positively validate an active
    fabric-VMM bounce arena rather than only the direct fallback?
17. Which Phase 1 control/acknowledgement closes future context creation for a
    transfer generation, and how is it retried independently from the stronger
    Phase 3 fence that closes future submission to published addresses?
18. What fixed-manifest operation/descriptor-segment/authorized-byte,
    per-transfer context/slice, and endpoint-global metadata-credit limits cover
    production fan-out without allowing one transfer to monopolize host
    capacity?
19. What canonical logical KV coordinate encodes source-to-target mappings across
    TP/PP/ADP, layer groups, and both KV managers so validation never depends on
    physical descriptor order?

No answer to these questions may weaken the invariant by interpreting
uncertainty as retirement.
