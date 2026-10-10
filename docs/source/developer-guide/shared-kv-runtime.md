<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared KV runtime contracts

The shared backend contract lives in
`tensorrt_llm._torch.disaggregation.base.shared`. Its symbols describe opaque
content units, logical outcomes, cancellation requests, backend access
completion, routes and memory registration. The existing paired transfer contract
remains in `base.backend`; paired-path convergence is a separate runtime milestone.
Import from the intended module explicitly because several type names overlap.

## Compatibility boundary

`resource.shared.SharedRuntimeProfile` records explicit runtime assembly facts.
Its contract revision is pinned to
`147ed68276e7fb89d5de60002d4e793a78707a8c+cancel-request-v1`. The suffix identifies
the local cancellation extension to the base contract, not an upstream spec
revision. Its backend revision identifies the actual package or build under test
and is a separate value. Validate the profile before taking a staging hold,
registering memory or exposing an extent.

The first supported shape is native NIXL, `KVCacheManagerV2`, BF16 MHA written by
TRTLLM in HND layout, TP=DP=PP=CP=1, manager-owned host staging and committed whole
blocks. Remapping, offload, compression, recurrent state, extra buffer roles and
retry remain outside that shape. An unknown extra feature is rejected as well.
These checks describe compatibility requirements; they neither enable runtime
integration nor establish deployment qualification. Exact adapter qualification
belongs to LC-MC-15 for the native profile and LC-MC-17 for the separate
Mooncake profile; runtime qualification belongs to RI-09. Mooncake remains
unsupported until its RI-08D integration and profile qualification are complete.

The assembly caller must obtain the cache dtype, writer/layout and feature facts
from the actual engine configuration. A lender layout digest alone does not prove
compatibility: dtype and computational meaning belong to the reuse scope, and
the writer must really use the declared byte layout. No new capabilities method
is added to the shared backend API.

## Content and physical identity

Use the public `pyexecutor.kv_cache.sharing` lender types. `build_extent` copies
`GroupRun.names[i].tobytes()` unchanged into `Unit.name`; it never reconstructs a
content key. A unit's `local_group` is the lender's layer group and its `local`
coordinate is `(address - part.address) // part.slot_bytes`. Logical block
ordinals are not local storage slots.

`STAGING_EXTENT_NAMESPACE` is a stable versioned adapter convention shared across
requests. Do not substitute request IDs or allocation identities. A `Part.name`
identifies a layout-compatible region, so matching names across workers do not
identify the same physical memory. Adapter, lease, allocation and registration
instances retain their distinct local ownership identities.

## Delivery and readiness

`Delivered.served` contains complete units only; an empty set is a miss.
`served_masks` rejects unknown names and maps the delivered subset into the
lender's row masks. It does not compute engine readiness or fill holes between
delivered units. On the manager's owner thread, call `Lease.mark_arrived` once,
then use `StagingLender.readiness` after the manager's local copy processing.
Across ranks, use the minimum `usable_until` and maximum `restart_floor`.

Logical completion and backend access completion are independent. A failed or
cancelled outcome, request exit, and a timeout cannot authorize `Lease.release`.
Keep leases and registration roots until physical access and local copies have
ended; only then deregister memory and release the `PartsHold` on the manager's
owner thread. `release` asserts that access ended; it does not stop a transfer.

## Staging lifetime binding

`resource.shared_lifetime.SharedStagingAdapter` binds an existing manager and its
staging lender to one registered provider. Construction validates the profile and
takes a `PartsHold`; the caller retains the adapter before calling `register`.
Pool registration happens once for this physical allocation, independently of
request lifetime. All adapter, lender and lease calls run on the manager thread.

Submit only a ready lease: a read lease's `poll` must have completed its device-to-
host copy before publication. Each operation takes a fresh `RetirementDeadline`
from the same watchdog as the adapter's separate registration-teardown owner.
The watchdog must run independently with the executor's containment callback.
This first binding supports one operation per deadline; multi-piece sessions,
retry and paired-path convergence require their later integration work.

The adapter roots the lease before submission. `SubmissionRejected` proves that
nothing escaped; an unexpected submission exception retains the loan because
there is no handle with which to establish access-end. Logical outcomes are
latched independently from background `quiesce` calls. A false or exceptional
quiescence result starts drain and needs explicit evidence retry; it never
authorizes reuse. Blocking backend waits do not run on the manager or watchdog
thread.

After positive backend quiescence, a write operation marks only the served whole
rows. Pass `cuda_copy_completion` the exact manager execution stream: its event
is recorded immediately after `mark_arrived` and its query remains valid after
request exit. Only backend access-end, local-copy completion and successful
lifecycle arbitration together permit lease release. Neither this event nor a
backend outcome replaces `StagingLender.readiness` for engine scheduling.

`close` stops admission and progresses retained operations without waiting on
backend threads. Once they drain, it closes registrations and then releases the
parts hold. Failed registration closure retains the same handle for retry;
unproven access, copy errors or fatal expiry retain roots for containment.
The adapter does not activate a scheduler, route selection or native transport.

## Cancellation requests

Call `request_cancel(attempt)` for any shared `Attempt`. An ordinary attempt
returns `CancelDisposition.UNSUPPORTED`; providers can implement the optional
`CancellableAttempt` protocol. Its nonblocking, idempotent method suppresses
queued work where feasible and requests SDK cancellation where supported.
Potentially blocking SDK work must run outside this call.

`REQUESTED` acknowledges a best-effort request. It does not guarantee a
`Cancelled` outcome or prove that memory access ended. Completion can win a
race with cancellation, and the first terminal outcome stays unchanged.
Cancellation cannot roll back already published content. Repeated calls must
not start duplicate cancellation work.

`SharedStagingAdapter.cancel` first commits its logical cancellation under the
lifecycle arbiter. It then invokes the shared helper once, outside that lock,
without waiting for SDK completion. The first locally committed outcome wins;
backend completion that the adapter has not observed cannot undo local
cancellation. An already committed `Delivered` outcome remains unchanged.

The watchdog timeout callback updates outcome metadata only. Owner-thread
`progress` forwards cancellation for a latched `Failed` or `Cancelled` outcome.
Repeated cancellation, progress and shutdown calls do not dispatch additional
requests. `SharedLeaseOperation.cancel_disposition` records the acknowledgement;
`cancel_error` records a provider exception or invalid acknowledgement. A failed
request does not change the outcome or stop physical evidence collection.

Neither disposition resets retirement deadlines or permits early lease,
registration, or staging release. A request must not start caller-memory access
or invalidate prior quiescence. Backend quiescence and local-copy completion
still govern physical retirement. Store providers keep queue and per-row SDK
details private behind this shared capability.
