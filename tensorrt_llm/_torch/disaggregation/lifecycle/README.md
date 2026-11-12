<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# Shared deadline and retirement arbitration

This internal package shares only deadline and retirement arbitration. Physical
operation ownership, the six sender states, backend-status polling, receive
cohorts, and `TransferNotSubmittedError` remain in `native/ownership.py`. Native
sessions still order publication, submit transfers, and translate protocol
evidence. This package introduces no KV owner, allocator, public backend API, or
supported configuration.

## Binding responsibilities

| Mechanism | Shared implementation | Integration responsibility |
| --- | --- | --- |
| Deadline arbitration | `RetirementDeadline` serializes exposure, evidence, transfer safety timeout, and grace expiry. | Bind all pieces of one session to the same controller; seal completion only after all expected pieces, including AUX, are accounted for. |
| Native containment | `RetirementWatchdog` latches a controller-wide fatal state, closes admission, and notifies containment once. | Preserve native's executor-fatal callback and keep sessions, manager loans, pools, and agents reachable. |
| Logical outcome | Remains with the integration, outside this package. | Commit failure/cancellation once; bind a metadata-only timeout callback. Late physical completion must not replace that outcome. |

`timeout_s` is the mandatory transfer safety budget from first exposure, not the
overall inference-request deadline. The optional `grace_s` override controls the
non-resettable drain window; omitting it or passing `None` uses `timeout_s`, preserving
native timing. Both effective durations must be finite and positive. Logical delivery
does not stop the safety clock while physical claims remain; externally managed
deadlines require a separate binding contract and are not supported by this API.

Only the backend settles its physical operations and performs required peer
notifications, including native `FAILED_QUIESCED`. For the planned shared-runtime
binding, the holder reads `Attempt.quiesced()` for access-end evidence and
`Attempt.poll()` separately for the logical outcome. It must not poll raw transport
status or drive native ownership transitions. These are binding requirements, not
new APIs introduced by this package.

The arbiter neither polls a backend nor determines physical settlement. It retains
opaque claims; `RetirementDeadline.settle()` grants retirement permission after
safe evidence, but does not establish quiescence or complete backend I/O.
A backend error, elapsed time, logical failure, or empty report count is not
quiescence. The existing native fatal policy is unchanged: retirement at or after
fatal expiry remains prohibited even if DONE arrives later.

Future bindings may use qualified lease isolation or binding/domain quarantine
instead of executor termination. The scope depends on the proven access boundary,
not host versus GPU memory alone. For example, host staging can retain affected
resources while independent work continues only when access is confined, associated
registrations and local copies remain protected, and retained resources stay bounded.
In these future nonfatal scopes, late `Attempt.quiesced()` evidence must permit
idempotent lease return once the lease's access claims are settled. Lease return
must not rewrite the committed logical outcome, reopen quarantined admission, or
bypass the lender's remaining local-copy and reuse checks.

In these qualified nonfatal scopes, a capacity-only admission pause may resume when
safe lease returns restore the holder's admission budget. A separate quarantine,
shutdown, or fatal gate must still block admission. Restored capacity does not reset
deadlines or revive failed or cancelled requests.

Only bindings that accept executor-fatal containment may register claims with the
current watchdog. A binding that must keep the executor running must keep its claims
off this watchdog. Its holder must separately qualify lease isolation, retention
limits, and evidence-driven late cleanup under the fixed deadline/grace policy;
keeping claims off this watchdog does not remove those safety obligations.
This extraction neither implements nor qualifies that nonfatal path. Changing only
the callback leaves the controller-wide fatal gate in place and cannot enable
continued admission or late retirement.

`RetirementWatchdog.has_unclosed_owners` is a non-raising snapshot of unclosed owners
or fatal retention, not access-end evidence or teardown permission. It does not close
admission or stop progress. Native `require_retired()` and `stop()` remain strict guards;
the query cannot replace their locked checks. Future shared shutdown orchestration may
clean up independent resources while retaining affected storage, registrations, owner
roots, and required progress; scoped shutdown is not implemented here.

The integration must couple evidence and strong resource roots. Merely constructing
an arbiter does not pin KV Manager pages, hold a registration, or acquire a
manager reference. `expose(*owners)` can retain multiple opaque claims atomically;
each claim requires its own access-end evidence before `settle()`. In particular,
native `resources_drained` includes the session predicate, so a separately exposed
loan cannot wait on that predicate to settle itself. Release/refcount operations
remain with the resource owner and must be idempotent.

Native retirement compatibility imports refer to these same arbitration classes;
native operation bookkeeping does not change wire messages, activation, or task
outcomes. The module-level tests cover opaque-claim arbitration and native owners
bound to it, not a production runtime adapter or a qualified shared-transfer profile.
The runtime integration must still agree its participant mapping, session boundary,
resource-root binding, backend evidence, and containment callback before adoption.
