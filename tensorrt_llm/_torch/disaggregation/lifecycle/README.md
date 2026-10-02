# Shared lifecycle enforcement

This internal package reuses the native Python transfer lifecycle implementation.
It does not introduce another KV owner, allocator, public backend API, or supported
configuration. Native sessions still select writers, order publication, submit
transfers, translate backend evidence, and synchronize local CUDA work.

## Binding responsibilities

| Mechanism | Shared implementation | Integration responsibility |
| --- | --- | --- |
| Source access | `SendOperationOwner` retains the exact request/status across ambiguous completion. | Map participants and root allocation/registration loans before submission; supply backend-defined completion. |
| Destination access | `ReceiveOperationOwner` accounts for the sealed writer cohort and local completion. | Select the cohort, serialize publication against cancellation, and retain destination storage. A receive owner tracks evidence, not the allocation itself. |
| Deadline arbitration | `RetirementDeadline` serializes exposure, evidence, request timeout, and grace expiry. | Bind all pieces of one session to the same controller; seal completion only after all expected pieces, including AUX, are accounted for. |
| Containment | `RetirementWatchdog` closes admission and notifies containment once. | Keep sessions, manager loans, pools, and agents reachable; provide the qualified executor containment callback. |
| Logical outcome | Remains with the integration, outside this package. | Commit failure/cancellation once; bind a metadata-only timeout callback. Late physical completion must not replace that outcome. |

Only `NOT_SUBMITTED` and `BACKEND_DONE` establish source access-end. The other
states remain `ADMITTED`, `SUBMITTING`, `SUBMITTED`, and `IN_DOUBT`. A backend
error, elapsed time, logical failure, or empty report count is not quiescence.
Retirement at or after fatal expiry remains prohibited even if DONE arrives later.

The integration must couple evidence and strong resource roots. Merely constructing
an operation owner does not pin KV Manager pages, hold a registration, or acquire a
manager reference. `expose(*owners)` can retain multiple opaque claims atomically;
each claim requires its own access-end evidence before `settle()`. In particular,
`resources_drained` includes the shared session predicate, so a separately exposed
loan cannot wait on that predicate to settle itself. Release/refcount operations
remain with the resource owner and must be idempotent.

Native compatibility imports refer to these same classes; native tasks delegate
source bookkeeping without changing wire messages, activation, or task outcomes.
The module-level binding tests exercise opaque resource holds and completion
probes, not a production runtime adapter or a qualified shared-transfer profile.
The runtime integration must still agree its participant mapping, session boundary,
resource-root binding, backend evidence, and containment callback before adoption.
