# DWDP — who owns what

| layer | owns |
|---|---|
| **runtime** | the MPI sub-communicator · the authoritative per-rank expert range · MNNVL handle exchange · the composite virtual-address layout that makes the full expert table addressable · the P2P copy stream and its events · the double-buffered prefetch state machine |
| **adapter** (`dwdp_adapter.py`, class `DwdpAdapter`) | the boundary: the local expert window, the facade the runtime's setup traversal walks, and the two forward hooks |
| **modeling code** | checkpoint conversion · the router · the shared expert · quantization and the fused expert-runner call · the forward |

The target copies none of the runtime's list. In particular the adapter
**creates no stream, issues no copy, and does not reach into the weight
manager** — it exposes the boundary and relays two calls.

## Two consequences for the forward

**It cannot be CUDA-graph captured.** The prefetch state machine is driven
from the host between layers, and a captured graph has no host between layers.
This is a property of the feature; it belongs in the run journal beside the
identity record, because every perf number taken on this target is a
non-captured number and a reader who assumes otherwise will misattribute the
gap.

**Bind and prefetch bracket the MoE call.** Bind before it, so the expert
weights the router selected are addressable; prefetch the next layer after it,
so the copy overlaps compute rather than stalling it. Both orderings are load
bearing and neither raises when wrong — bind late and you address the previous
layer's window; prefetch early and you evict what the current call still
needs.

## Why the digests

The four upstream files the adapter binds against are **private runtime
surface, not public API**. There is no version contract on them, and there is
no version pin left to notice a drift either, so their SHA-256 goes in the run
journal: a change to any of them re-opens every question the adapter answers,
and the digest is what turns that from a silent drift into a visible diff.
