---
name: staircase-modeling-dwdp
description: Load when a MoE target must hold only part of the expert table locally and let the runtime bind the rest over NVLink — distributed weight data parallelism. An adapter-class feature whose forward cannot be CUDA-graph captured, and whose launch has three preconditions the runtime does not assert.
---

# staircase-modeling-dwdp

`--features dwdp`. Each rank loads a **fraction** of the routed experts; the
runtime binds the full table per layer over MNNVL, so the forward sees all of
them. On a 256-expert checkpoint over a group of four, each rank holds 64.

An adapter-class feature — read
`staircase-modeling-core/references/adapter-rules.md` first.

| reference | read it when |
|---|---|
| `references/runtime-boundary.md` | always — the ownership split, and the constraints it imposes on the forward |
| `references/adapter-interface.md` | writing `dwdp_adapter.py`, or placing the bind and prefetch calls |

## What this feature changes

**The manifest loads a window.** Only this rank's expert range is materialized;
the rest of the checkpoint's expert keys go into the known-not-loaded set with
DWDP as the recorded reason.

**The forward binds before the MoE call and prefetches the next layer after
it.** Two hooks, and their placement is the feature: bind too late and the
expert weights are not addressable; prefetch too early and you evict what the
current layer still needs.

**The forward is not CUDA-graph capturable.** The runtime's prefetch is a
double-buffered state machine driven from the host between layers, which a
captured graph cannot express. This is a property of the feature, not a
tuning choice — it goes in the run journal beside the identity record, because
a reader who assumes capture will misread every subsequent perf number.

## Three boot requirements that are not asserted anywhere

A DWDP launch fails by hanging, not by raising, so these are worth checking
before concluding anything about the modeling code:

- **the single-process worker flag** must be set for the TP1 ranks
- **the re-executed interpreter must be the one the run declares.** The disagg
  leader re-execs a literal `python3`; if that resolves to a different
  interpreter, the workers come up without trtllm and the launch stalls with
  no message
- **the cache transceiver must be usable.** The MPI transceiver crashes or
  stalls; UCX works but needs `libzmq` loadable and its transports pinned to
  same-node

Nothing in tree asserts any of them. The **boot gate** for this target should —
it is the minutes-scale gate, and turning a silent hang into a named failure is
exactly what it is for.

## Gates

boot with the feature's arguments → accuracy, both with
`TRTLLM_MODELING_V2=require` exported before the ranks start. **No third
gate.**

Accuracy sees this feature: each rank holds a fraction of the expert table, so
a binding that does not deliver the remote experts cannot silently produce the
same answer — either the forward fails, or it computes over a different expert
set and the score moves.

What the gates do *not* tell you is whether the remote path is carrying
traffic efficiently, or at all beyond correctness. That is a perf question and
it belongs to `/tuning`.

## Report additions

- the group size, the local expert count and the per-rank window
- confirmation that the non-DWDP path is unchanged
- which of the three boot requirements the boot gate now asserts
- the digests of the upstream files the adapter binds against — this boundary
  is private runtime surface rather than public API, so the digest is what
  makes an upstream change visible instead of silent
