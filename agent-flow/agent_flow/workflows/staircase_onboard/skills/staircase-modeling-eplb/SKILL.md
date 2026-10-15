---
name: staircase-modeling-eplb
description: Load when adding dynamic expert-parallel load balancing to a MoE target — the runtime owns placement and migration, the target owns slot-addressed weights and six per-layer forward calls. Gate the static variant before the dynamic one; whether the balancer actually balances is a tuning question, not a gate.
---

# staircase-modeling-eplb

`--features eplb`. An adapter-class feature: the balancer is a runtime-owned
service with its own threads and its own state machine, so the target binds to
it through one file rather than absorbing it.

Read `staircase-modeling-core/references/adapter-rules.md` first — the
one-import-site rule, the no-work-of-its-own rule and digest pinning all apply
here.

| reference | read it when |
|---|---|
| `references/runtime-boundary.md` | always — which layer owns what |
| `references/adapter-interface.md` | writing `eplb_adapter.py`, or placing the six calls |
| `references/placement.md` | always — slots vs experts, the `layer_updates_per_iter` knob, **and the routing trap** |
| `references/weight-migration.md` | **always** — this is where the silent wrong answer lives |

## What this feature changes

**Weights become slot-addressed.** Every per-expert tensor is indexed
`[local_slots]`, and which expert occupies a slot is the balancer's to change
mid-run. This is the change that reaches furthest — it is in `weights.py`, in
the manifest, and in every place the forward touches an expert tensor.

**The forward gains six per-layer calls** around the MoE block, bracketing the
window in which that layer's weights may be migrating.

**Routing goes through the balancer.** `route()` maps token→slot, not
token→expert, and everything downstream addresses slots.

## Two things that will bite

Both are in the references; they are here because they are the ones worth
knowing before you start.

**Per-expert derived scalars must migrate with the slot.** Upstream registers
four weight tensors for migration and knows nothing about a target's own
per-expert scalars. Migration runs in a C++ thread between forwards, so there
is no callback to recompute in — the only shape that works is to derive them
for the full expert set at load time, store them by slot, and register them
alongside the weights. `weight-migration.md`.

**`offset_by_ep_rank` must be `False`.** The in-tree caller passes
`moe.use_dp`, which is `True` under attention DP; copying that breaks the
element-identical-routing invariant this forward is built on. `placement.md`
also records what the flag measurably does and does not do.

## The arguments assembler owns

The feature is activated through the LLM API's `MoeLoadBalancerConfig`
(`num_slots`, `initial_global_assignments`, `layer_updates_per_iter`), not
through a target-owned config file:

- **static** — `layer_updates_per_iter: 0`. Placement computed once, no
  migration. **Gate this first**: it exercises slot addressing, registration
  and routing with the migration machinery out of the picture
- **dynamic** — `layer_updates_per_iter > 0`

Neither is a tuning knob. tuner may sweep *around* them; it may not change
what they select. Record both in the run journal, with the geometry they
imply.

## Gates

boot with each variant → accuracy, both with `modeling_v2=require`.
**No third gate.**

Accuracy is not blind here: a wrong slot mapping routes tokens to the wrong
experts and the score moves. The failure this feature can hide — a balancer
wired up but inert, so placement never adapts — produces *correct output*, and
that makes it a **performance** finding rather than a correctness one. Perf in
this repo is measured, never gated: whether the placement actually adapts, and
what it buys, belongs to `/tuning` and to the journal's performance section.

What replaces a gate is the **ladder**, and it costs nothing extra:

1. the static variant first — placement computed once, no migration. Green
   here means slot addressing, registration, routing and the manifest are all
   correct, with the migration machinery out of the picture
2. the dynamic variant after. A failure now is a migration failure, and step 1
   is what makes that attribution possible

Run them in that order and record both. A dynamic run gated without the static
one first tells you the target works; it does not tell you *which half*.

## Report additions

- the slot geometry: `num_experts`, `num_slots`, per-rank local slots
- **the per-slot tensor list** — every tensor registered for migration, and
  for each per-layer scalar that is *not* registered, why it is not per-expert
- the static gate and the dynamic gate, separately
- what the static gate established, and what the dynamic one added on top
- the digests of the upstream files the adapter binds against
