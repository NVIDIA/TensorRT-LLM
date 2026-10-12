# Placement — slots, experts, replicas, and the routing trap

## Slots are not experts

The balancer addresses **slots**. `num_slots >= num_experts`, and the surplus
is what lets a hot expert exist in more than one place:

```
num_experts = 256, num_slots = 288   → 32 experts get a second replica
```

Every per-expert tensor in the target is therefore indexed `[local_slots]`,
not `[local_experts]`, and the mapping from expert to slot is the balancer's
to change at runtime. A target that indexes weights by expert id has already
lost.

`layer_updates_per_iter` decides how much moves per step:

| value | behaviour |
|---|---|
| `0` | placement computed once, never moves. **The right first gate** — it exercises slot addressing and routing with no migration machinery |
| `n > 0` | up to `n` layers updated per iteration, walked as a circular queue over the MoE layers |

## Which replica a token goes to

With replicas, `(token, expert)` picks one by:

```
globalSlotIds[ StartOffset[e] + (k + rot) % x ]      x = ReplicaCount[e]
rot = blockIdx.x + (use_dp ? ep_rank : 0)            ← offset_by_ep_rank
```

## The trap: `offset_by_ep_rank` must be `False`

`route()`'s second argument defaults to `False`, but **the in-tree caller
passes `moe.use_dp`**, which is `True` under attention DP. Copying that breaks
the invariant this target's forward is built on.

In-tree is right to enable it: under route-then-dispatch each rank routes only
its own tokens, and without the offset all ranks would pile onto the same
replica. Our forward does not work that way, and it requires every rank to
produce element-identical routing.

One measured detail that changes how the flag should be read: **`ep_rank` is
added inside the `% ReplicaCount[e]`**, so with 288 slots over 256 experts —
at most two replicas each — it only groups ranks by `ep_rank mod
ReplicaCount`. Measured on B200 (`sm_100`), 256 experts over 288 slots at EP4:
rank 0 and rank 2 produced **bitwise identical** output, as did ranks 1 and 3
(512 tokens, top-8; the per-rank difference vector against rank 0 was
`[0, 482, 0, 482]`). The rotation formula is upstream's — re-check it if the
routing kernel changes.

That does not change the decision — pass `False` regardless — but it corrects
the reading: "offset by ep_rank" does **not** mean "each rank gets a distinct
round-robin start". And the direction that matters to us is the opposite one:
one rank differing is enough to break the invariant, so the `False` side's
"all ranks element-identical" is not a requirement that can be relaxed.

## Statistics

Three paths exist upstream for feeding the balancer load statistics. Take the
one keyed by **global expert ids** — the others are keyed by slot or are
rank-local, and both make the balancer's view of load wrong in ways that look
like a bad placement algorithm rather than a wiring error.
