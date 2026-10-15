# EPLB — who owns what

Dynamic expert-parallel load balancing moves experts between slots while the
engine serves. Three layers, and the split is the reason a target can use it
without absorbing it.

| layer | owns |
|---|---|
| **runtime** | the balancer itself: expert placement, the migration machinery, three C++ threads, and the 64-bit signal word the Python side and the C++ side synchronize through. It decides *what* moves and *when* |
| **adapter** (`eplb_adapter.py`, class `EplbAdapter`) | the boundary: layer and slot registration, the staged-weight buffers, and closures over this target's flat `ParameterDict`. It hands over geometry and placement and relays calls |
| **modeling code** | six per-layer calls at the points in the forward where they belong, and the routing call that consumes the balancer's slot mapping |

The adapter creates no tensor, launches no kernel and moves no byte of its
own. See `trtllm-modeling-v2-core/references/adapter-rules.md` for why that
constraint is load-bearing rather than stylistic, and for the digest-pinning
rule that goes with binding a private runtime surface.

## What this buys, and what it costs

Buys: the balancer's placement algorithm, its migration scheduling, and its
cross-rank coordination, none of which the target reimplements.

Costs: the target's weights must be **addressable by slot** rather than by
expert, and everything that is per-expert must move when a slot's occupant
changes. `weight-migration.md` is that whole story, and it contains the
silent-wrong-answer trap of this feature.

## Static and dynamic are the same code

`layer_updates_per_iter: 0` means the placement is computed once and never
moves. That is not a different feature — it is this one with migration turned
off, and it is the right first gate: it exercises registration, slot
addressing and routing without the migration machinery. `placement.md` has the
knob.
