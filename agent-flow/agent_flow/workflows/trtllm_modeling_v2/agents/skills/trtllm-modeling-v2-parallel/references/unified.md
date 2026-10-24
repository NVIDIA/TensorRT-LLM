# Unified targets — one codebase over a family of parallel configurations

A `unified` segment holds one `modeling.py` and one `weights.py` that serve
several parallel configurations, folded at runtime from the `mapping` object
the engine hands the model. Measured on deepseek-v3-lite over
`{tp1, tep4, dep4}`: 1475 + 448 lines against 3538 + 1032 for the three
single-configuration siblings.

## How the folding works

Every configuration-dependent branch reads `mapping`, never a constant:

```
self.weight_tp   > 1     tensor parallelism is on
self.ep_size     > 1     expert parallelism is on
self.dp                  attention data parallelism is on
```

These are **per-capture constants** — read once at construction, never per
step — so they are legal inside a captured forward and cost nothing at
replay. Branching on anything that varies per step or per rank at runtime is
not; see `trtllm-modeling-v2-core/references/graph-and-sync.md`.

## The degenerate path is not free

A unified target at `tp1` does not run the code a `tp1` sibling would. It runs
the general path with every branch folded off, and that path can contain
structure a single-configuration target has no reason to have (an expert-call
chunk loop that runs once over one full-length chunk, say). **The degenerate
configuration exercises code no other member does**, so it is worth gating
when the checkpoint fits on one card, and "the multi-rank members are green"
does not imply it.

## Choosing the member set

Ask, per candidate: **does this checkpoint fit at this configuration on this
machine?**

- `TEP<N>` and `DEP<N>` at the run's world size are the baseline pair. TEP
  shards attention weights and reduces; DEP replicates attention, so the batch
  composition diverges per rank
- `TP1` if it fits: the only configuration that folds every branch off
- larger world sizes on request

A candidate that does not fit is recorded as excluded with the arithmetic:
per-rank weight estimate against card capacity.

## Gating

The member set is `task.yaml`'s `members:` (legal only under
`parallel: unified`; default `[tp<W>, tep<W>, dep<W>]`). The orchestrator
runs every gate a unit owes at every member, cheapest first, records one
verdict per member (`accuracy[gsm8k]@dep4`), and advances the unit only when
all pass. There is no static equivalence proof between members: the gates
are the only evidence that they agree, at `N × (boot + accuracy)`. Preflight
prints the gate count before the run starts.

## Adding a configuration later

Add the member and run its gates. **The directory does not get renamed** —
the segment name promises structure, not coverage.
