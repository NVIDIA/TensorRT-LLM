# Unified targets — one codebase over a family of parallel configurations

A `unified` segment holds one `modeling.py` and one `weights.py` that serve
several parallel configurations, folded at runtime from the `mapping` object
the engine hands the model. Measured on deepseek-v3-lite over
`{tp1, tep4, dep4}`: 1475 + 448 lines against 3538 + 1032 for the three
single-configuration siblings — 58% less code, one place to fix a bug.

## How the folding works

Every configuration-dependent branch reads `mapping`, never a constant:

```
self.weight_tp   > 1     tensor parallelism is on
self.ep_size     > 1     expert parallelism is on
self.dp                  attention data parallelism is on
```

These are **per-capture constants** in graph terms — read once at
construction, never per step — so they are legal inside a captured forward and
cost nothing at replay. What is *not* legal is branching on anything that
varies per step or per rank at runtime; see
`trtllm-modeling-v2-core/references/graph-and-sync.md`.

## The degenerate path is not free

A unified target at `tp1` is **not** running the same code its `tp1` sibling
runs. It runs the general path with every branch folded off, and the general
path can contain structure the sibling has no reason to have.

Measured instance: the unified deepseek-v3-lite emits one extra
`torch/split.py` at `tp1` and `tep4` that the siblings do not, because it
always goes through the expert-call chunk loop and they have none. At those
configurations the loop runs once over a single full-length chunk, and
`torch.split(t, [t.shape[0]], 0)[0]` is a zero-copy view — same storage
pointer, shape, strides and contiguity — so every downstream op sees identical
bytes at the same address.

Harmless there, and the point is the shape of the reasoning: **the degenerate
configuration exercises code no other member does**, which is why it is worth
gating when the checkpoint fits on one card, and why "the multi-rank members
are green" does not imply it.

## Choosing the member set

Ask, per candidate: **does this checkpoint fit at this configuration on this
machine?**

- `TEP<N>` and `DEP<N>` at the world size the user gave — default 4 — are the
  baseline pair. They differ in what they exercise: TEP shards attention
  weights and reduces; DEP replicates attention and equalizes nothing, so the
  batch composition diverges per rank
- `TP1` if it fits. It is the only configuration that folds every branch off
- larger world sizes on request; they extend Coverage rather than replacing it

A candidate that does not fit is **recorded as excluded with the arithmetic**
— per-rank weight estimate against card capacity. That line is what stops the
next reader re-deriving it.

## Gating

Each member runs the **full** accuracy gate, not a spot check. There is no
static equivalence proof to lean on: an earlier experiment carried one — an
AST-level check comparing the unified target's vocabulary, weight manifest and
per-layer op sequence against each sibling's — and it was deliberately removed
once `unified` became the default form, because both of its comparisons need
sibling targets to exist and new unified targets have no siblings.

So the gates are the only evidence that the members agree, and the cost is
`N × (boot + accuracy)`. The boot gate is minutes; accuracy is roughly an hour
each. Budget it before promising a member.

**The orchestrator enforces this; it is no longer a promise made in prose.**
The member set is declared in `task.yaml` as `members:` — a list of concrete
segments (`[tp4, tep4, dep4]`), or a mapping from assembly unit to one layered
over `default` — and it is legal only under `parallel: unified`, since a
concrete segment already names the single configuration it is gated at. Every
member must declare the same world size, because one allocation serves them
all, and `unified` is not a legal member: it is the family, not a member of
it. Omitted, it defaults to `[tp<W>, tep<W>, dep<W>]` at the unified world
size — the baseline pair plus the degenerate path.

The per-unit gate loop runs boot and accuracy once **per member**, records one
verdict each (never collapsed — three measurements is the point), and advances
the unit only when all of them pass. Each verdict names its member, so
`[PASS] accuracy[gsm8k]@dep4` is what the journal, `progress.yaml` and the
reviewer's evidence carry; a line that named only the gate could not be told
apart from the two members it did not run. Preflight prints the member set and
the resulting gate count before the run starts, which is what makes `members:`
the knob for buying a cheaper answer honestly — one member, named — rather
than a cheap answer labelled as an expensive one.

The witness gates (`host_offload`, `disagg`) run at the **first** member only.
They measure a contrast between two legs of one configuration — a residency
delta, or token identity across the disaggregation hop — rather than a number
per configuration, and the residency witness's `expected_mib` is a single
prediction from the checkpoint's own shapes, so re-asking it at a member whose
attention weights are replicated rather than sharded compares a different
delta against an expectation computed for another configuration.

Every member's gates run with `modeling_v2=require`, so a member whose
topology missed the target's routing criteria
fails loudly instead of recording the built-in implementation's numbers.

## Adding a configuration later

Extend Coverage, run that member's gates with that member's topology
arguments, record them. **The directory does not get renamed** — that is the
whole reason the segment name promises structure rather than coverage. PP and
DCP land the same way when they land.
