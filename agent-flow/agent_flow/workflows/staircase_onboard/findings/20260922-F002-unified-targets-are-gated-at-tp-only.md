# F002 — a `unified` target is gated at TP only, and claims a member set it never ran

- **Found:** 2026-09-22 03:45 PDT, deciding how to express TP4 / TEP4 / ADP4
- **Component:** `agent-flow/agent_flow/workflows/staircase_onboard/gates.py`, `topology_flags`
- **Severity:** a unified run reports green for configurations it did not measure
- **Status:** fix designed, not yet applied

## What happens

`topology_flags` translates the parallel segment into `quickstart_advanced.py`
flags. Its own docstring states the invariant it exists to protect:

> Every gate is **topology-blind** on its own: a `tep4` target served at tp2
> boots, scores, and produces a self-consistent curve while the directory
> name lies.

For the three concrete segments it honours that exactly — `tep` adds
`--moe_ep_size`, `dep` adds `--moe_ep_size` and `--enable_attention_dp`. For
`unified` it returns `[--tp_size=N]` and nothing else.

So a unified target is built once and gated once, **at pure tensor
parallelism**. Expert parallelism is never switched on, attention data
parallelism is never switched on, and the two code paths that exist only to
serve those configurations are never executed by any gate.

## Why that is worse than it sounds

The parallel skill's own reference is unambiguous about the intended
contract (`skills/staircase-modeling-parallel/references/unified.md`):

> Each member runs the **full** accuracy gate, not a spot check. There is no
> static equivalence proof to lean on: an earlier experiment carried one — an
> AST-level check comparing the unified target's vocabulary, weight manifest
> and per-layer op sequence against each sibling's — and it was deliberately
> removed once `unified` became the default form...
>
> So the gates are the only evidence that the members agree.

The static proof was removed *because the gates were going to carry the
claim*. The gates do not carry it. The member set is prose in `plan.md` and
in the target's Coverage section, and nothing in the orchestrator reads it.

Worse, `unified` is `DEFAULT_PARALLEL` (`task_schema.py`), so a `task.yaml`
that simply omits `parallel:` gets the configuration with the weakest
gating and the broadest claim.

## What this costs the MiMo run specifically

The request is TP4 **and** TEP4 **and** ADP4. Under today's behaviour the run
would:

- build one codebase whose `self.ep_size > 1` and `self.dp` branches are
  live code,
- gate it at `--tp_size=4` only, where both branches fold off,
- and deliver a target documented as covering three configurations, two of
  which never ran.

For a 256-expert MoE the expert-parallel branch is not a minor variant — it
is where the expert weight sharding, the all-to-all and the per-rank expert
slice live. An EP bug is invisible to a TP-only gate by construction.

## Proposed change

1. **`task_schema.py`** — a new optional top-level key `members`, a list of
   parallel segments, legal only when `parallel: unified`. Each entry is
   validated by the existing `parse_parallel_segment`, and each must carry
   the same world size, since one allocation serves them all. Default when
   omitted: `[tp<W>, tep<W>, dep<W>]` where `W` is the unified world size —
   which is what `references/unified.md` already calls the baseline pair
   plus the degenerate path.
2. **`gates.py`** — `topology_flags(segment, world_size, member=None)`.
   When `member` is given it is what gets translated; the `unified` branch
   stops being a special case and becomes "translate whichever member you
   were handed".
3. **`workflow.py`** — at each unit's gate boundary, loop the boot and
   accuracy gates over the member set instead of running them once. Record
   one verdict per member in `progress.yaml`; the unit passes when every
   member passes.
4. **The cost has to be stated where it is chosen**, not discovered: with 3
   members and 4 units that is 12 accuracy gates. `members:` is therefore the
   knob a run uses to buy a cheaper answer honestly — one member, named —
   rather than getting a cheap answer labelled as an expensive one.

## Why not just run staircase three times

Three runs with `parallel: tp4` / `tep4` / `dep4` produce three targets and
three codebases, which is the thing `unified` was measured to avoid (58% less
code on deepseek-v3-lite). It would also triple the smith and assembler
budget, not just the gate budget.
