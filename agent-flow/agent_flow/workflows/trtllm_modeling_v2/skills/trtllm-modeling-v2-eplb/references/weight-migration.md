# Weight migration — what must move when a slot changes occupant

When slot 37 stops holding expert 51 and starts holding expert 200,
**everything indexed by that slot must change together.** Anything left behind
pairs expert 200's weights with expert 51's scalars, and that is a silent
wrong answer: no crash, no assertion, a slightly wrong number.

## What upstream registers, and what it misses

The migration queue upstream enrols four tensors per layer:

```
w3_w1_weight        w2_weight        w3_w1_weight_scale        w2_weight_scale
```

An NVFP4 target's expert runner also consumes **per-expert scalar arrays** —
on the DeepSeek-R1 shape, three of them (`output1_scale`,
`output1_gate_scale`, `output2_scale`, derived from the checkpoint's
`e_ws2_1` / `e_isc2` / `e_ws2_2`). Those are per-expert, so they are per-slot,
so **they must migrate too**. Upstream does not know about them.

## Why you cannot fix it after the fact

Migration is performed by a **C++ thread in the gap between forwards**. There
is no Python callback point after a migration — nothing to hook, nothing to
recompute in. So "register the raw checkpoint scalars and re-derive after the
move" does not work; there is no "after the move".

## The shape that does work

**At load time, compute the derived scalars for all experts, store them
indexed by slot as `[local_slots]` tensors, and register them in the migration
queue alongside the weights.**

The derivation is per-expert elementwise arithmetic, so computing it for the
full expert set costs nothing extra and depends on nothing that is not already
present. The staging buffers hold the full derived values too. Migration then
keeps them consistent automatically, with no callback required.

## What does not migrate

Scalars that are not per-expert. On the same target `routed_input_scale =
1.0 / mlp_isc1` is a **shared-expert** scalar — one value for the layer, not
one per expert — and enrolling it in the migration queue would be wrong in the
other direction.

The test is mechanical: **is this value indexed by expert?** If yes it is
indexed by slot and it migrates. If no it does not. Write the answer for every
per-layer scalar in the target into the run journal; the next reader should
not have to re-derive it.

## Changing storage breaks views

Registering a tensor for migration can replace its storage. Any view built
over the old storage — a reshape, a slice held in a cached structure — is
silently stale afterwards. Rebuild views after registration; that is what the
`rebuild_views` callback in `register_weights` is for.
