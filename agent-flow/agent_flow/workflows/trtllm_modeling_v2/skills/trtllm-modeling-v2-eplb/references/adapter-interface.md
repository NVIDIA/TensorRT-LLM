# EplbAdapter — what it exposes, and when each call happens

The adapter is the target's whole view of the balancer. Everything below is
called by the target, except where noted; nothing here does work of its own.

## Construction and registration

| member | when |
|---|---|
| `create(...)` | once, at model construction. Returns `None` when EPLB is off — **and that is a different thing from "the balancer exists but is static"**, which is the distinction the caller most often collapses |
| `initial_global_assignments(layer_idx)` | before weights load — which expert sits in which slot at start |
| `initial_local_expert_ids(layer_idx)` | the manifest reads this to know which experts this rank must load |
| `attach_layers(weights)` | registers each MoE layer with the balancer and hands it closures over the flat `ParameterDict` |
| `register_weights(moe_slots, rebuild_views)` | enrols the per-slot tensors in the migration queue |
| `stage_layer(layer_idx, staged)` | the CPU-side staging buffers |
| `migrates_weights` | whether this configuration will move anything at all |

## The six forward contact points

Per MoE layer, per step, in this order:

```
start_wait_gpu_stage(layer)     ─┐ the layer's weights may be mid-migration;
done_wait_gpu_stage(layer)      ─┘ these bracket the wait
update_statistic(layer, ...)     record this step's expert load
route(layer, token_selected_experts)   expert ids → slot ids
start_set_cpu_stage(layer)      ─┐ release the layer back to the migration
done_set_cpu_stage(layer)       ─┘ machinery
```

**Timing is the content.** Each of these is a synchronization point with a C++
thread; calling one at the wrong place in the forward does not raise — it
races. The wait bracket must enclose every use of that layer's expert weights,
and the release bracket must come after the last one.

`route` is the only one that returns a tensor the forward consumes: token →
*slot*, not token → expert. Everything downstream addresses slots. Its second
argument is a trap; see `placement.md`.

## Observability

`placement_drift()`, `dump_placement_at_exit()`, `staging_expert_ids()` — they
read state and change nothing. Useful for answering "is the placement actually
moving", which is a tuning question rather than a gate: `/tuning` reads them
when the campaign has a placement axis.
