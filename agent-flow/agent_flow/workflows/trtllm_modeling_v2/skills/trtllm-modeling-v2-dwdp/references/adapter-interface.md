# DwdpAdapter — what it exposes, and when each call happens

## Construction

| member | when |
|---|---|
| `create(...)` | once, at model construction; returns `None` when DWDP is off |
| `local_experts` / `expert_offset` | the manifest reads these to know which expert range this rank materializes. Everything outside it goes into the known-not-loaded key set |
| `moe_layers` | which layer indices participate |
| `attach_weights(weights)` | builds the module tree the runtime's setup traversal expects, over this target's flat `ParameterDict` |
| `backend(layer_idx)` | the bound module for a layer, once the runtime has composed it |

## The facade, and why it exists

The runtime's setup traversal walks `model.layers[i].mlp` and reads expert
weights off it. Our target has no such tree — it has a flat `ParameterDict`.
Two private helpers bridge that without duplicating any tensor:

- **`_DwdpExpertFacade`** — the minimal `layers[i].mlp` the traversal needs.
  Properties returning the layer's flat parameters, plus the gate-side expert
  count the runtime reads to size the composite address space. It owns no
  storage; every property is a view onto the `ParameterDict`
- **`_DwdpLayerStub`** — one entry of the `layers` list. **Dense layers get an
  empty stub rather than being skipped**, because the list is indexed by
  absolute layer number and a list holding only the MoE layers would shift
  every index

## The two forward hooks

```
before_moe(layer_idx)   → the bound module. Call it immediately before the
                          expert call; it makes the full table addressable
<the MoE call>
after_moe(layer_idx)      releases the layer and starts the next one's prefetch
```

Neither raises when misplaced. `before_moe` too late addresses the previous
layer's window; `after_moe` too early evicts weights the current call still
needs. Both produce plausible output.
