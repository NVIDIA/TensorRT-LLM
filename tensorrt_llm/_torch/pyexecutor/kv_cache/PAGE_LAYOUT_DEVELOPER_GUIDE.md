# Page Layout Developer Guide

A `KVCacheManagerV2` describes its pages once, as a `PageLayout`: which buffers a page holds, where
each sits in its pool slot, and how its bytes split across ranks. Everything that moves pages reads
that one description and nothing else about the cache:

* the native P/D transfer (`disaggregation/page_mapping/`) maps two ranks' pages onto each other;
* the lender (`kv_cache/sharing/`) lends pages in place and names their layout (`layout_id`).

Adding a cache means describing it in its manager. Neither consumer changes.

| File | Holds |
|------|-------|
| `page_layout.py` | `PageLayout` / `LayerGroupDesc` / `BufferDesc` / `Segment`: one rank's pages as immutable values, no addresses |
| `role_layout.py` | `head_major`, `token_major`, `replicated`, `sectioned`: the segments of the usual layouts |
| `layout_builder.py` | `build_page_layout(manager)`: the storage walk that turns a manager's hooks into its `PageLayout`; `device_pools(manager)` |
| `kv_cache_manager_v2.py` | `get_page_layout()` and the hooks below, with the defaults of plain attention |
| `disaggregation/page_mapping/` | the native transfer's reading of two layouts: `RankPages` (layout + pool addresses), the correspondence, unit rules by layer-group unit |
| `sharing/_layout.py` | the lender's reading: `derive_layout`, `layout_id` |

## The layout

```
PageLayout
├─ tokens_per_page, pool_slot_bytes        pool group -> slot widths of its pools
├─ layer_groups: [LayerGroupDesc]          one per storage life cycle, in the manager's order
│    ├─ kind: "attention" | "state"        what the group holds; sets the defaults below
│    ├─ grid                               the parallel dimensions its split buffers spread over
│    ├─ unit: "block" | "slot"             a request's units: blocks by ordinal, or one slot
│    ├─ blocks_by_cp                       under Helix, CP rank r holds blocks r::cp_size
│    ├─ kept_window, read_window, num_sink_blocks
│    ├─ pool_group, layers, model_layers   global layer ids, the same under any PP split
│    └─ shard                              the share of the group's content this rank holds
├─ buffers: [BufferDesc]                   one per (layer, role) in a page
│    ├─ layer, role                        what it is; peers match buffers on it
│    ├─ pool_group, pool_index, slot_offset, size
│    ├─ segments: [Segment]                how its bytes split across ranks
│    └─ transferred                        False: kept in local copies of whole pages, never moved
└─ topology                                this rank's TP / PP / CP / DP place
```

### How a layer group splits

| | `grid` | `unit` | `blocks_by_cp` |
|---|---|---|---|
| `kind="attention"` (default) | `("tp",)` | `"block"` | `True` |
| `kind="state"` (default) | `("tp", "cp")` | `"slot"` | `False` |

`grid` names the parallel dimensions, outer to inner, whose ranks split the group's buffers:
`("tp",)` or `("tp", "cp")`. A rank's place in it is `grid_position(topology, grid)` (the last
dimension minor). Ranks along a dimension the grid leaves out hold the same bytes -- unless the
group's blocks split by CP rank (`blocks_by_cp`) -- and one of them is elected to send. Under
attention data parallelism every rank holds its buffers whole, whatever the grid. Native and the
lender both read the grid from here; neither knows what a kind means.

Every range follows one formula, `held_span(units, size, rank, total)`: rank `r` of `size` holds
`units` from `r * total // size`. It gives a rank's slice of a split axis in the transfer, its
share of KV heads in the lender, and which TP and CP ranks of a peer it exchanges with. Peer ranks
come from the topology alone, since a receiver knows one sender rank's layout only; they cover
every pair whose bytes intersect, for any grid above.

### Segments

A buffer is its segments back to back. Each names its axes outer to inner and the axis split across
the layer group's grid of ranks:

```python
Segment(n, (("head", h),), "head")                 # head-major K/V: head_major(n, h)
Segment(n, (("token", t), ("head", h)), "head")    # token-major K/V: token_major(n, t, h)
Segment(s, (("byte", s),), "byte")                 # one section of conv state: sectioned([...])
Segment(n)                                         # every rank holds the same bytes: replicated(n)
```

`nbytes` is what this rank holds; axis names are free strings. From two ranks' segments and their
group's grid alone the transfer derives which ranks pair (their spans along the split axis
intersect), which bytes move (one run per row of the axes outside the split axis) and which of
several identical holders sends (rotated by the receiver's DP rank). A layout outside the four
builders needs no new code: declare its segments.

### Which peers pair

Two ranks pair when their block ordinals cover the same tokens -- equal `tokens_per_page` whenever
both hold block units; a mismatch is refused at peer registration -- and, buffer by buffer, when
matched buffers lie in groups that split alike (`grid`, `unit`, `blocks_by_cp`) and declare the same
axes and split axis. Bytes per unit that differ (another dtype or head size) are refused when the
two sides' runs are planned.

## The hooks

`get_page_layout()` walks the storage (pool groups, slot variants, coalesced buffers) and asks the
manager about each buffer. A subclass overrides only the hooks whose answer differs.

| Hook | Default (plain attention) |
|---|---|
| `get_layer_group_traits(layer_group)` | `"state"` for a recurrent (`SsmLayerConfig`) group, else `"attention"`, with the kind's grid, unit and `blocks_by_cp` |
| `get_buffer_layout(layer_id, role, size, tokens)` | K/V and block scales head-major by the layer's KV heads; `INDEX_KEY` replicated; **any other role raises** |
| `get_untransferred_roles()` | none |
| `get_layer_ids()` | internal layer -> (global id, model layer): the model layer, or `model_layer * num_types + type` for several attention types per layer |
| `get_kv_heads(layer_id)` | the layer's KV heads on this rank and over all TP ranks |
| `_attention_read_window(layer_group)` | the window the attention reads |

Per request, the native transfer asks:

| Hook | Default |
|---|---|
| `get_transfer_length(request)` | `prompt_len` |
| `sends_first_tokens(request)` | `True` |
| `get_untransferred_buffers(request)` | none; `(global layer, role)` pairs the receiver rebuilds |

A request's units come from its cache, not from a hook: its blocks by ordinal, and in a state
layer group the slot the cache holds (`get_ssm_block_base_index`), as the lender reads them.

A role the manager creates but does not lay out is an error at `get_page_layout()`, never a guess.

### The built-in managers

| Manager | Overrides |
|---|---|
| `MiniMaxM3KVCacheManagerV2` | `get_buffer_layout`: K/V head- or token-major as its backend writes them |
| `Fp4MlaKVCacheManagerV2` | `get_buffer_layout` (token-major latents, replicated high-precision tail), `get_untransferred_roles` (rebuilt V sidecars), `get_layer_ids` (tail layers) |
| `DeepseekV4CacheManager` | `get_buffer_layout`, `get_untransferred_roles` (NVFP4 alignment pages) |
| `CSA2CacheManager` | `get_buffer_layout` (all replicated); remote tail: `get_transfer_length`, `sends_first_tokens`, `get_untransferred_buffers` |
| `MambaHybridCacheManagerV2` | `get_buffer_layout` (sectioned conv, head-sharded SSM, replicated PLE and KDA replay state), `get_untransferred_roles` (KDA replay) |
| `QSAMambaHybridCacheManagerV2` | `get_buffer_layout`: replicated index positions |

DSA and GLM-5 Next add only an `INDEX_KEY` buffer, which the defaults lay out.

## Adding a cache

1. Make the runtime manage the memory: buffers registered by a `KVCacheManagerV2` subclass.
2. Override `get_buffer_layout` for every role the subclass creates (and the other hooks if their
   answer differs). Build segments with `role_layout.py`, or write them. A layer group that splits
   unlike its kind's default -- a cache split along tokens over TP x CP, a state with one snapshot
   per block -- overrides `get_layer_group_traits`.
3. Add the layout to the resharding matrix in `tests/unittest/disaggregated/test_page_mapping.py`,
   which checks byte by byte, across TP / PP / CP / attention-DP layouts, that every receiving rank
   gets every byte exactly once from the right source byte.

Nothing under `disaggregation/` or `sharing/` changes.

## When the model itself must grow

A page is "pool base + unit * slot bytes + offset", moved byte for byte and split along one axis per
segment over one grid of ranks. A cache outside that -- variable-length or non-contiguous
addressing, a segment split along two axes by two parallel dimensions, a unit spanning several
slots, bytes transformed in flight, a grid over a dimension other than TP and CP -- needs
`PageLayout` and its readers extended, not a special case in one manager.
