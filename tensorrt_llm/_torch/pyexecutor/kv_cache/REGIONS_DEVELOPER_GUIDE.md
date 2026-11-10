# KV Cache Regions Developer Guide

Model and attention-backend code declares persistent memory that the KV cache manager does not
derive from a layer config as a **region**. A region names its planes per model layer and one life
cycle. `KVCacheManagerV2` lays each region out as a native `RegionConfig`, so the region's buffers
get a life cycle, and a layer group, of their own. Shared code asks what a region declared, not
what a model is.

Code: `tensorrt_llm/_torch/pyexecutor/kv_cache/regions.py` (declarations and their layout),
`KVCacheManagerV2` (declaration and layout hooks, region lookups),
`tensorrt_llm/_torch/modules/kimi_kda/recurrent_region.py` and `replay_region.py` (the KDA
providers), and the native side in
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/` (`RegionConfig` in `config.h`, the life-cycle
registry and pool grouping; see that directory's `AGENTS.md`).

## Vocabulary

| Term | Meaning | Type |
|---|---|---|
| Plane | One kind of data a region stores per model layer, under one buffer role | `Plane` |
| Region layer | The planes a region stores for one rank-local model layer | `RegionLayer` |
| Life cycle | How the region's memory is allocated and when it is handed off to the prefix index | `RegionLifeCycle` |
| Region | Planes over a set of model layers with one life cycle, and optionally a fixed slot count | `Region` |
| Provider | Code that declares regions | `RegionProvider` |
| Layout | The declared regions and which region owns each buffer | `RegionLayout` |

## The native contract

| Property | Rule |
|---|---|
| Supported life cycles | `REQUEST` allocation, handed off `NEVER` (reuse policy `PRIVATE`) or at `DECLARED_POINTS` (`REQUIRED`). `lay_out_regions` rejects `TOKEN_BLOCK` regions until the native side supports them. |
| Slots | One slot per living KV cache and beam, held while the cache is suspended and released when it closes. |
| Hand-off | A `PRIVATE` region is never published to the prefix index. A `REQUIRED` region is an SSM life cycle like a recurrent layer's state: its slot is snapshotted at the last token of each `commit()`, and a request that matches a prefix restores the snapshot at the matched endpoint. The match backs off to the last endpoint that has both a snapshot and its attention pages. |
| One `REQUIRED` SSM life cycle | Prefix matching gates on one recurrent checkpoint, so layer configs and regions together have at most one `REQUIRED` SSM life cycle. The native `LifeCycleRegistry` rejects a second at construction. A region that takes over a layer's recurrent state therefore takes all of it, leaving the layer config no `REQUIRED` buffer. |
| Identity | The region index is part of the native life cycle, so a region never shares a layer group with a layer config's buffers or another region, even with equal parameters. |
| Numbering | Region layer groups follow every layer-config group, in declaration order. Declaring a region never renumbers the other groups. |
| Ownership | Each `(layer, role)` belongs to one layer config or one region. A region buffer sits on a layer that has an `SsmLayerConfig`, which may hold no buffers of its own, and sets no reuse policy, `tokens_per_block_override` or sparsity of its own. |
| Fixed slot count | Set, the region's group gets a pool group of its own with exactly that many GPU slots, taken from the quota before ratio sizing. `resize()`, `adjust()` and the pool-rebalance tuner leave it at that size, and a GPU quota too small to hold it is raised, as for the constraint floors. Unset, the group is sized by ratio like the others. |

**The provider computes the fixed slot count.** The core does not count owners. A count that must
let every owner resume includes the headroom `KvCache.resume` needs: resume fails when any pool
group is above `max_util_for_resume`, so `N` owners need `ceil(N / max_util_for_resume)` slots.
The pool group belongs to the region alone, so this arithmetic is exact.

## Manager hooks

| Hook | When | Use |
|---|---|---|
| `_region_providers()` | `KVCacheManagerV2.__init__`, after the per-layer views and the IndexMapper capacity exist and before the memory budget is costed | Return the providers of this manager's regions |
| `_declare_regions()` | Same point | Collects the providers' regions into `region_layout` |
| `_remove_zero_size_buffers()` | Config build | Keeps every layer a declared region covers, as an empty `SsmLayerConfig` when the region holds all of its buffers, and renumbers the declared regions when it compacts the layer list |
| `_lay_out_regions(config)` | Config build, after `_build_cache_config`, zero-size removal and mixed-precision layers | Adds one native `RegionConfig` per region to `config.regions` |
| `region_layout` | After declaration | The regions and, once laid out, the owner of each buffer |
| `get_region_group_ids(name)` | After construction | The region's layer group, through `KVCacheManager.get_region_group_id`; empty for an undeclared region |
| `get_disagg_ignored_roles()` | Disaggregated transfer | Model and backend managers identify locally rebuilt buffers to exclude |

## The KDA recurrent region

Kimi K3 declares the recurrent state of its local KDA layers as the `recurrent_state` region of
`KDARecurrentRegion`, on every rank with local KDA layers: with or without speculative decoding,
and on a disaggregated context server too. It declares this region before the replay region, so
the recurrent group id does not depend on whether a replay region follows. GLM5Next keeps its
recurrent state in its layer configs.

| Field | Value |
|---|---|
| Life cycle | `REQUEST`, `DECLARED_POINTS`: reuse policy `REQUIRED` |
| Layers | The local KDA layers |
| Planes | `SSM_STATE` and `CONV_STATE` per layer, at the sizes the layer configs would declare |
| Fixed slot count | None: sized by ratio, above the live floor of one slot per resident sequence and reserved dummy |

While the region is declared, Kimi K3's `_get_recurrent_buffer_configs` returns no buffers, and
each KDA layer stays an empty `SsmLayerConfig` under the region's buffers. Shared Mamba code finds
the recurrent state by its `SSM_STATE` plane rather than by the region's name:
`ssm_layer_group_id` resolves through `get_buffer_group_id` to the region's group, and
`_mamba_state_bytes_per_slot()` costs a recurrent-state slot from the declared region's
`bytes_per_slot()`, which the budget reads before the regions are laid out.

Snapshots are taken at the commit boundaries `prepare_expect_snapshot_points` declares, and the
hybrid manager stops committing at the prompt. A checkpoint is therefore always a prefill state,
which the two planes describe completely. Replay state never enters a checkpoint: a request that
restores one is a context request, whose accepted-draft count is reset and whose replay windows
prefill seeds from `CONV_STATE`.

## The KDA replay region

`KDAReplayRegion` declares `replay_state` from the geometry supplied by `KDAReplayState`.
Kimi K3 and GLM5Next compose this provider in their model-specific cache managers and register it
through `_region_providers()`. The provider owns the region's identity, lifecycle, planes,
capacity. The shared managers consume its declaration through the generic
region interfaces.

The KDA managers bind the provider after the native pools and final layer mapping exist, before
binding the replay algorithm. The provider borrows typed views through `_get_state_buffer()`;
KVCM owns the underlying pages. The region's memory is its fixed pool,
`fixed_slot_count * bytes_per_slot()`, counted once and independently of recurrent snapshots.
`_fixed_region_quota()` adds it to the GPU quota and to the minimum allocation as it is, outside the
`1 / max_util_for_resume` scaling of the ratio-sized pools, because the fixed count already carries
the resume headroom. Before the manager exists, KDA's `_estimate_fixed_region_bytes()` gives the
cost estimate the same bytes, from the layout `kda_draft_records` selects: `KvCacheCreator` passes
that flag to the estimate as it passes it to the manager. `_update_model_state()` dispatches
acceptance without choosing replay slots.

| Field | Value |
|---|---|
| Life cycle | `REQUEST`, `NEVER` |
| Layers | The local KDA layers |
| Planes | `replay_layout()`: the convolution replay windows per layer and either the k/g/v/beta caches or, with `kda_draft_records`, the drafts' records (`kda_state_tok`); plus the accepted-draft count on the first local KDA layer |
| Fixed slot count | `KDAReplayState.replay_slot_count()`: `ceil(IndexMapper capacity / max_util_for_resume)`. The IndexMapper capacity bounds the living caches, active and suspended, reserved dummies included. KDA models run at beam width 1. |

Kimi K3 and GLM5Next exclude replay buffers through `get_disagg_ignored_roles()`. The
disaggregation extractor uses that hook to select transferable views. The generation worker
seeds replay from the transferred recurrent state after the transfer completes.

Replay state follows its request through suspend and resume; the physical slot can move. The
provider uploads each request's replay slot; the KDA manager exposes it through
`get_replay_state_indices` and `cuda_replay_state_indices`. The KDA metadata keeps these slots in
`KimiK3MambaMetadata.replay_state_indices`, and the mixer, the kernels and the acceptance update
address the replay buffers by them. The shared `Mamba2Metadata` and `MambaAcceptanceBatch` carry no
replay slots. A replay slot a request acquires for
the first time is reset (`reset_replay_slots`). The provider tracks replay-slot changes independently
of the recurrent-state upload, so a moved replay page refreshes its index even when the recurrent
slot stays fixed. Release clears the request's initialization marker; suspend retains it.

A disaggregated context server declares no replay region. With `kda_draft_records`, the replay
region holds the drafts' records like any other replay plane, in replay slots independent of the
recurrent-state slots, so retained recurrent snapshots carry no draft payload.

KDA keeps `enable_kv_pool_rebalance` off: its views are bound once to the recurrent-state pool.

## Adding a region

1. Implement `declare_regions()` on a provider and return it from `_region_providers()`.
2. Give every plane its exact per-slot byte size; the manager costs the budget from the declaration.
3. Choose the hand-off. `NEVER` keeps the region private to its request. `DECLARED_POINTS` makes
   the region the manager's one `REQUIRED` SSM life cycle: declare in it every plane a restored
   request needs, and leave the layer configs no `REQUIRED` buffer. A region with an `SSM_STATE`
   plane is the hybrid manager's recurrent state, and its slot is the recurrent-state slot.
4. Decide whether the region needs a fixed slot count. A request-private region whose demand is
   its owners' count gains nothing from slots beyond it, so give it the count those owners need,
   including the resume headroom. The manager adds the fixed pool to the GPU quota as it is;
   override `_estimate_fixed_region_bytes()` so the cost estimate reserves the same bytes.
5. Resolve the region's layer group with `get_region_group_ids(name)` and its buffers with the
   per-buffer APIs (`get_buffer_group_id`, page strides, base addresses).
6. Describe transfer views through the manager's disaggregation hooks. List buffers the
   receiving worker rebuilds in `get_disagg_ignored_roles()`.
