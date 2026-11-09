# KV Cache Regions Developer Guide

Model and attention-backend code declares persistent memory that the KV cache manager does not
derive from a layer config as a **region**. A region names its planes per model layer and one life
cycle. `KVCacheManagerV2` lays each region out as a native `RegionConfig`, so the region's buffers
get a life cycle, and a layer group, of their own. Shared code asks what a region declared, not
what a model is.

Code: `tensorrt_llm/_torch/pyexecutor/kv_cache/regions.py` (declarations and their layout),
`KVCacheManagerV2` (declaration and layout hooks, region lookups),
the model families' replay providers (`modules/mamba/replay_region.py`, `modules/fla/replay_region.py`,
`modules/kimi_kda/replay_region.py`), and the native side in
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
| Supported life cycle | `REQUEST` allocation, `NEVER` handed off: reuse policy `PRIVATE`. `lay_out_regions` rejects every other pair until the native side supports it. |
| Slots | One slot per living KV cache and beam, held while the cache is suspended and released when it closes. Never published to the prefix index. |
| Identity | The region index is part of the native life cycle, so a region never shares a layer group with a layer config's buffers or another region, even with equal parameters. |
| Numbering | Region layer groups follow every layer-config group, in declaration order. Declaring a region never renumbers the other groups. |
| Ownership | Each `(layer, role)` belongs to one layer config or one region. A region buffer sits on a layer that has an `SsmLayerConfig`, and sets no reuse policy, `tokens_per_block_override` or sparsity of its own. |
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
| `_remove_zero_size_buffers()` | Config build | Renumbers the declared regions when it compacts the layer list |
| `_lay_out_regions(config)` | Config build, after `_build_cache_config`, zero-size removal and mixed-precision layers | Adds one native `RegionConfig` per region to `config.regions` |
| `region_layout` | After declaration | The regions and, once laid out, the owner of each buffer |
| `get_region_group_ids(name)` | After construction | The region's layer group, through `KVCacheManager.get_region_group_id`; empty for an undeclared region |
| `get_disagg_ignored_roles()` | Disaggregated transfer | Model and backend managers identify locally rebuilt buffers to exclude |
| `mamba_metadata_cls` | Attention metadata creation | The Mamba metadata class the manager's state needs (the model's own class wins); `ReplayMamba2Metadata` carries replay slots and replay work items |

## Replay regions of the model families

Speculative decoding's replay state belongs to its model family. The shared managers
(`KVCacheManagerV2`, `MambaHybridCacheManagerV2`) and the shared Mamba metadata know nothing of it:
each family's provider declares its region and keeps its request slots, and the family's cache
managers compose the provider through the generic hooks above and the recurrent-state ones
(`_setup_model_state`, `_setup_state_indices`, `_reset_context_mamba_slots`, `_update_model_state`,
`_get_recurrent_buffer_configs`, `_estimate_fixed_region_bytes`, `free_resources`, `shutdown`).

| Family | Provider | Managers | Metadata |
|---|---|---|---|
| Mamba2 | `Mamba2ReplayRegion` (`modules/mamba/replay_region.py`) | `NemotronHybridCacheManager` | `ReplayMamba2Metadata` |
| GDN | `GDNReplayRegion` (`modules/fla/replay_region.py`) | `Qwen35HybridCacheManager`, `Qwen4ExpHybridCacheManager` | `ReplayMamba2Metadata` |
| KDA | `KDAReplayRegion` (`modules/kimi_kda/replay_region.py`) | `KimiK3HybridCacheManager`, `Glm5NextCacheManager` | `KimiK3MambaMetadata` |

Every provider follows the same contract:

| Case | Storage |
|---|---|
| Own slots (Mamba2, GDN Triton / CuTe / fused verify, KDA) | `replay_state` (`REQUEST`, `NEVER`) over the local recurrent layers, `fixed_slot_count = ceil(IndexMapper capacity / max_util_for_resume)` |
| Shared slots (GDN FlashInfer ring, the Kimi K3 engine's draft records) | No region: `shared_buffers()` adds the replay buffers to the layer's `SsmLayerConfig`, and they follow the recurrent state's life cycle |
| Disaggregated context server | Neither: the manager creates no provider |

The provider binds the replay views (`bind_replay_pool`) before the state's `bind()`, keeps each
request's replay slot (the manager's `replay_layer_group_id`, `get_replay_state_indices` and
`cuda_replay_state_indices` delegate to it), resets a replay slot a generation request acquires for
the first time, and hands the generation rows' replay slots to the state's `update`. Its `prepare`
runs after the manager's state-index upload on every step, because a replay page can move while the
recurrent-state slot stays. The family's metadata copies the replay slots into a static buffer the
kernels read (`replay_state_indices`), or aliases `state_indices` when the replay buffers share the
recurrent-state slots. Before the manager exists, the manager's `_estimate_fixed_region_bytes()`
reserves the provider's fixed pool. Replay buffers are never transferred between disaggregated
workers: the manager lists them in `get_disagg_ignored_roles()`.

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
cost estimate the same bytes. `_update_model_state()` dispatches acceptance without choosing replay
slots.

| Field | Value |
|---|---|
| Life cycle | `REQUEST`, `NEVER` |
| Layers | The local KDA layers |
| Planes | `replay_layout()`: the convolution replay windows and k/g/v/beta caches per layer, plus the accepted-draft count on the first local KDA layer |
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

A disaggregated context server declares no replay region. When the kernels address the replay
buffers by the recurrent-state slot (`replay_shares_state_slot`, the experimental Kimi K3 engine's
draft-records mode), the KDA provider supplies buffers for the recurrent `SsmLayerConfig` and
declares no region; the replay slot then equals the state slot.

KDA keeps `enable_kv_pool_rebalance` off: its views are bound once to the recurrent-state pool.

## Adding a region

1. Implement `declare_regions()` on a provider and return it from `_region_providers()`.
2. Give every plane its exact per-slot byte size; the manager costs the budget from the declaration.
3. Decide whether the region needs a fixed slot count. A request-private region whose demand is
   its owners' count gains nothing from slots beyond it, so give it the count those owners need,
   including the resume headroom. The manager adds the fixed pool to the GPU quota as it is;
   override `_estimate_fixed_region_bytes()` so the cost estimate reserves the same bytes.
4. Resolve the region's layer group with `get_region_group_ids(name)` and its buffers with the
   per-buffer APIs (`get_buffer_group_id`, page strides, base addresses).
5. Describe transfer views through the manager's disaggregation hooks. List buffers the
   receiving worker rebuilds in `get_disagg_ignored_roles()`.
