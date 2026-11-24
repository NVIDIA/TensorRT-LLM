# KV Cache Regions Developer Guide

Model and attention-backend code declares persistent memory that the KV cache manager does not
derive from a layer config as a **region**. A region names its planes per model layer and one life
cycle. `KVCacheManagerV2` lays each region out as a native `RegionConfig`, so the region's buffers
get a life cycle, and a layer group, of their own. Shared code asks what a region declared, not
what a model is.

Code: `tensorrt_llm/_torch/pyexecutor/kv_cache/regions.py` (declarations and their layout),
`KVCacheManagerV2` (declaration and layout hooks, region lookups),
the model families' replay providers (`modules/mamba/replay_region.py`, `modules/fla/replay_region.py`,
`modules/kimi_kda/replay_region.py`), Kimi K3's recurrent-state provider
(`modules/kimi_kda/recurrent_region.py`), and the native side in
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/` (`RegionConfig` in `config.h`, the life-cycle
registry and pool grouping; see that directory's `AGENTS.md`).

## Vocabulary

| Term | Meaning | Type |
|---|---|---|
| Plane | One kind of data a region stores per model layer, under one buffer role | `Plane` |
| Region layer | The planes a region stores for one rank-local model layer | `RegionLayer` |
| Life cycle | How the region's memory is allocated, when it is handed off to the prefix index, and the window of a token-block region | `RegionLifeCycle` |
| Region | Planes over a set of model layers with one life cycle, and optionally a fixed slot count | `Region` |
| Provider | Code that declares regions | `RegionProvider` |
| Layout | The declared regions and which region owns each buffer | `RegionLayout` |

## The native contract

| Property | Rule |
|---|---|
| Supported life cycles | `REQUEST` allocation, handed off `NEVER` (reuse policy `PRIVATE`) or at `DECLARED_POINTS` (`REQUIRED`). `TOKEN_BLOCK` allocation, handed off at `BLOCK_BOUNDARIES` (`REQUIRED`). |
| Request slots | One slot per living KV cache and beam, held while the cache is suspended and released when it closes. |
| Request hand-off | A `PRIVATE` region is never published to the prefix index. A `REQUIRED` request region is an SSM life cycle like a recurrent layer's state: its slot is snapshotted at the last token of each `commit()`, and a request that matches a prefix restores the snapshot at the matched endpoint. The match backs off to the last endpoint that has both a snapshot and its attention pages. |
| One `REQUIRED` SSM life cycle | Prefix matching gates on one recurrent checkpoint, so layer configs and regions together have at most one `REQUIRED` SSM life cycle. The native `LifeCycleRegistry` rejects a second at construction. A region that takes over a layer's recurrent state therefore takes all of it, leaving the layer config no `REQUIRED` buffer. |
| Identity | The region index is part of the native attention or SSM life cycle, so a region never shares a layer group with a layer config's buffers or another region, even with equal parameters. |
| Numbering | Region layer groups follow every layer-config group, in declaration order. Declaring a region never renumbers the other groups. |
| Ownership | Each `(layer, role)` belongs to one layer config or one region. A request region sits on an `SsmLayerConfig` and a token-block region on an `AttentionLayerConfig`, which may hold no buffers of its own, and sets no reuse policy, `tokens_per_block_override` or sparsity of its own. |
| Fixed slot count | Only a request region may set a fixed count. Set, the region's group gets a pool group of its own with exactly that many GPU slots, taken from the quota before ratio sizing. `resize()`, `adjust()` and the pool-rebalance tuner leave it at that size, and a GPU quota too small to hold it is raised, as for the constraint floors. Unset, the group is sized by ratio like the others. |
| Token-block region | One page per token block, committed to the prefix index at block boundaries and required by every prefix match. Its native life cycle is an `AttnLifeCycle` with the region's window and no sink blocks, so its slot floor, stale blocks, page-loss handling, conversation drop plans and match coverage are those of an attention layer with that window. |

**The provider computes the fixed slot count.** The core does not count owners. A count that must
let every owner resume includes the headroom `KvCache.resume` needs: resume fails when any pool
group is above `max_util_for_resume`, so `N` owners need `ceil(N / max_util_for_resume)` slots.
The pool group belongs to the region alone, so this arithmetic is exact.

## Manager hooks

| Hook | When | Use |
|---|---|---|
| `_region_providers()` | `KVCacheManagerV2.__init__`, after the per-layer views and the IndexMapper capacity exist and before the memory budget is costed | Return the providers of this manager's regions |
| `_declare_regions()` | Same point | Collects the providers' regions into `region_layout` |
| `_remove_zero_size_buffers()` | Config build | Keeps every layer a declared region covers, and renumbers the declared regions when it compacts the layer list |
| `_lay_out_regions(config)` | Config build, after `_build_cache_config`, zero-size removal and mixed-precision layers | Adds one native `RegionConfig` per region to `config.regions` |
| `region_layout` | After declaration | The regions and, once laid out, the owner of each buffer |
| `get_region_group_ids(name)` | After construction | The region's layer group, through `KVCacheManager.get_region_group_id`; empty for an undeclared region |
| `KVCacheManager.get_group_life_cycle(group)` | After construction | Any layer group's native life cycle: kind, window, sinks, reuse policy and region index. The manager's window readers (KV events, stats, stale masks, the disaggregation extractor, the connector layout) take a group's window from it, so a region's group reports its own window |
| `get_disagg_ignored_roles()` | Disaggregated transfer | Model and backend managers identify locally rebuilt buffers to exclude |
| `mamba_metadata_cls` | Attention metadata creation | The Mamba metadata class the manager's state needs (the model's own class wins); the family classes below carry replay slots and replay work items |

## Replay regions of the model families

Speculative decoding's replay state belongs to its model family. The shared managers
(`KVCacheManagerV2`, `MambaHybridCacheManagerV2`) and the shared Mamba metadata know nothing of it:
each family's provider declares its region and keeps its request slots, and the family's cache
managers compose the provider through the generic hooks above and the recurrent-state ones
(`_setup_model_state`, `_setup_state_indices`, `_reset_context_mamba_slots`, `_update_model_state`,
`_get_recurrent_buffer_configs`, `_estimate_fixed_region_bytes`, `free_resources`, `shutdown`).

| Family | Provider | Managers | Metadata |
|---|---|---|---|
| Mamba2 | `Mamba2ReplayRegion` (`modules/mamba/replay_region.py`) | `NemotronHybridCacheManager` | `ReplayMamba2Metadata` (`modules/mamba/replay_metadata.py`) |
| GDN | `GDNReplayRegion` (`modules/fla/replay_region.py`) | `Qwen35HybridCacheManager`, `Qwen4ExpHybridCacheManager` | `GDNMambaMetadata` (`modules/fla/gdn_metadata.py`) |
| KDA | `KDAReplayRegion` (`modules/kimi_kda/replay_region.py`) | `KimiK3HybridCacheManager`, `Glm5NextCacheManager` | `KimiK3MambaMetadata` |

Every provider follows the same contract:

| Case | Storage |
|---|---|
| Own slots (Mamba2, GDN Triton / CuTe / fused verify, KDA including Kimi K3 draft records) | `replay_state` (`REQUEST`, `NEVER`) over the local recurrent layers, `fixed_slot_count = ceil(IndexMapper capacity / max_util_for_resume)` |
| Shared slots (GDN FlashInfer ring, GLM5Next draft records, Kimi K3 SM107 replay) | `shared_buffers()` adds the replay buffers to the layer's `SsmLayerConfig`, and they follow the recurrent state's life cycle |
| Disaggregated context server | The manager omits the replay provider. Kimi K3 still declares its recurrent region on ranks with local KDA layers. |

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

The Mamba2 and GDN metadata classes extend `Mamba2Metadata` side by side; they share the replay
primitives of `modules/mamba/replay_metadata.py` (the work-item layout and builders,
`ReplaySlotIndices`), and only the GDN class applies the GDN gates (FlashInfer ring replay, the
fused small-batch commit). The models name their family's class as well (`Qwen3NextForCausalLM`,
`Qwen4ExpForCausalLM`, `NemotronHForCausalLM` and their multimodal wrappers), because the V1
managers serve both families and name none.

## The families' other state

The shared hybrid manager and Mamba metadata leave the rest of the families' state to them too,
through hooks that name no family:

| State | Owner | Hooks of `MambaHybridCacheManagerV2` |
|---|---|---|
| FP4 MLA latent-cache layout | `KimiK3HybridCacheManager` selects `Fp4MlaV2CacheLayoutPolicy` for an NVFP4 MLA cache | `_configure_attention_cache_layout()` installs `_attention_cache_layout_policy`, to which the manager forwards attention-layout operations; `_estimate_attention_cache_cost()` gives the cost estimate its pages and rings. The layout is the manager's `fp4_mla_layout` object, which the V2 manager hooks reach through `Fp4MlaLayoutHooks` in the manager's bases |
| A standalone drafter's KV layers | `KimiK3HybridCacheManager` (`accepts_appended_draft_layers`) | `_append_layers()` extends the layer masks, `_bind_appended_layers()` maps them to local layers |
| GDN in-forward snapshots | `Qwen35HybridCacheManager` owns `InforwardSnapshots` (`modules/fla/inforward_snapshot.py`); `GDNMambaMetadata` plans the split | `_setup_model_state()`, `prepare_resources()`, `shutdown()`; the executor and scheduler read `inforward_snapshot_rows` (0 on `KVCacheManagerV2`) |
| GDN mixer state (iteration context, state-reset gate, aligned decode slots) | `GDNMambaMetadata` | `prepare()`, `prepare_steady_gen_step()` |

`tests/unittest/_torch/pyexecutor/kv_cache/test_replay_decoupling.py` checks that the shared
modules name none of this state and import none of the families' modules.

## The KDA recurrent region

Kimi K3 declares the recurrent state of its local KDA layers as the `recurrent_state` region of
`KDARecurrentRegion`, on every rank with local KDA layers: with or without speculative decoding,
and on a disaggregated context server too. It declares this region before the replay region, so
the recurrent group id does not depend on whether a replay region follows. GLM5Next and Kimi K3
SM107 keep their recurrent state in their layer configs.

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

A disaggregated context server declares no replay region. On Kimi K3 targets with independent replay slots, the replay
region holds the drafts' records like any other replay plane, in replay slots independent of the
recurrent-state slots, so retained recurrent snapshots carry no draft payload.

KDA keeps `enable_kv_pool_rebalance` off: its views are bound once to the recurrent-state pool.

## The FP4 MLA regions

With the FP4 MLA cache layout, `Fp4MlaCacheLayout.declare_regions()` declares the MLA
state of the local MLA layers as three token-block regions, all handed off at block boundaries:

| Region | Layers | Planes | Window |
|---|---|---|---|
| `fp4_mla_latent` | Local FP4 MLA layers | `KEY`, `KEY_BLOCK_SCALE`, and the optional `MLA_V_SCALE` and `MLA_V_PACKED` | None |
| `fp8_mla_latent` | Local FP8 fallback layers; omitted on a rank without one | `KEY` | None |
| `mla_hp_ring` | Local FP4 MLA layers | `MLA_HP_TAIL`, one BF16 ring of `fp4_mla_hp_pool_size` tokens per block | `fp4_mla_hp_pool_size` |

The MLA layers hold their region buffers on empty attention layer configs. Kimi registers
its FP4 layout as a region provider between recurrent and replay state when those are present.
The standalone `Fp4MlaKVCacheManagerV2` places its private HP ring on virtual layers.

Kimi reads configured pool weights in model-layer order, followed by the HP ring and replay.
It maps them to native lifecycle ids after region layout. When the drafter’s share is omitted,
its page capacity follows the scarcer of the two full-history latent pools. The windowed HP
ring keeps its configured share.

The layout finds each group with `get_buffer_group_id(layer, role)`. A disaggregation transfer ID
follows the buffer's role: latent planes map to `2 * L` and the HP ring to `2 * L + 1`, where `L` is
the global model layer, wherever the ring is stored. Buffered KV events and per-lifecycle iteration stats carry the region name.

## Adding a region

1. Implement `declare_regions()` on a provider and return it from `_region_providers()`.
2. Give every plane its exact byte size per request slot or token block. Token-block regions are sized by ratio.
3. Token blocks hand off at `BLOCK_BOUNDARIES` with an optional window. For a request region, choose the hand-off. `NEVER` keeps the region private to its request. `DECLARED_POINTS` makes
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
