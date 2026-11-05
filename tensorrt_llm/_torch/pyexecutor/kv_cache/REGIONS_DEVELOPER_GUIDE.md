# KV Cache Regions Developer Guide

Model and attention-backend code declares the persistent memory it needs as **regions**. The KV
cache manager lays every region out on KV cache manager V2 storage, records which region owns each
buffer, and derives from the declarations what the shared runtime needs: allocation, sizing,
disaggregated transfer and lookups of a region's layer groups. Shared code asks what a region
declared, not what a model is.

Code: `tensorrt_llm/_torch/pyexecutor/kv_cache/regions.py` (declarations and layout),
`KVCacheManagerV2` (declaration and layout hooks, region lookups), `MambaHybridCacheManagerV2`
(the regions of linear-attention layers and what the hybrid manager derives from them). The
linear-attention state of the hybrid managers is the user of regions today.

## Vocabulary

| Term | Meaning | Type |
|---|---|---|
| Plane | One kind of data a region stores per model layer, under one buffer role | `Plane` |
| Region layer | The planes a region stores for one rank-local model layer, plus its per-token cost | `RegionLayer` |
| Life cycle | When the region's memory is allocated, released early and handed off to the prefix index | `RegionLifeCycle` |
| Transfer | How the region's planes move between disaggregated workers | `RegionTransfer` |
| Region | Planes over a set of model layers with one life cycle | `Region` |
| Provider | Code that declares regions | `RegionProvider` |
| Layout | The declared regions and which region owns each buffer | `RegionLayout` |

## How a region maps onto V2 storage

V2 storage offers two allocation shapes on the same paged pools. The life cycle selects one:

| Allocation unit | Hand-off | Early release | V2 layer config, reuse policy |
|---|---|---|---|
| `TOKEN_BLOCK` | `BLOCK_BOUNDARIES` | outside `window`, or none | `AttentionLayerConfig`, `REQUIRED` |
| `TOKEN_BLOCK` | `NEVER` | outside `window`, or none | `AttentionLayerConfig`, `PRIVATE` |
| `REQUEST` | `DECLARED_POINTS` | all but the latest state | `SsmLayerConfig`, `REQUIRED` |
| `REQUEST` | `NEVER` | all but the latest state | `SsmLayerConfig`, `PRIVATE` |

**A life cycle binds to a buffer, not to a layer.** Every region covering a model layer stores
its planes as buffers of that layer's one V2 layer config (`lay_out_regions`). The layer takes the
reuse policy of the first region that covers it, in declaration order; the buffers of a region
with another policy override it. V2 groups buffers by their effective policy, so a layer covered
by regions of several life cycles spans several V2 layer groups and keeps its id. A layer no
region covers keeps the manager's own config.

**Ids.** Region storage takes the ids of the model layers it covers: no storage layer is added,
every storage id indexes the manager's per-model-layer tables, and disaggregated transfer ids stay
the global model layer ids. Compacting zero-size layers renumbers the buffer owners.

**What one layer config can hold.** The layout rejects two regions storing the same role on a
layer, regions with different allocation units on a layer (one layer is either slot or span
storage), and token-block regions with different windows on a layer (the window is a property of
the V2 layer).

**Ownership.** `RegionLayout.owners` maps `(layer id, role)` to the owning region. The layout
answers which region owns a buffer (`owner`), which layers and roles a region stores (`layer_ids`,
`roles`), and, given the V2 runtime, a region's layer groups (`group_ids`, through
`get_buffer_group_id`).

## Manager hooks

| Hook | When | Use |
|---|---|---|
| `KVCacheManagerV2._region_providers()` | Construction | The providers whose regions the manager stores; none by default |
| `RegionProvider.declare_regions()` | Once the per-layer views (`pp_layers`, `layer_offsets`) exist, before the memory budget is costed | Return the regions; costing reads them before any buffer exists |
| `KVCacheManagerV2._declare_regions()` | Same | Collects the providers' regions into `region_layout` |
| `KVCacheManagerV2._lay_out_regions(config)` | After the base config, before `_build_cache_config` | Replaces the configs of covered layers with the regions' buffers |
| `KVCacheManagerV2.region_layout` | Any time | The declared regions and, once laid out, the buffer owners |
| `KVCacheManagerV2.get_region_group_ids(region)` | After the pools exist | The V2 layer groups of a region's buffers |
| `get_disagg_role_mapper_kinds()`, `get_disagg_role_layouts()`, `get_disagg_ignored_roles()` | Disaggregated page-table build | Add each region's `RegionTransfer` declarations |

A subclass's `_build_cache_config` runs after the layout and must leave region buffers as declared.

## Linear-attention regions

`MambaHybridCacheManagerV2` is the provider of its local recurrent layers' regions
(`declare_regions`):

| Region | Life cycle | Planes | Transfer |
|---|---|---|---|
| `recurrent_state` | per request, snapshotted at declared points (`REQUIRED`) | `ssm_state`, `conv_state`, and the planes a model adds through `_get_recurrent_buffer_configs` | replay planes it holds are rebuilt |
| `replay_state` | per request, never handed off (`PRIVATE`) | the model state's replay buffers | all rebuilt |

The model state declares its replay buffers: a `MambaReplayState` gives their per-slot dtype and
shape per layer (`replay_layout`) and says whether its kernels address them by the recurrent-state
slot (`replay_shares_state_slot`). Replay buffers addressed by their own slot form `replay_state`;
the others share the recurrent state's life cycle and are planes of `recurrent_state`, so one
region still has one life cycle. A disaggregated context server never decodes and declares no
replay buffers.

| Manager, mode | `recurrent_state` planes per layer | `replay_state` planes per layer |
|---|---|---|
| Any, no speculative decoding or context server | `ssm_state`, `conv_state` | none |
| Nemotron, Mamba2 replay | `ssm_state`, `conv_state` | `mamba2_old_x`, `mamba2_old_B`, `mamba2_old_dt`, `mamba2_old_dA_cumsum`; `mamba2_replay_meta` on the first layer |
| Qwen3.5 / Qwen4Exp, GDN Triton replay | `ssm_state`, `conv_state` (Qwen4Exp PLE layers: `ple_ngram_context`, `ple_conv_state`) | `gdn_old_x`, `gdn_old_B`, `gdn_old_dt`; `gdn_replay_meta` on the first layer |
| Qwen3.5 / Qwen4Exp, GDN FlashInfer ring replay | as above, plus `gdn_k_ring`, `gdn_u_ring`, `gdn_g_ring` and `gdn_replay_meta` on the first layer | none: FlashInfer takes one state index |
| Kimi K3 / Glm5Next, KDA replay | `ssm_state`, `conv_state` | `kda_conv_q/k/v`, `kda_k/g/v/beta_cache`; `prev_num_accepted_tokens` on the first layer |
| Kimi K3, KDA with draft records | as above, plus `kda_conv_q/k/v`, `kda_state_tok` and `prev_num_accepted_tokens` on the first layer | none: the engine's kernels take one state index |

## What the hybrid manager derives from the declarations

| Runtime need | Declaration it reads | Where |
|---|---|---|
| SSM layer configs of the recurrent layers | the regions' planes and life cycles | `KVCacheManagerV2._lay_out_regions` |
| Typed views registered for binding | buffers of per-request regions | `_build_cache_config` (`_recurrent_buffers`) |
| Bytes per recurrent-state slot | planes of `REQUIRED` per-request regions | `_mamba_state_bytes_per_slot` |
| Fixed replay intercept of the quota | planes of `PRIVATE` per-request regions | `_private_replay_bytes_per_slot`, `_private_replay_quota` |
| `max_live_kv_caches` (IndexMapper capacity) | a `PRIVATE` per-request region exists | `_max_live_kv_caches` |
| Replay slot index source | the `replay_state` region's layer group | `replay_layer_group_id` |
| Roles kept out of disaggregated transfer | `RegionTransfer.rebuilt_roles` | `get_disagg_ignored_roles` |

## Backward compatibility

A manager that declares no region builds the same cache config and answers every layer-based API
(`layer_grouping`, `get_layer_group_id`, `layer_to_pool_mapping_dict`, the page-table tensors)
exactly as before: `_lay_out_regions` returns its config unchanged. Region users additionally get
the region-level lookups above.

## Not covered yet

- **Attention regions.** A token-block region would replace the manager's own attention-layer
  config, and regions with different windows cannot share a layer until V2 supports per-buffer
  windows. The declarations attention regions need already exist (`RegionLayer.bytes_per_token`,
  `Region.resumes_at_block_boundaries`, `Region.positioned_by_history`), but the manager does not
  read them yet; FP4 MLA keeps its layout policy.
- **Class-level cost estimates.** `get_cache_size_per_token` runs before any manager exists and
  still uses the per-class estimators (`_estimate_private_replay_bytes_per_slot`,
  `_extra_state_bytes_per_rank`).
- **Executor-wide gates.** Context-only workers are still recognized through the managers'
  existing flags (`kda_context_only`, `TRTLLM_DISAGG_ROLE`).

## Rules for shared code

| Rule | Instead of |
|---|---|
| Ask which region owns a buffer and what it declared | Asking whether a layer is a Mamba, GDN or KDA layer |
| Give every life cycle its own region | Placing one region's planes in another life cycle |
| Declare a buffer with its own life cycle as a region | Overriding a buffer's reuse policy in a manager hook |
| Read sizes, bounds and transfer roles from the declarations | Special-casing a model's replay layout in shared code |
| Resolve a region's layer groups with `get_region_group_ids` | Looking up the group of one of its roles on one layer |

## Migrating state to regions

1. Declare one region per life cycle, with its planes per model layer and its transfer
   declaration, from the manager (or a provider it assembles in `_region_providers`).
2. Replace the shared-code branches the declarations make redundant.
3. Show that nothing changed: for every configuration the model supports, compare before and after
   the built `KVCacheManagerConfig`, `pool_group_descs`, the page-table tensors
   (`kv_cache_pool_pointers` by owning pool and offset, `kv_cache_pool_mapping`, `index_scales`),
   the class cost estimate (`get_cache_size_per_token`), the disaggregated page table
   (`build_page_table_from_manager`) and the views bound for the kernels. Raw device addresses
   differ between processes; compare them by the pool and offset they fall in.
