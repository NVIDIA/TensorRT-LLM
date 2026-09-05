# KV Cache Connector

The KV Cache Connector is a flexible interface in TensorRT-LLM that enables remote or external access to the Key-Value (KV) cache. It allows developers to implement custom logic for loading, saving, and managing KV cache blocks, extending the capabilities of the standard KV cache manager.

This document explains the KV Cache Connector architecture, common use cases, and provides a detailed walkthrough of the included example.

## Use Cases

The KV Cache Connector is designed to support a variety of advanced serving scenarios:

1. **KV Cache Offloading**: Move KV cache blocks from GPU memory to cheaper/larger storage (CPU RAM, NVMe SSD, or network storage) when they are not immediately needed, and reload them when required.
2. **Custom Disaggregated Serving**: Separate the prefill (context processing) and decode (token generation) phases onto different instances or machines. The connector can be used to transmit the KV cache generated during prefill to the decode instances.
3. **KV Cache Sharing / P2P Transfer**: Share KV cache states between different model instances or across peer-to-peer connections.

## Architecture

The connector architecture is split into two main components:

* **Scheduler (Leader)**: Responsible for orchestration. It decides *what* needs to be loaded or saved and builds metadata instructions. It runs only on the leader rank (rank 0).
* **Worker**: Responsible for execution. It receives metadata from the scheduler and performs the actual data transfers (loading/saving) on the KV cache tensors. It runs on all ranks.

### API Reference

To implement a custom connector, you must subclass `KvCacheConnectorScheduler` and `KvCacheConnectorWorker`.

#### 1. Scheduler (Leader) Interface (`KvCacheConnectorScheduler`)

These methods run on the leader process and drive the connector's behavior.

* **`build_connector_meta(self, scheduler_output: SchedulerOutput) -> object`**
  * **Description**: The core orchestration method. Called during the scheduling phase. It examines the current requests and decides which blocks need to be loaded from or saved to the external store.
  * **Arguments**: `scheduler_output` contains information about new requests, blocks allocated, current request states, and the cumulative `RequestData.block_hashes` chain. `block_hashes` is read directly from each KV cache block's stored hash, which the KV cache manager commits as soon as a block becomes full -- the value matches the hash that KV cache events will subsequently emit for the same block. The chain only covers beam 0; the executor rejects `kv_connector_config` at startup when `max_beam_width > 1`, so connectors may assume beam-width-1 inputs.
  * **Returns**: An arbitrary metadata object (picklable) that describes the tasks for the workers. This object is broadcasted to all workers.

* **`get_num_new_matched_tokens(self, request: LlmRequest, num_computed_tokens: int) -> tuple[int, bool]`**
  * **Description**: Called when a new request arrives. It checks to see if any KV cache can be loaded from an external KV store.
  * **Returns**: A tuple `(num_tokens, is_async)`. `num_tokens` is the number of tokens found in the external cache. `is_async` indicates if the loading will happen asynchronously (background) or requires blocking.

* **`request_finished(self, request: LlmRequest, cache_block_ids: list[int]) -> bool`**
  * **Description**: Called when a request completes generation.
  * **Returns**: A boolean indicating if an asynchronous save operation is underway. If `True`, the system waits for the operation to complete before releasing the KV cache blocks.
  * **Note**: under sliding-window attention `cache_block_ids` covers the live window, not the whole prompt. See [What a connector can persist under a sliding window](#what-a-connector-can-persist-under-a-sliding-window).

* **`update_state_after_alloc(self, request: LlmRequest, block_ids: list[int])`**
  * **Description**: a callback to update internal state after KV cache blocks have been allocated for the prefill.
  * **Note**: on `KVCacheManagerV2` with chunked prefill, `block_ids` covers only the blocks allocated for the first chunk, because V2 allocates per chunk rather than for the whole prompt. The remaining blocks arrive as append-deltas in `RequestData.new_block_ids` on subsequent chunks. A connector that treats this callback as its only source of block ids will under-plan; drive off `build_connector_meta` instead.

* **`update_state_after_alloc_by_layer_group(self, request: LlmRequest, block_ids_by_layer_group: list[list[int]])`**
* **`request_finished_by_layer_group(self, request: LlmRequest, cache_block_ids_by_layer_group: list[list[int]]) -> bool`**
  * **Description**: the per-layer-group forms of the two callbacks above, indexed by layer group id. Entry `[g][i]` is the page slot of block ordinal `i` in layer group `g`.
  * **When they are called**: whenever the KV cache has more than one layer group — that is, one attention window size per group, as under variable sliding-window attention (VSWA). A page index is scoped to a group, so indices from different groups cannot share one list, and the flat `block_ids` / `cache_block_ids` are empty in that case. With a single layer group — every non-VSWA, non-hybrid model — only the flat forms are called and these are never reached.
  * **Both are optional, and both are required together.** A connector that implements neither runs unchanged on any single-group model. Running it against a multi-group model is rejected at startup with a message naming these two methods, rather than silently reporting empty lists.

##### Running under VSWA

Under variable sliding-window attention the KV cache allocates one pool per attention window size, and a page index only means something inside its own layer group. A single tensor and a single flat block list cannot describe that, so three methods have to be implemented together:

| Method | Replaces |
|---|---|
| `KvCacheConnectorWorker.register_kv_cache_layout` | `register_kv_caches` |
| `KvCacheConnectorScheduler.update_state_after_alloc_by_layer_group` | `update_state_after_alloc` |
| `KvCacheConnectorScheduler.request_finished_by_layer_group` | `request_finished` |

Implement all three or none. Implementing none is fine for any single-window model — the defaults handle it.

The three defaults refuse at different moments, so a partial implementation surfaces later than a missing one. `register_kv_cache_layout` runs during executor bring-up, so a connector that overrides none of the three fails there, before any request is admitted, with a message naming that method and the group and region counts it could not describe. The two scheduler defaults are only reached once a request is scheduled, so a connector that overrides `register_kv_cache_layout` but not the other two starts up cleanly and raises `NotImplementedError` on the first request instead. Implementing all three together is what keeps the failure at startup.

"Replaces" above means at call time, not at class-definition time. The three flat methods are still abstract, so a connector that only ever runs under VSWA must **also define them**, or it will not instantiate at all: `TypeError: Can't instantiate abstract class ... without an implementation for abstract methods 'request_finished', 'update_state_after_alloc'`. Overriding the per-layer-group form does not clear the abstract flag on the flat one — abstractness is tracked per method name. Define them raising `NotImplementedError` and pointing at the per-group form, as `examples/llm-api/llm_kv_cache_connector_vswa.py` does; nothing calls them once the per-group forms are overridden.

`examples/llm-api/llm_kv_cache_connector_vswa.py` is a worked connector for this case. Five things it does that the single-window connector does not need to:

1. **Address pages per group.** `layout.groups[g].regions[r]` gives the byte ranges; `region.slot_tensor(i)` is page slot `i` *of that group*. `layout.group_of_layer(layer_id)` maps a model layer back to its group, which is what the per-layer `wait_for_layer_load` / `save_kv_layer` hooks need.
2. **Read the per-group block lists.** `RequestData.new_block_ids_by_layer_group[g]` carries the page slots; the flat `new_block_ids` is empty.
3. **Carry the layer group in the cache key, and in every transfer target.** This one is a correctness requirement, not a convenience. The same token range exists in *every* layer group holding **different** KV, so a key derived from the token sequence alone collides across groups and one group's bytes will overwrite another's — then be loaded back into the wrong group. Mix `layer_group_id` (or the window size, or the layer set) into the identifier, and carry `(layer_group_id, page_slot)` rather than `page_slot` alone as the transfer target.
4. **Filter out-of-window blocks through `valid_page_slots`**, and size the store for the window rather than the prompt. See below.
5. **Serve a block only when every group holds it.** A full-attention group keeps the whole prompt while a sliding group keeps only its window, so the prefix that can be served back is bounded by the smallest window. Stop the lookup at the first block ordinal any group misses.

##### `KvCacheLayout` reference

```python
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import KvCacheLayout
```

The type is passed to `register_kv_cache_layout`; importing it is only needed for a type annotation.

A layout describes the byte ranges that repeat once per page slot. It describes ranges rather than
implying them, which is what lets one type cover MLA (a pool simply has no `value` buffer), block
scales, sliding-window attention and hybrid models without any of them being a special case.

| Attribute | Meaning |
|---|---|
| `layout.tokens_per_block` | Tokens covered by one page. |
| `layout.dtype` | Element type of the KV data, for a typed view over a region. |
| `layout.groups` | The layer groups, as `KvCacheLayerGroupLayout`. |
| `layout.group(layer_group_id)` | One group by id. Raises `KeyError` if absent. |
| `layout.group_of_layer(layer_id)` | The group owning a model layer — what the per-layer hooks route on. |
| `layout.as_single_pool_tensor()` | The `[num_blocks, num_layers, kv_factor, block_size]` view a single-pool cache hands `register_kv_caches`, or `None` when the cache cannot be described that way. This is what the default `register_kv_cache_layout` calls. |

Each `KvCacheLayerGroupLayout`:

| Attribute | Meaning |
|---|---|
| `group.layer_group_id` | The index page slots are scoped to. Dense, starting at 0. |
| `group.window_size` | Attention window for the group, or `None` for full attention. |
| `group.layer_ids` | Global model layer indices in the group — the same index space `wait_for_layer_load` and `save_kv_layer` receive. |
| `group.regions` | The `KvCacheRegion`s making up one page of this group. |
| `group.bytes_per_page` | Total bytes the group occupies for one page slot. |

Each `KvCacheRegion` is a contiguous byte range that repeats once per page slot:

| Attribute | Meaning |
|---|---|
| `region.base` | Device address of page slot 0. |
| `region.size` | Bytes the region covers within one slot. |
| `region.stride` | Distance between consecutive slots. |
| `region.num_slots` | Number of page slots. |
| `region.buffers` | The `(layer_id, role, expansion)` tuples the region covers, in memory order. `role` is the cache manager's own name, e.g. `"key"` / `"value"`. |
| `region.address_of(slot)` | `base + stride * slot`. Raises `IndexError` outside `[0, num_slots)`. |
| `region.as_tensor(dtype=torch.uint8)` | A strided `[num_slots, size // itemsize]` view; row `i` is page slot `i`. Accepts any subscript, including `-1`. |
| `region.slot_tensor(slot_id, dtype=torch.uint8)` | The bytes of one page slot, raising `IndexError` outside `[0, num_slots)`. The guarded form of `as_tensor(dtype)[slot_id]`. |

`size` is not necessarily `stride`: a region covers one run of adjacent buffers within a slot, and a
slot may hold several runs. For a model with uniform layer shapes the buffers coalesce into a single
region spanning the whole slot, which is the whole-page transfer. A group with more than one region
must be addressed region by region.

The addresses are device addresses, and they stay valid because every cache tier below GPU is
rejected at bring-up while a connector is attached. See [KV cache tiers](#kv-cache-tiers-are-gpu-only-under-a-connector).

##### KV cache tiers are GPU-only under a connector

A connector registers device addresses and holds them across iterations. Evicting a page to another
tier reassigns its GPU slot underneath the connector, so on `KVCacheManagerV2`:

* Setting `KvCacheConfig.host_cache_size` or `KvCacheConfig.disk_cache_size` above zero **fails at
  bring-up**, with a message naming both settings.
* Leaving `host_cache_size` unset **drops the host tier** rather than failing, with a log line. That
  tier is provisioned automatically only to give the `MAX_UTILIZATION` scheduler somewhere to spill
  to via suspend/resume, which a connector run does not use.
* `enable_kv_pool_rebalance` is **ignored** — startup and inference continue, and the rebalance
  simply never runs. Rebalance suspends every active request and runs a defragmenting migration that
  reassigns the same page slots a tier eviction would.

The practical consequence is that a KV-exhausted connector deployment on V2 has no secondary tier to
fall back on. The remedies are `kv_cache_config.max_tokens`,
`kv_cache_config.free_gpu_memory_fraction`, or lowering `max_num_tokens` to hand memory back to the
KV pool; the V2 scheduler's exhaustion error says so directly when a connector is attached.

##### Block reuse alongside the connector

Specify `KvCacheConfig.enable_block_reuse=True` alongside a connector on `KVCacheManagerV2`, or the
combination is rejected at start-up.

The two are not the same mechanism. `enable_block_reuse` governs TensorRT-LLM's own radix-tree
prefix reuse; a connector is a separate source of prefix KV and is not gated by that flag. On
`KVCacheManagerV2` the connector's prefix is therefore honoured whatever the flag says, and with
reuse off the restored KV is incorrect — prefill is skipped for the served range and generation
drifts within a few tokens. The pair is refused rather than allowed to produce wrong output
silently.

The check reads the value the manager resolved, not the one you passed. `enable_block_reuse` is
also turned off automatically for some quantization algorithms, some SM versions and hybrid linear
models, so this error can appear without the flag being set anywhere in your configuration. The
message names the setting; the fix is to make block reuse available for that deployment.

On `KVCacheManager` (V1) the same pair is **not** rejected, and it is not equivalent. V1 asks the
connector for a match and then schedules the whole prompt anyway, so the offer is never honoured:
the lookup, the reads and the device copies are performed and discarded, at no correctness cost but
at full latency cost. `RequestData.computed_position` is also reported negative in that case,
because the unhonoured offer is still subtracted from a position that never advanced — a connector
that derives block ordinals from that field will compute negative indices. Whether a connector
should serve a prefix at all when local reuse is disabled is an open design question; until it is
settled, treat V1 with `enable_block_reuse=False` as a configuration to avoid rather than one to
rely on.

##### Fields not populated on `KVCacheManagerV2`

Two `RequestData` fields are reported empty when the connector runs on `KVCacheManagerV2`.

| Field | On V2 | Consequence |
|---|---|---|
| `block_hashes` | always `[]` | V2 has no block-hash accessor on this path. Nothing in the runtime reads the field, and neither example connector uses it — both hash the token sequence themselves. A connector that keys its external store on `block_hashes` gets no key and therefore no hits and no saves; it does not mis-address a transfer. |
| `priorities` | always `None` | `KvCacheRetentionConfig` does not reach `KVCacheManagerV2` at all, so every page carries the default priority. A warning is logged the first time a request carrying a retention config is reported. The gap is wider than the connector: a retention config set on V2 has no effect either way. |

Both are gaps to be closed rather than intended differences, and neither is a regression: on V1 both
fields behave as they always have.

##### Blocks with no page

A block that has no page in a layer group is reported as `-1` (`BAD_PAGE_INDEX`) **in place**, not dropped from the list. This keeps each entry aligned with its block ordinal, so entry `i` always describes tokens `[i * tokens_per_block, (i+1) * tokens_per_block)` and an append-delta over successive calls stays valid.

That alignment is also why the list is not safe to index with directly: `-1` is a valid Python and PyTorch subscript, so it resolves to the *last* page slot of the pool rather than raising — a transfer against another request's KV. Two API points keep a page index from reaching device memory unchecked.

| | |
|---|---|
| `valid_page_slots(page_indices)` | Yields `(block_ordinal, page_slot)` for the entries that address a page. The ordinal is preserved, so the token range a page covers is still recoverable. |
| `region.slot_tensor(slot_id)` | The bytes of one page slot, raising `IndexError` on a slot outside `[0, num_slots)`. |

Build transfer targets with `valid_page_slots` and address them with `slot_tensor`. This covers `block_ids`, `cache_block_ids`, `RequestData.new_block_ids`, and both `*_by_layer_group` forms.

```python
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import valid_page_slots

for ordinal, slot in valid_page_slots(cache_block_ids):
    tokens = all_tokens[ordinal * tokens_per_block:(ordinal + 1) * tokens_per_block]
    store.put(self._key(tokens), region.slot_tensor(slot))
```

##### What a connector can persist under a sliding window

Under sliding-window attention, a connector can persist **at most `window_size` tokens per sequence**, not `prompt_len`.

The KV cache manager reclaims a block's page once the window has moved past it, so by the time `request_finished` runs there is no readable KV for anything older than the last `window_size` tokens. Those ordinals report no page (see [Blocks with no page](#blocks-with-no-page)), and the page slots offered to save from cover the live window only. A prefix-caching connector on such a model therefore caches a tail rather than a prefix, and the prefix it can serve back on a later request is bounded the same way.

This is a property of the cache, not of the connector: the blocks are gone whether or not a connector is attached. The same bound applies to the KV cache transceiver, which drops the same range before sending.

##### Serving a prefix on `KVCacheManagerV2`

A connector written against the V1 manager runs on `KVCacheManagerV2` unchanged for any model with a single attention window size. `KVCacheManagerV2` describes its pools rather than handing over one tensor, so it calls `register_kv_cache_layout` instead of `register_kv_caches` — but that method's default reconstructs the single-pool tensor, in the same `[num_blocks, num_layers, kv_factor, block_size]` shape and KV dtype, and forwards it to `register_kv_caches`. The same applies to the two block-id callbacks: their per-layer-group forms default to the flat ones when there is a single layer group.

Variable sliding-window attention is the case where that stops working, because the cache then allocates one pool per window size and a page index is scoped to a layer group. See [Running under VSWA](#running-under-vswa).

Both managers ask `get_num_new_matched_tokens` at the same point in the iteration: once the batch for the upcoming forward pass is final. On V1 that is inside `addSequence`, called from `KVCacheManager.prepare_resources`; on V2 it is `KVCacheManagerV2.prepare_resources` directly. A request that is asked is therefore a request that runs, and the connector can take ownership of remote blocks in the query and release it in `request_finished`.

Two differences are worth knowing when tuning a deployment.

* **The runtime may honour less than you offer.** V1 allocates KV for the whole prompt when the request's first chunk is scheduled, so an offer always fits. V2 allocates per context chunk, which is what lets chunked prefill bound its memory, so an offer reaching past the current chunk requires the runtime to grow the allocation and that can fail under pressure. The runtime then serves the part it can cover and computes the rest locally. The amount actually served is what `RequestData.computed_position` reflects; the unserved remainder needs no action from the connector beyond its usual `request_finished` cleanup.
* **The query is not part of the scheduler's budget.** The V2 scheduler sizes a request's chunk as if the connector will serve nothing, so a served prefix reduces the work in the forward pass but does not free budget for another request in the same iteration.

Specify `enable_block_reuse=True` alongside the connector for any of this to run on `KVCacheManagerV2`; see [Block reuse alongside the connector](#block-reuse-alongside-the-connector).

`get_num_new_matched_tokens` is called **at most once per KV allocation**. This is the precise form of the "once per request" rule, and it holds on both managers: if a request's KV cache is destroyed and the request is replayed -- which `MAX_UTILIZATION` does under memory pressure -- the replay asks again, because the pages the first answer described are gone.

**Deployment note.** Under V2 with a connector, a workload that was token-bound becomes KV-bound: the connector removes forward-pass tokens but its prefix still occupies GPU pages. Lowering `max_num_tokens` to hand memory back to the KV pool is usually the right adjustment, the opposite of the guidance for a connector-free deployment.

#### 2. Worker Interface (`KvCacheConnectorWorker`)

These methods run on all workers (GPU processes) and interact with the actual GPU data.

* **`register_kv_caches(self, kv_cache_tensor: torch.Tensor)`**
  * **Description**: Called at initialization. Provides the worker with the GPU KV cache tensors.
  * **Arguments**: `kv_cache_tensor` is the underlying storage tensor for the KV cache, shaped `[num_blocks, num_layers, kv_factor, block_size]`. Row `block_id` is that block's KV for every layer. Dimension 1 is indexed by model layer in ascending order, so the `layer_idx` passed to `wait_for_layer_load` and `save_kv_layer` indexes it directly — no mapping is supplied, and none is needed. This holds on both managers: V1 has a layer-to-pool-offset map but it is the identity for a single pool, and `KVCacheManagerV2` lays each pool out layer-major and ascending.

* **`register_kv_cache_layout(self, layout: KvCacheLayout)`**
  * **Description**: Called at initialization *instead of* `register_kv_caches` when the cache describes itself as pools rather than one tensor, which is what `KVCacheManagerV2` does. `KvCacheLayout` gives byte ranges per layer group: `layout.groups[g].regions[r]`, where the data for page slot `i` is at `region.base + region.stride * i` for `region.size` bytes, or `region.slot_tensor(i, dtype)`. Full attribute reference: [`KvCacheLayout` reference](#kvcachelayout-reference).
  * **Default**: reconstructs the single-pool tensor and forwards it to `register_kv_caches`, so a connector that does not override this needs no changes for any single-window model. It raises when the cache cannot be described as one tensor — several layer groups (VSWA), or several regions (block scales, layers of differing size).

* **`start_load_kv(self, stream: torch.cuda.Stream)`**
  * **Description**: Initiates the loading of KV blocks from the external source into the GPU memory.
  * **Arguments**: `stream` is the CUDA stream where the forward pass is executed in.

* **`wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream)`**
  * **Description**: A synchronization point. Ensures that the KV cache for a specific layer is fully loaded before the model attempts to perform the forward pass on that layer.

* **`save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream)`**
  * **Description**: Triggers the saving of a specific layer's KV cache.

* **`wait_for_save(self, stream: torch.cuda.Stream)`**
  * **Description**: A synchronization point to ensure all save operations are enqueued or completed.

* **`get_finished(self, finished_gen_req_ids, started_loading_req_ids) -> tuple[list[int], list[int]]`**
  * **Description**: Polled by the runtime to check the status of asynchronous operations.
  * **Returns**: Two lists of request IDs: those that have finished saving, and those that have finished loading.

## Example Implementation

The file `examples/llm-api/llm_kv_cache_connector.py` provides a reference implementation of a **Persistent KV Cache**.

### Overview

This example implements a file-system based KV cache.
1. **Save**: When a request finishes or needs to be swapped out, its KV blocks are saved to disk as `.pt` files.
2. **Load**: When a new request arrives with the same prompt prefix, the connector identifies the cached files and loads them back into GPU memory, skipping re-computation.

### Implementation Details

* **Metadata**: The example defines a `PersistentKvCacheConnectorMetadata` dataclass containing lists of `(file_path, block_id)` tuples for both loading and saving. This simple structure allows the Scheduler to tell the Worker exactly which file corresponds to which GPU block index.

* **Hashing Strategy**: The `PersistentKvCacheConnectorLeader` hashes the token sequence of a block to generate a unique filename (e.g., `hash_value.pt`). This acts as the lookup key.

* **Worker Logic**:
  * `start_load_kv`: Iterates through the load list provided in the metadata, loads the `.pt` file to CPU, and copies it to the specific `block_id` in the GPU tensor.
  * `wait_for_save`: Performs the reverse. It copies data from the GPU `block_id` to CPU and saves it to disk using `torch.save`.

### Limitations & Patterns

This example illustrates the API mechanics but has several limitations that make it unsuitable for high-performance production use without modification:

1. **Blocking I/O**: The example uses `torch.load` and `torch.save` synchronously. In a real implementation, these should be offloaded to a background thread or asynchronous I/O handler to avoid stalling the GPU.
2. **Simplified Block Matching**: The `get_num_new_matched_tokens` implementation in the example only matches full blocks. It does not handle partial cache hits.
3. **FileSystem Latency**: Storing one file per block can create high filesystem overhead.

### Usage

To run the example:

```bash
python examples/llm-api/llm_kv_cache_connector.py <model_path>
```

The script demonstrates:

1. Generating text for a prompt (First run).
2. Destroying the LLM instance.
3. Creating a new LLM instance with the same connector config.
4. Generating text for the same prompt (Second run).
5. Asserting that the outputs match, proving the state was correctly restored from the disk cache.
