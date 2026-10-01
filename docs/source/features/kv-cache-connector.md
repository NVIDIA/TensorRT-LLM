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
  * **Arguments**: `scheduler_output` contains information about new requests, blocks allocated, current request states, and the cumulative `RequestData.block_hashes` chain. `block_hashes` is read directly from each KV cache block's stored hash, which the KV cache manager commits as soon as a block becomes full, so the value matches the hash that KV cache events will subsequently emit for the same block. The chain only covers beam 0; the executor rejects `kv_connector_config` at startup when `max_beam_width > 1`, so connectors may assume beam-width-1 inputs.
  * **Returns**: An arbitrary metadata object (picklable) that describes the tasks for the workers. This object is broadcasted to all workers.

* **`get_num_new_matched_tokens(self, request: LlmRequest, num_computed_tokens: int) -> tuple[int, bool]`**
  * **Description**: Queries external KV after the compute batch is selected on the legacy path. Connectors that implement the complete reservation protocol use `reserve_prefix` during admission when prefix-aware scheduling and KV cache manager V2 are enabled.
  * **Returns**: A tuple `(num_tokens, is_async)`. `num_tokens` is the number of tokens found in the external cache. `is_async` indicates if the loading will happen asynchronously (background) or requires blocking.

* **`reserve_prefix(self, request: LlmRequest, num_computed_tokens: int, reservation_id: int) -> tuple[int, bool]`**
  * **Description**: Reserves an additional contiguous prefix beginning at `num_computed_tokens` during batch construction. A positive answer protects that source range against mutation and eviction. The query must start no KV transmission or destination writes.
  * **Returns**: `(additional_tokens, is_async)`. The runtime may accept a shorter, block-aligned range and keeps at least the final prompt token for local computation. `is_async=True` parks an accepted request until all workers finish its load.
  * **Identity**: The runtime assigns a fresh `reservation_id` to each attempt. Keep reservation state by this identity, including when the same request is retried.

* **`release_prefix_reservation(self, request: LlmRequest, reservation_id: int, start: int, end: int) -> None`**
  * **Description**: Releases the named reservation's protection for the absolute half-open token interval `[start, end)`. The runtime releases rejected or clipped portions before transmission and the accepted portion after all workers report completion. Overlapping reservations must keep their own protection.
  * **Lifetime**: This callback never asks the connector to interrupt an active transfer. Client cancellation after dispatch drains the load before releasing its source and destination resources.

* **`request_finished(self, request: LlmRequest, cache_block_ids: list[int]) -> bool`**
  * **Description**: Called when a request completes generation.
  * **Returns**: A boolean indicating if an asynchronous save operation is underway. If `True`, the system waits for the operation to complete before releasing the KV cache blocks.
  * **Note**: under sliding-window attention `cache_block_ids` covers the live window, not the whole prompt. See [What a connector can persist under a sliding window](#what-a-connector-can-persist-under-a-sliding-window).

* **`update_state_after_alloc(self, request: LlmRequest, block_ids: list[int])`**
  * **Description**: a callback to update internal state after KV cache blocks have been allocated for the prefill.
  * **Note**: with chunked prefill, `block_ids` covers only the blocks allocated for the first chunk. The remaining blocks arrive as append-deltas in `RequestData.new_block_ids` on later calls to `build_connector_meta`, on the entries under `scheduler_output.cached_requests`. A connector that treats this callback as its only source of block ids will under-plan. Read both lists:

    ```python
    def build_connector_meta(self, scheduler_output):
        for req in scheduler_output.new_requests:      # the first chunk
            self._plan(req.request_id, req.new_block_ids)
        for req in scheduler_output.cached_requests:   # every later chunk
            self._plan(req.request_id, req.new_block_ids)
    ```

    Both example connectors walk `new_requests` only, so neither one demonstrates this.

* **`cancel_load(self, request: LlmRequest, start: int, end: int)`**
  * **Description**: Optional, with a no-op default. Tells the connector that the runtime will not consume KV it offered from `get_num_new_matched_tokens` for prompt tokens `[start, end)`, so any ownership taken for that range can be released. Offsets are absolute prompt positions, on the same scale as `num_computed_tokens`.
  * **When it fires**: only on `KVCacheManagerV2`, which asks during a speculative scheduling pass and resolves the answer later. Two things can happen in between, and both are reported here: the runtime may fail to allocate pages to cover the offer, in which case the request falls back to computing the prefix locally; or the request may be cancelled, time out or fail before it ever reaches a batch, in which case the whole offer is released. A third case, the local cache overtaking part of the offer because another request committed the same prefix, is handled by the same callback but cannot arise today, since a request's local match is fixed when its cache is created and only its own completed forward passes extend it.
  * **Caveat**: best-effort. For a synchronous load nothing has been transferred yet, so cancelling is exact. For `is_async=True` the transfer necessarily started inside `get_num_new_matched_tokens`, so it may already be in flight.
  * **Why V2 needs it**: V1 answers `get_num_new_matched_tokens` from C++ while the block manager holds its radix-tree mutex, so the local match and the query are atomic and the answer is consumed immediately. V2 has no such mutex, and its scheduling pass is speculative: a prepared request can still be dropped at the token budget, at resize, at multimodal alignment or at cross attention, and retried in a later iteration.

    `get_num_new_matched_tokens` is still called **exactly once per request** on both managers, so a request that is asked and then deferred is not asked again when it comes back. What differs is that on V2 the runtime may resolve the answer in a later iteration than the one it asked in, and may by then be unable to honour part or all of it. That is what `cancel_load` reports.

* **`request_reset(self, request: LlmRequest)`**
  * **Description**: Optional, with a no-op default. Tells the connector that the request's allocation has been released and the request will run again from the start, unlike `request_finished`, which ends it. Any per-request state the connector keeps in page indices describes pages that are now free for another request to own, so it has to be dropped or rebuilt here.
  * **When it fires**: on `KVCacheManagerV2`, from `free_resources`, which is the path a rollback or a failed admission takes. Direct preemption calls `request_finished` instead, so a connector that keys state by request id sees exactly one of the two.
  * **Why it matters**: without it a replayed request's new pages are appended to the record of the pages it held before, so the block ordinals a later save reads from point at slots a different request now owns, and the connector writes that request's KV into the store under this one's keys.

* **`update_state_after_alloc_by_layer_group(self, request: LlmRequest, block_ids_by_layer_group: list[list[int]])`**
* **`request_finished_by_layer_group(self, request: LlmRequest, cache_block_ids_by_layer_group: list[list[int]]) -> bool`**
  * **Description**: the per-layer-group forms of the two callbacks above, indexed by layer group id. Entry `[g][i]` is the page slot of block ordinal `i` in layer group `g`.
  * **When they are called**: whenever the KV cache reports page indices per layer group. A page index is scoped to its group — one group per attention window size — so a cache with more than one group can be described no other way, and the flat `block_ids` / `cache_block_ids` are empty there. With a single layer group the flat lists carry that group's indices as well, so a connector that implements only the flat forms keeps working on those models.
  * **Which form to implement**: implement exactly one complete set.

    | Set | Models it covers |
    |---|---|
    | per-layer-group | every model, VSWA included |
    | flat | non-VSWA, non-hybrid only |

    A set is complete when both of its methods are defined. Mixing the two — one method from each — is rejected during executor bring-up, naming the method that is missing. An existing flat connector needs no change on the models it already covers: with a single layer group the base per-layer-group implementation folds back to the flat call. Hybrid / linear-attention models are not enabled for the connector yet; the per-layer-group set is the shape their cache will need. See [Running under VSWA](#running-under-vswa).

##### Running under VSWA

Under variable sliding-window attention the KV cache allocates one pool per attention window size, and a page index only means something inside its own layer group. A single tensor and a single flat block list cannot describe that, so three methods have to be implemented together:

| Method | Replaces |
|---|---|
| `KvCacheConnectorWorker.register_kv_cache_layout` | `register_kv_caches` |
| `KvCacheConnectorScheduler.update_state_after_alloc_by_layer_group` | `update_state_after_alloc` |
| `KvCacheConnectorScheduler.request_finished_by_layer_group` | `request_finished` |

Implementing the per-layer-group form of a pair is enough — the flat method it replaces does not also have to be defined.

All three are checked during executor bring-up, before any request is admitted. `register_kv_cache_layout` refuses there, naming the group and region counts it could not describe, and the two scheduler methods are checked alongside it. Nothing is deferred to the first request, so a partial implementation costs a start-up failure rather than one after the model is loaded.

`examples/llm-api/llm_kv_cache_connector_vswa.py` is a worked connector for this case. A VSWA connector has to do five things:

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
tier reassigns its GPU slot underneath the connector, so with a connector attached:

* Setting `KvCacheConfig.host_cache_size` or `KvCacheConfig.disk_cache_size` above zero **fails at
  bring-up**, with a message naming both settings.
* Leaving `host_cache_size` unset **drops the host tier** rather than failing, with a log line. That
  tier is provisioned automatically only to give the `MAX_UTILIZATION` scheduler somewhere to spill
  to via suspend/resume, which a connector run does not use.
* `enable_kv_pool_rebalance` is **ignored** — startup and inference continue, and the rebalance
  simply never runs. Rebalance suspends every active request and runs a defragmenting migration that
  reassigns the same page slots a tier eviction would.

The practical consequence is that a KV-exhausted connector deployment has no secondary tier to
fall back on. The remedies are `kv_cache_config.max_tokens`,
`kv_cache_config.free_gpu_memory_fraction`, or lowering `max_num_tokens` to hand memory back to the
KV pool; the scheduler's exhaustion error says so directly when a connector is attached.

##### Block reuse alongside the connector

* Specify `KvCacheConfig.enable_block_reuse=True` alongside a connector. Without it the connector's
  prefix is never honoured: either the combination is rejected at start-up, or the lookup, the reads
  and the device copies are performed and discarded, at no correctness cost but at full latency
  cost.
* The start-up check reads the value the cache resolved, not the one you passed: some quantization
  algorithms, some SM versions and hybrid linear models turn block reuse off on their own, so this
  error can appear without the flag being set anywhere in your configuration.

##### A page slot must not be reassigned underneath the connector

The connector holds page indices across iterations, and `RequestData` reports only the pages appended
since the last call. Anything that hands a slot the connector already knows about to a different
request therefore requires resetting the connector state before replay. The runtime applies the
following configuration limits at bring-up.

| Configuration | Mechanism |
|---|---|
| Speculative decoding | Rejected draft tokens shrink a request's page list, and the freed slot goes to whichever request allocates next. The connector is never told the tail block moved. |
| A capacity scheduler policy other than `GUARANTEED_NO_EVICT` with KV cache manager V1 | V1 does not reset the connector block delta when a request is destroyed and replayed. V2 resets that state and permits replay; an active connector load retains its allocation until completion. |
| A host or disk cache tier | Tier eviction reassigns the GPU slot. See [KV cache tiers are GPU-only under a connector](#kv-cache-tiers-are-gpu-only-under-a-connector). |

The exact set the runtime refuses depends on your cache configuration; the bring-up error is
authoritative. Speculative decoding is unsupported with a connector wherever it is not refused —
the same page-list shrink happens there.

##### `RequestData` fields that may not be populated

Two `RequestData` fields can be reported empty depending on the cache configuration. A connector must tolerate both.

| Field | When empty | Consequence |
|---|---|---|
| `block_hashes` | `[]` | No block-hash accessor exists on this path. Nothing in the runtime reads the field, and neither example connector uses it — both hash the token sequence themselves. A connector that keys its external store on `block_hashes` gets no key and therefore no hits and no saves; it does not mis-address a transfer. |
| `priorities` | `None` | `KvCacheRetentionConfig` is not honoured, so every page carries the default priority. A warning is logged the first time a request carrying a retention config is reported. The gap is wider than the connector: the retention config has no effect either way in that configuration. |

Both are gaps to be closed rather than intended behaviour.

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

Alignment is not the same as stability. Under a sliding window an entry that was reported with a page reads back as `-1` once the window passes its block, and the delta — which carries only the ordinals appended since the last call — does not restate it. Save a block when its tokens complete rather than deferring: a completed block is far inside the window for any usable window size, whereas a deferred save can reach a slot the cache has already reclaimed.

##### What a connector can persist under a sliding window

Under sliding-window attention, a connector can persist **at most `window_size` tokens per sequence**, not `prompt_len`.

The KV cache manager reclaims a block's page once the window has moved past it, so by the time `request_finished` runs there is no readable KV for anything older than the last `window_size` tokens. Those ordinals report no page (see [Blocks with no page](#blocks-with-no-page)), and the page slots offered to save from cover the live window only. A prefix-caching connector on such a model therefore caches a tail rather than a prefix, and the prefix it can serve back on a later request is bounded the same way.

This is a property of the cache, not of the connector: the blocks are gone whether or not a connector is attached. The same bound applies to the KV cache transceiver, which drops the same range before sending.

##### Serving a prefix

A connector that implements only the flat callbacks and `register_kv_caches` runs unchanged on any model with a single *non-sliding* attention window. Where the cache describes itself as pools rather than one tensor it calls `register_kv_cache_layout` instead — but that method's default reconstructs the single-pool tensor, in the same `[num_blocks, num_layers, kv_factor, block_size]` shape and KV dtype, and forwards it to `register_kv_caches`. The same applies to the two block-id callbacks: their per-layer-group forms default to the flat ones when there is a single layer group.

Variable sliding-window attention is the case where that stops working, because the cache then allocates one pool per window size and a page index is scoped to a layer group. See [Running under VSWA](#running-under-vswa).

A model whose layers all share one sliding window stays a single layer group, so the flat callbacks still apply and such a connector is not refused. What differs is that the callbacks cover the **live window only**. Blocks the window has passed report `-1` (`BAD_PAGE_INDEX`) in place — the list stays aligned to block ordinals, so entry `i` still describes tokens `[i * tokens_per_block, (i+1) * tokens_per_block)`, but the up-front blocks carry no page and are not available to load into or save from. Filter with `valid_page_slots`, described in [Blocks with no page](#blocks-with-no-page); without it a `-1` resolves to the last page slot of the pool. A warning naming the window size is logged at start-up when a flat-only connector is attached to such a model.

##### Reserving a prefix during admission

For connector implementers, scheduler participation requires three optional methods together:
`reserve_prefix` and `release_prefix_reservation` on the scheduler, and
`get_finished_prefix_loads` on the worker. Bring-up rejects a partial implementation. KV cache
manager V2 enables this protocol when `SchedulerConfig.enable_prefix_aware_scheduling=True`.
Connectors using the existing methods retain the final-batch query path.

The scheduler charges compute tokens after the local and reserved external prefix. The complete
prefix still needs destination KV pages, so a token-budget hit can remain limited by KV capacity.
After the final admission checks, `SchedulerOutput.prefix_loads` authorizes the exact loads:

| `PrefixLoad` field | Meaning |
|---|---|
| `reservation_id` | The identity supplied to `reserve_prefix`, also returned on completion. |
| `request_id` | The request owning the allocation. |
| `start`, `end` | Absolute half-open token interval to load. |
| `is_async` | Whether the request waits outside the compute batch. |
| `block_ids_by_layer_group` | Complete destination page lists, indexed by group and block ordinal. Filter entries through `valid_page_slots`. |
| `tokens`, `cache_salt` | Request tokens and cache-key isolation salt. |

Build transfers from this collection. A confirmed async load appears here even when it is absent
from `new_requests` and `cached_requests`, and even when the compute batch is empty. Those two
lists continue to describe computation and its token/block deltas. The reservation itself supplies
no permission to write destination memory.

| Event | Connector and runtime behavior |
|---|---|
| Candidate queried | Connector protects the promised source; no transmission starts. |
| Candidate rejected or offer shortened | Runtime releases the unused source interval. A later attempt gets a new reservation identity. |
| Accepted load dispatched | Connector starts only the confirmed interval; runtime retains its destination allocation. |
| Client cancels during transmission | Runtime keeps the allocation while the connector completes the load. |
| Every worker completes | Runtime releases the source reservation, resumes a live request, or finalizes a cancelled request and frees its allocation. |

Each worker reports completion after its destination writes and source reads finish. Reports are
sent to the leader without a collective. Once every worker has reported, an ordered control item
retires the load on all ranks before scheduling. This releases the source reservation and allows
an async request to rejoin the compute batch, or a cancelled request to release its allocation.
The runtime continues polling when no forward pass is scheduled. A transfer that never completes
retains its resources; elapsed time alone cannot make pages safe to reuse.

For a load admitted with computation, `wait_for_layer_load` establishes the dependency on that
worker's transfer stream. Inference does not wait for the retirement control item.

##### Legacy final-batch queries

`get_num_new_matched_tokens` is called at most once per KV allocation after batch selection.
Its offer can reduce computation for that request, but does not free token budget for another
request in the same iteration. With chunked prefill, allocation growth can limit the served prefix;
`RequestData.computed_position` reflects the load interval that the runtime honors.

Destroying an allocation clears its connector state, so replay queries again. Specify
`enable_block_reuse=True` alongside the connector; see
[Block reuse alongside the connector](#block-reuse-alongside-the-connector).

**Deployment note.** Under a connector, a workload that was token-bound becomes KV-bound: the connector removes forward-pass tokens but its prefix still occupies GPU pages. Lowering `max_num_tokens` to hand memory back to the KV pool is usually the right adjustment, the opposite of the guidance for a connector-free deployment.

#### 2. Worker Interface (`KvCacheConnectorWorker`)

These methods run on all workers (GPU processes) and interact with the actual GPU data.

* **`register_kv_caches(self, kv_cache_tensor: torch.Tensor)`**
  * **Description**: Called at initialization. Provides the worker with the GPU KV cache tensors.
  * **Arguments**: `kv_cache_tensor` is the underlying storage tensor for the KV cache, shaped `[num_blocks, num_layers, kv_factor, block_size]`. Row `block_id` is that block's KV for every layer. Dimension 1 is indexed by model layer in ascending order, so the `layer_idx` passed to `wait_for_layer_load` and `save_kv_layer` indexes it directly — no mapping is supplied, and none is needed.

* **`register_kv_cache_layout(self, layout: KvCacheLayout)`**
  * **Description**: Called at initialization *instead of* `register_kv_caches` when the cache describes itself as pools rather than one tensor. `KvCacheLayout` gives byte ranges per layer group: `layout.groups[g].regions[r]`, where the data for page slot `i` is at `region.base + region.stride * i` for `region.size` bytes, or `region.slot_tensor(i, dtype)`. Full attribute reference: [`KvCacheLayout` reference](#kvcachelayout-reference).
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

* **`get_finished_prefix_loads(self) -> list[int]`**
  * **Description**: Reports locally completed reservation identities for loads dispatched through `SchedulerOutput.prefix_loads`, including synchronous loads. Completion means this worker has finished all reads and writes and established the required CUDA stream visibility. This method must not wait for other workers. The runtime collects reports asynchronously and distributes retirement decisions before scheduling; parked async requests resume and shared resources are released only after every worker has reported.
  * **Compatibility**: Legacy request-ID load and save completions continue through `get_finished`.

## Built-in Connectors

Named presets can be selected without naming a module or class:

```python
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig

kv_connector_config = KvCacheConnectorConfig(connector="mooncake-store")
```

The available presets are `lmcache`, `lmcache-mp`, `kvbm` and `mooncake-store`. The first three are external packages; `mooncake-store` ships with TensorRT-LLM and is described below.

### Mooncake distributed store (`mooncake-store`)

Publishes KV pages into a [Mooncake](https://github.com/kvcache-ai/Mooncake) store, a shared CPU memory pool addressed by content, so a prefix computed by one engine can be replayed by another. Regular block reuse cannot do this, because it never leaves the instance that computed the prefix.

This is a **different component** from the Mooncake transfer engine that the C++ cache transceiver uses for disaggregated prefill/decode handoff. That moves KV point to point between two known peers; this publishes pages into a pool that any peer can read. The two compose: a context server can write pages into the store and still hand off to a generation server over NIXL.

#### Requirements

* `KVCacheManagerV2` (`kv_cache_config.use_kv_cache_manager_v2: true`), since that is the manager that can describe its pools through `register_kv_cache_layout`. Not needed for `role: capacity`, which describes no pools.
* The Mooncake Python bindings, which `tensorrt-llm` pulls in as `mooncake-transfer-engine-cuda13`, so reinstall that package if this environment dropped it. The source build of the C++ transfer engine does not provide them.
* A reachable Mooncake master (and metadata server, unless using `P2PHANDSHAKE`), run as `trtllm-serve mooncake_master`. See the [Mooncake documentation](https://kvcache-ai.github.io/Mooncake/).

The connector also forces `kv_cache_config.host_cache_size` and `disk_cache_size` to 0, overriding any configured value with a warning. A page evicted to another tier has its GPU slot reassigned, which would invalidate the addresses registered with the store, and the pool is already the deployment's offload tier. Give that memory to `segment_size` instead, where every server on the node can reuse what any of them stored.

#### The master is infrastructure

One master owns a pool, and it is not part of any engine. Run it before the servers that join:

```bash
trtllm-serve mooncake_master --pool_file /shared/pool.json
```

It publishes a **manifest** at that path describing the pool it owns, and removes it on exit so a stale address is never dialed:

```json
{
  "master_server_address": "10.0.0.1:50051",
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "rdma",
  "namespace": "trtllm",
  "eviction_ratio": 0.05,
  "metrics_port": 9004
}
```

The manifest exists because the settings in it have to be identical for every participant, and restating them in each worker config is how they come to differ. Naming a file rather than an address also closes a gap that a scheduler opens: the master's host is unknown when the configs are written, so the configs name the path instead, and a server reading it waits for it. The master and the servers can therefore be started in any order.

#### Configuration

Each server names the pool and says what it contributes and what it does:

```yaml
kv_connector_config:
  connector: mooncake-store
  mooncake_store:
    pool: file:///shared/pool.json
    role: both            # both | producer | consumer | capacity
    segment_size: 160GiB  # per rank
    model_key: minimax-m3-fp4
```

`trtllm-serve` then reads the manifest, adds this server's settings and this node's detected RDMA devices, renders the Mooncake client config, and exports `MOONCAKE_CONFIG_PATH` before the ranks that open store handles are spawned.

`pool` and `model_key` are required. `pool` also accepts a bare `host:port` for joining a master run without this CLI, in which case the pool-wide settings take their defaults and keeping them consistent is the deployment's problem.

`model_key` is what the pool's keys identify this checkpoint by. Two engines share cache only when they agree on it, and two that disagree about which checkpoint it names would read each other's pages as their own, so it has no default. Give it a value that separates this checkpoint from any other an engine on the same pool might load; a model path is a poor choice, since `org-a/model` and `org-b/model` share a directory name while disagreeing on what the pages mean. Every role needs it, `capacity` included, because each rank resolves it while starting up.

Sizes are written as binary units (`160GiB`) or byte counts. `GB` and friends are **refused** in anything both engines may read: this parser scales them by 1000 and vLLM's Mooncake parser by 1024, so `80GB` would name two different segments. The rendered client config always holds resolved integers for the same reason.

`master_timeout` (default 60s) is how long a server waits for the manifest to appear and the master to accept connections. Without that wait, a master that is not there yet fails inside every rank after the model has loaded. `run_dir` keeps the generated client config and the segment records, which are otherwise in a temporary directory removed at shutdown. The master command takes `--timeout` for the same wait and `--binary` to name the executable.

There is no environment override for any of this. `MOONCAKE_CONFIG_PATH` is the one variable the connector reads, and it belongs to Mooncake rather than to TensorRT-LLM.

#### Servers whose ranks the launcher starts

Provisioning happens in the server process and reaches the ranks that open store handles by exporting `MOONCAKE_CONFIG_PATH` for them to inherit. That holds when the LLM constructor spawns them. It does not when the launcher starts one task per rank, as `trtllm-llmapi-launch` under a scheduler does, because those ranks were already running.

Setting `run_dir` covers that case: the rendered config is read back from `<run_dir>/mooncake.json` by any rank that inherited no path, so every rank of a multi-GPU server joins the pool its own leader provisioned. It has to be a directory all of that server's ranks see, which under a scheduler means a shared filesystem, and **one per server rather than one per job**, since two servers sharing it render one client config between them:

```yaml
mooncake_store:
  pool: file:///shared/run/pool.json
  model_key: minimax-m3-fp4
  run_dir: /shared/run/ctx0
```

Because it comes from the worker config rather than the environment, every rank of the server reads the same value without the launch script having to export anything. Without it, a rank that inherited nothing fails during bringup naming `MOONCAKE_CONFIG_PATH`, rather than serving without a store.

#### Reading bringup in the log

Everything the pool is assembled from is logged under the `mooncake-store:` prefix before the model loads, because a pool that came up wrong is otherwise visible only as a low hit rate hours later. In order: the run directory, the manifest that was read, the rendered client config in full, the segment each rank will contribute in both GiB and bytes, and the total that implies for this node. A capacity-only rank says so explicitly, because every other sign of a working connector is absent by design there and their absence otherwise reads as a broken deployment.

Both waits report progress every five seconds, since waiting for a master in another job step is normal and indistinguishable from a hang if it is silent. A master that dies during startup has the tail of its own log quoted in the failure, which is where the reason, a port in use or a bad flag, actually is.

#### Pool capacity, and the `capacity` role

Capacity comes from processes that open a store handle, and every rank that joins contributes `segment_size`. Pool capacity is therefore the sum over participating ranks, and grows with the deployment's parallelism by design.

In a disaggregated deployment you want the generation side in that sum. Its nodes hold most of the deployment's host DRAM, and a pool built only from context ranks is prefill's DRAM caching prefill's GPUs, which is close to what a native host tier would have done for those ranks alone. But a generation engine has no use for the pool's contents: it receives prompt KV over the cache transceiver, so its lookups would nearly all miss.

`role: capacity` is that combination — contribute memory, drive no traffic:

```yaml
# generation server
kv_connector_config:
  connector: mooncake-store
  mooncake_store:
    pool: file:///shared/pool.json
    role: capacity
    segment_size: 160GiB   # the same per-rank figure as the context servers
    model_key: minimax-m3-fp4
```

Such a rank opens its handle, mounts its segment, and stops. It runs no prefix lookup, issues no load or save, starts no background save thread, and **registers no KV cache with Mooncake** — which also means it needs no GPUDirect RDMA, so a host whose HCA cannot pin GPU pages can still lend memory.

Because it moves no KV, the restrictions that exist to protect registered page addresses do not apply to it. Such a server keeps its capacity scheduler policy, its partial block reuse and its per-layer forward hooks exactly as configured, which is what lets the generation side stay on `MAX_UTILIZATION` while lending the pool memory.

Its native cache tiers are the exception, and are turned off as they are for every other role. Not because of page addresses, which a capacity rank does not register, but because a local tier would compete for the DRAM that rank lent the pool. With no tier to spill to, the V2 scheduler reclaims pages by preemption rather than suspension.

Contribution is per rank, which is the right interface: capacity then tracks the hardware in the deployment. Host DRAM, however, is a per-node limit, and the ranks sharing a node each claim the segment independently. A tensor-parallel-4 generation server on a 4-GPU node claims `4 x segment_size`; under attention DP with 8 owners it claims eight times. The connector checks this at startup and refuses a segment the node cannot afford, because the failure otherwise is not an allocation error but the OOM killer arriving minutes later, while weights are still loading, naming no cause.

Keep `segment_size` the same on every server. Pool capacity is meant to be uniform per rank, but each server only ever sees its own value, so a mismatch is invisible from every vantage point — except the run's report, which names the distinct sizes it finds.

#### What the pool actually was

Every rank records the segment it mounted under `<run_dir>/segments/`, and the run's capacity is read back from those records rather than from log lines. The report is given the pool's own directory — the one holding the manifest every participant named — and gathers the records from the tree beneath it, so a job whose servers each have a run directory of their own still totals up:

```bash
trtllm-serve mooncake_pool_report --run_dir /shared/run/$SLURM_JOB_ID
```

```text
master        : 10.66.5.9:50051
ranks         : 24 across 7 host(s)
capacity      : 3840.0 GiB
  role both     :   4 rank(s),    640.0 GiB   16.7%
  role capacity :  20 rank(s),   3200.0 GiB   83.3%
per rank      : 160.0 GiB, uniform
```

Reading declared facts rather than log messages is deliberate. Recovering the same figures by grepping worker logs ties the report to the wording of one provisioning mechanism, and when the mechanism changes the report does not fail — it silently attributes the bytes to the wrong side. With the master's log available the report also joins its `allocation_succeeded` lines against these records, which is what answers the question the pool exists for: did prefill's writes reach memory on the other side of the deployment, or only its own nodes?

#### Pointing at a pool directly

Topology can equally come from a JSON file named by `MOONCAKE_CONFIG_PATH`, using the same schema as the vLLM Mooncake store connector so one deployment can point both engines at the same pool:

```json
{
  "metadata_server": "http://127.0.0.1:8080/metadata",
  "master_server_address": "127.0.0.1:50051",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "global_segment_size": 34359738368,
  "role": "both"
}
```

Only `master_server_address` is required. `metadata_server` may be left out, in which case it is `P2PHANDSHAKE`, Mooncake's peer-to-peer handshake, which is what the manifest defaults to as well; the example above names a metadata service instead.

An inherited `MOONCAKE_CONFIG_PATH` wins over `mooncake_store` and is logged as doing so, so an orchestrator that already provisions the pool keeps working unchanged.

`role` and `stage_through_host` are read from that file too, so a hand-written config controls them the same way `mooncake_store` does.

#### Partial block reuse is forced off

`kv_cache_config.enable_partial_reuse` is set to `false` when this connector is configured for traffic, with a warning, whether or not it was requested explicitly. It defaults to `true`, so most deployments will see that warning. A `role: capacity` server consults the store for nothing and keeps partial reuse.

The store is addressed by whole blocks. The connector is handed the device match as `num_computed_tokens` and offers only blocks beyond it, but it can resume only from a block boundary, so when the device match ends mid-block it declines the lookup and the store is not consulted at all. Partial reuse is precisely what puts the match off a boundary, so it trades part of one block of device reuse for every stored block of the remaining prefix. Measured on MiniMax-M3, leaving it enabled declined 97.2% of lookups and left actual prompt cache read at 35% against a 96% ceiling; forcing it off raised that to 94% and roughly doubled throughput.

#### How it keys pages

`KVCacheManagerV2` reports `RequestData.block_hashes` empty, so the connector derives block identity itself: a blake2b chain where each block's hash covers its own tokens *and* every token before it, seeded by the request's `cache_salt`. A key is `<prefix>/<model>/w<world size>r<rank>/lg<layer group>/t<tokens per block>b<bytes per page>/<block hash>`. The namespace pins down everything that would make the stored bytes mean something different, so a mismatched shard count, layer group or page geometry reads as a cache miss rather than as garbage.

The value for one key is the concatenation of that layer group's regions for one page slot, handed to Mooncake's multi-buffer batch APIs as a list of `(address, size)` pairs.

#### Transfer behavior

* **Loads are synchronous**, performed in `start_load_kv` before the forward pass. A failed load raises: the runtime has already counted those tokens as computed, so a partial load is a wrong answer rather than a slow one.
* **Saves are asynchronous**, handed to a background thread behind a CUDA event recorded on the forward stream. The pages are only complete once the pass that wrote them retires, and blocking the executor loop on an RDMA write is the cost the store exists to avoid. The leader reports such requests as saving asynchronously, so their pages stay pinned until `get_finished` confirms the writes landed. A dropped save is logged rather than raised, since it only costs a future cache miss.
* Pages the store already holds are skipped, so several ranks or instances converging on the same prefix write it once.

#### Unsupported configurations

These are rejected at startup, before any request is admitted:

| Configuration | Reason |
|---|---|
| Context parallelism | A rank holds a slice of the sequence rather than whole blocks of it, so one key would name different bytes on different ranks. |
| Sliding-window attention / VSWA | A page's validity depends on where the window sits, which is a property of the request that read it rather than of the tokens it holds. |
| MiniMax-M3 with `sparse_disable_index_value: false` | The index-V cache is a plain tensor outside the paged pools, so a replayed prefix would pair stored index-K with stale index-V. Disaggregated serving applies the same restriction. |
| Pipeline parallelism | Untested rather than unsound. Use tensor parallelism. |
| `KVCacheManagerV1` | Identity here is a per-layer-group hash chain; V1 supplies real block hashes over a single flat block space. Allowed under `role: capacity`, which addresses no pages. |
| `segment_size` a node cannot afford | `ranks_on_node x segment_size` against available host memory. Rejecting it here turns an OOM kill during weight loading into a startup error. |

Beam search, attention data parallelism and Mamba caches are rejected for all connectors by the executor, as are non-GPU cache tiers for any connector that registers pages.

#### Example

`examples/llm-api/configs/trtllm_mooncake_store_connector_extra.yaml` is a starting point for `trtllm-serve`.

## Example Implementation

The file `examples/llm-api/llm_kv_cache_connector.py` provides a reference implementation of a **Persistent KV Cache**.

### Overview

This example implements a file-system based KV cache.
1. **Save**: After prefill, complete computed KV blocks are saved to disk as immutable `.pt` files.
2. **Load**: When a new request arrives with the same prompt prefix, the connector identifies the cached files and loads them back into GPU memory, skipping re-computation.

### Implementation Details

* **Metadata**: `PersistentKvCacheConnectorMetadata` carries `(file_path, block_id)` load/save targets and accepted reservation identities. The worker returns those identities after the copies finish.

* **Source protection**: Saves publish complete files atomically and never overwrite an existing inode. `reserve_prefix` creates private hard links to these immutable files without reading KV data. Releasing a range removes only its reservation links, so another reservation keeps its source available. External cache management must preserve this immutability: remove or replace a path rather than writing into a published file.

* **Hashing Strategy**: `PersistentKvCacheConnectorLeader` hashes the entire token prefix through each block, together with `cache_salt`, using SHA-256. This distinguishes identical blocks reached through different preceding tokens and remains stable across Python processes. Use separate cache directories for different models and KV layouts.

* **Worker Logic**:
  * `start_load_kv`: Reads only accepted load targets, copies their `.pt` data to GPU with blocking copies, and records their reservation identities for `get_finished_prefix_loads`.
  * `wait_for_save`: Performs the reverse. It copies data from the GPU `block_id` to CPU and saves it to disk using `torch.save`.

### Limitations & Patterns

This example illustrates the API mechanics but has several limitations that make it unsuitable for high-performance production use without modification:

1. **Blocking I/O**: The example uses `torch.load` and `torch.save` synchronously. In a real implementation, these should be offloaded to a background thread or asynchronous I/O handler to avoid stalling the GPU.
2. **Simplified Block Matching**: The example matches and loads complete blocks, requires one layer group, and saves only the first context chunk. It does not demonstrate chunked-prefill persistence.
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
