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

* **Scheduler (Leader)**: Responsible for orchestration. It decides *what* needs to be loaded or saved and builds metadata instructions. With tensor parallelism it runs on rank 0. With attention data parallelism (ADP), each rank runs an owner-local scheduler adapter.
* **Worker**: Responsible for execution. It receives metadata from the scheduler and performs the actual data transfers (loading/saving) on the KV cache tensors. It runs on all ranks.

### API Reference

To implement a custom connector, you must subclass `KvCacheConnectorScheduler` and `KvCacheConnectorWorker`.

#### 1. Scheduler (Leader) Interface (`KvCacheConnectorScheduler`)

These methods run on the leader process for TP, or on each request-owning rank for ADP.

* **`build_connector_meta(self, scheduler_output: SchedulerOutput) -> object`**
  * **Description**: The core orchestration method. Called during the scheduling phase. It examines the current requests and decides which blocks need to be loaded from or saved to the external store.
  * **Arguments**: `scheduler_output` contains information about new requests, blocks allocated, current request states, and the cumulative `RequestData.block_hashes` chain. `block_hashes` is read directly from each KV cache block's stored hash, which the KV cache manager commits as soon as a block becomes full, so the value matches the hash that KV cache events will subsequently emit for the same block. The chain only covers beam 0; the executor rejects `kv_connector_config` at startup when `max_beam_width > 1`, so connectors may assume beam-width-1 inputs.
  * **Returns**: An arbitrary metadata object (picklable) that describes the tasks for the workers. Under TP it is broadcast to all workers. Under ADP it is bound only to the local worker; `scheduler_output.attention_dp_rank` identifies the owner of its block IDs.

* **`get_num_new_matched_tokens(self, request: LlmRequest, num_computed_tokens: int) -> tuple[int, bool]`**
  * **Description**: Called when a new request arrives. It checks to see if any KV cache can be loaded from an external KV store.
  * **Returns**: A tuple `(num_tokens, is_async)`. `num_tokens` is the number of tokens found in the external cache. `is_async` indicates if the loading will happen asynchronously (background) or requires blocking.

* **`request_finished(self, request: LlmRequest, cache_block_ids: list[int]) -> bool`**
  * **Description**: Called when a request completes generation.
  * **Returns**: A boolean indicating if an asynchronous save operation is underway. If `True`, the system waits for the operation to complete before releasing the KV cache blocks.

* **`request_reset(self, request: LlmRequest)`**
  * **Description**: Optional, with a no-op default. Tells the connector that the request's allocation has been released and the request will run again from the start, unlike `request_finished`, which ends it. Any per-request state the connector keeps in page indices describes pages that are now free for another request to own, so it has to be dropped or rebuilt here.
  * **When it fires**: on `KVCacheManagerV2`, from `free_resources`, which is the path a rollback or a failed admission takes. Direct preemption calls `request_finished` instead, so a connector that keys state by request id sees exactly one of the two.
  * **Why it matters**: without it a replayed request's new pages are appended to the record of the pages it held before, so the block ordinals a later save reads from point at slots a different request now owns, and the connector writes that request's KV into the store under this one's keys.

* **`update_state_after_alloc(self, request: LlmRequest, block_ids: list[int])`**
  * **Description**: a callback to update internal state after KV cache blocks have been allocated for the prefill.
  * **Note**: on `KVCacheManagerV2` with chunked prefill, `block_ids` covers only the blocks allocated for the first chunk, because V2 allocates per chunk rather than for the whole prompt. The remaining blocks arrive as append-deltas in `RequestData.new_block_ids` on subsequent chunks. A connector that treats this callback as its only source of block ids will under-plan; drive off `build_connector_meta` instead.
  * **Note**: on `KVCacheManagerV2` under sliding-window attention, a block that the window has already passed holds no page, and is reported as `-1` (`BAD_PAGE_INDEX`) **in place** rather than being dropped from the list. This keeps each entry aligned with its block ordinal, so entry `i` always describes prompt tokens `[i * tokens_per_block, (i+1) * tokens_per_block)` and an append-delta over successive calls stays valid. Connectors must skip `-1` entries rather than treating them as page slots. The same applies to `RequestData.new_block_ids` and to `cache_block_ids` in `request_finished`.

* **`cancel_load(self, request: LlmRequest, start: int, end: int)`**
  * **Description**: Optional, with a no-op default. Tells the connector that the runtime will not consume KV it offered from `get_num_new_matched_tokens` for prompt tokens `[start, end)`, so any ownership taken for that range can be released. Offsets are absolute prompt positions, on the same scale as `num_computed_tokens`.
  * **When it fires**: only on `KVCacheManagerV2`, which asks during a speculative scheduling pass and resolves the answer later. Two things can happen in between, and both are reported here: the runtime may fail to allocate pages to cover the offer, in which case the request falls back to computing the prefix locally; or the request may be cancelled, time out or fail before it ever reaches a batch, in which case the whole offer is released. A third case, the local cache overtaking part of the offer because another request committed the same prefix, is handled by the same callback but cannot arise today, since a request's local match is fixed when its cache is created and only its own completed forward passes extend it.
  * **Caveat**: best-effort. For a synchronous load nothing has been transferred yet, so cancelling is exact. For `is_async=True` the transfer necessarily started inside `get_num_new_matched_tokens`, so it may already be in flight.

##### Serving a prefix on `KVCacheManagerV2`

V1 answers `get_num_new_matched_tokens` from C++ while the block manager holds its radix-tree mutex, so the local match and the query are atomic and the answer is consumed immediately. V2 has no such mutex, and its scheduling pass is speculative: a prepared request can still be dropped at the token budget, at resize, at multimodal alignment or at cross attention, and retried in a later iteration.

`get_num_new_matched_tokens` is still called **exactly once per request** on both managers, so a request that is asked and then deferred is not asked again when it comes back. What differs is that on V2 the runtime may resolve the answer in a later iteration than the one it asked in, and may by then be unable to honour part or all of it. That is what `cancel_load` reports.

#### 2. Worker Interface (`KvCacheConnectorWorker`)

These methods run on all workers (GPU processes) and interact with the actual GPU data.

* **`register_kv_caches(self, kv_cache_tensor: torch.Tensor)`**
  * **Description**: Called at initialization. Provides the worker with the GPU KV cache tensors.
  * **Arguments**: `kv_cache_tensor` is the underlying storage tensor for the KV cache.

* **`register_kv_cache_layout(self, layout: KvCacheLayout)`**
  * **Description**: Called at initialization **instead of** `register_kv_caches` when the KV cache manager is `KVCacheManagerV2`, whose memory cannot be expressed as one tensor: there is one slot address space per pool and one page-index space per layer group. The default implementation raises, so a connector that does not implement it can only run on V1.
  * **Arguments**: `layout` describes the byte ranges that repeat per page slot. Each `KvCacheLayerGroupLayout` carries a tuple of `KvCacheRegion`s, and the bytes for page slot `i` of a region live at `region.base + region.stride * i` for `region.size` bytes, or equivalently at `region.as_tensor()[i]`. Page indices arriving in `RequestData.new_block_ids_by_layer_group` are scoped to a layer group and index that group's regions.
  * **Why regions rather than a tensor**: because the ranges are described rather than implied, the same structure covers MLA (a pool simply has no `value` buffer), sliding-window and hybrid models (one layer group per window size), and non-uniform slots such as MiniMax-M3's index-K buffer sitting beside K/V, without any of them being a special case.

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

## Attention data parallelism

Set `enable_attention_dp=True` with a connector whose **scheduler and worker
classes both declare `supports_attention_dp = True`**. Existing connectors that
have not opted in are rejected before construction. No extra scheduler adapter
configuration is required: the executor creates a scheduler and worker on each
ADP rank, including rank 0.

Each adapter owns its request lookup, allocation feedback and worker metadata.
Connector callbacks and completion polling do not perform collectives between
ADP owners. The current ADP mapping has one attention worker per owner (model TP
ranks still cooperate for the non-attention computation). TP without ADP keeps
the rank-0 scheduler and waits for all attention shards to finish a transfer.

Adapters can use **one shared logical storage pool**. They must use distinct
worker endpoints and owner-scoped request/transfer IDs, while using common,
representation-compatible content keys for reusable KV. A local block ID is an
index into the registered local tensor, not a globally addressable page.
Do not partition the content namespace by ADP rank: distinguish model revision,
KV layout/dtype, block size, complete preceding prefix and cache salt instead.
Pool capacity and placement remain the backend's responsibility; enabling ADP
does not turn unused peer HBM into directly usable local attention memory.

The capability flag commits an implementation to these requirements:

* Scheduler constructors and callbacks work on nonzero ranks and never require
  all ADP owners to issue the same requests or call sequence.
* Workers consume only local metadata. Dummy requests do not appear in storage
  lookup, allocation feedback, metadata or request-finished callbacks. Empty
  metadata is valid, including during a dummy-only forward.
* `get_finished` reports only IDs previously provided to that worker and only
  after all transfers touching the corresponding local blocks have completed.
  Cancellation is deferred until those DMA users drain; the generic API does
  not provide a transport abort operation.
* Independently arriving writes/readers share content safely, with complete
  publication, compatible representations and backend pinning during reads.

Async loading may remove every real request from an owner's scheduled batch.
The executor attempts to add a compute dummy so other owners can advance while
the transfer proceeds. If the owner has no dummy capacity or sequence slot,
the existing ADP forward gate still defers the batch.

ADP supports the V1 single-primary-pool interface and V2 layer-group layouts.
PP=1, CP=1 and beam width 1 are required. V1 retains guaranteed-no-evict
scheduling; V2 retains its allocation and asynchronous-save lifetime handling.
Internal host/disk tiers and Mamba/hybrid state remain unsupported. Mooncake
requires V2 and rejects sliding-window layouts. Third-party presets must
explicitly opt in.

The built-in `mooncake-store` adapter shares one unsharded attention namespace
across ADP owners. Each owner opens its own store client and contributes its
configured segment to the common master. TP uses separate keys for each
attention shard and still requires all shards for a prefix hit. A disaggregated
DEP4 prefill / TEP8 decode deployment attaches the store connector to prefill;
decode can donate host memory while keeping its native KV transceiver. The
store does not convert attention shard layouts during prefill-to-decode handoff.

## Example Implementation

The file `examples/llm-api/llm_kv_cache_connector.py` provides a reference implementation of a **Persistent KV Cache**.

### Overview

This example implements a file-system based KV cache.
1. **Save**: When a request finishes or needs to be swapped out, its KV blocks are saved to disk as `.pt` files.
2. **Load**: When a new request arrives with the same prompt prefix, the connector identifies the cached files and loads them back into GPU memory, skipping re-computation.

### Implementation Details

* **Metadata**: The example defines a `PersistentKvCacheConnectorMetadata` dataclass containing lists of `(file_path, block_id)` tuples for both loading and saving. This simple structure allows the Scheduler to tell the Worker exactly which file corresponds to which GPU block index.

* **Hashing Strategy**: The `PersistentKvCacheConnectorLeader` uses SHA-256 over the complete prefix through each block and its cache salt. Keys are stable across independently started ADP processes.

* **Worker Logic**:
  * `start_load_kv`: Iterates through the load list provided in the metadata, loads the `.pt` file to CPU, and copies it to the specific `block_id` in the GPU tensor.
  * `wait_for_save`: Performs the reverse. It copies data from the GPU `block_id` to CPU and saves it to disk using `torch.save`.
    It writes a temporary file in the same directory and atomically publishes
    the completed file so concurrent owners cannot read a partial write.

For an ADP demonstration, point `TLLM_CONNECTOR_CACHE_FOLDER` at the same shared
filesystem directory on every rank. The `TLLM_` prefix matters: only `TRTLLM*`
and `TLLM*` variables are forwarded to spawned MPI workers, and the scheduler
reads this variable inside them. Use a dedicated directory for each model
revision and KV representation. The example supports unsharded attention KV
(single rank or ADP); it does not support sharded attention TP or chunked prefill.
This filesystem example demonstrates sharing, not elastic HBM placement or
production performance.

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
