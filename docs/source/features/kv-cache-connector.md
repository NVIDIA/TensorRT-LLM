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
  * **Arguments**: `scheduler_output` contains information about new requests, blocks allocated, current request states, and the cumulative `RequestData.block_hashes` chain. `block_hashes` is read directly from each KV cache block's stored hash, which the KV cache manager commits as soon as a block becomes full -- the value matches the hash that KV cache events will subsequently emit for the same block. The chain only covers beam 0; the executor rejects `kv_connector_config` at startup when `max_beam_width > 1`, so connectors may assume beam-width-1 inputs.
  * **Returns**: An arbitrary metadata object (picklable) that describes the tasks for the workers. Under TP it is broadcast to all workers. Under ADP it is bound only to the local worker; `scheduler_output.attention_dp_rank` identifies the owner of its block IDs.

* **`get_num_new_matched_tokens(self, request: LlmRequest, num_computed_tokens: int) -> tuple[int, bool]`**
  * **Description**: Called when a new request arrives. It checks to see if any KV cache can be loaded from an external KV store.
  * **Returns**: A tuple `(num_tokens, is_async)`. `num_tokens` is the number of tokens found in the external cache. `is_async` indicates if the loading will happen asynchronously (background) or requires blocking.

* **`request_finished(self, request: LlmRequest, cache_block_ids: list[int]) -> bool`**
  * **Description**: Called when a request completes generation.
  * **Returns**: A boolean indicating if an asynchronous save operation is underway. If `True`, the system waits for the operation to complete before releasing the KV cache blocks.

* **`update_state_after_alloc(self, request: LlmRequest, block_ids: list[int])`**
  * **Description**: a callback to update internal state after KV cache blocks have been allocated for the prefill.

#### 2. Worker Interface (`KvCacheConnectorWorker`)

These methods run on all workers (GPU processes) and interact with the actual GPU data.

* **`register_kv_caches(self, kv_cache_tensor: torch.Tensor)`**
  * **Description**: Called at initialization. Provides the worker with the GPU KV cache tensors.
  * **Arguments**: `kv_cache_tensor` is the underlying storage tensor for the KV cache.

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

Initial support retains the V1, single-primary-pool contract and existing
restrictions: PP=1, CP=1, beam width 1, guaranteed-no-evict scheduling, no VSWA,
no internal host offload and no Mamba/hybrid state. V2 and heterogeneous
attention shard layouts require separate integration work. Third-party
connector presets are not automatically opted in by this change.

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

For an ADP demonstration, point `CONNECTOR_CACHE_FOLDER` at the same shared
filesystem directory on every rank. Use a dedicated directory for each model
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
