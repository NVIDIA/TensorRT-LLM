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

* **`update_state_after_alloc(self, request: LlmRequest, block_ids: list[int])`**
  * **Description**: a callback to update internal state after KV cache blocks have been allocated for the prefill.
  * **Note**: on `KVCacheManagerV2` with chunked prefill, `block_ids` covers only the blocks allocated for the first chunk, because V2 allocates per chunk rather than for the whole prompt. The remaining blocks arrive as append-deltas in `RequestData.new_block_ids` on subsequent chunks. A connector that treats this callback as its only source of block ids will under-plan; drive off `build_connector_meta` instead.
  * **Note**: on `KVCacheManagerV2` under sliding-window attention, a block that the window has already passed holds no page, and is reported as `-1` (`BAD_PAGE_INDEX`) **in place** rather than being dropped from the list. This keeps each entry aligned with its block ordinal, so entry `i` always describes prompt tokens `[i * tokens_per_block, (i+1) * tokens_per_block)` and an append-delta over successive calls stays valid. Connectors must skip `-1` entries rather than treating them as page slots. The same applies to `RequestData.new_block_ids` and to `cache_block_ids` in `request_finished`.

* **`cancel_load(self, request: LlmRequest, start: int, end: int)`**
  * **Description**: Optional, with a no-op default. Tells the connector that the runtime will not consume KV it offered from `get_num_new_matched_tokens` for prompt tokens `[start, end)`, so any ownership taken for that range can be released. Offsets are absolute prompt positions, on the same scale as `num_computed_tokens`.
  * **When it fires**: only on `KVCacheManagerV2`, which asks during a speculative scheduling pass and resolves the answer later. Two things can happen in between, and both are reported here: the runtime may fail to allocate pages to cover the offer, in which case the request falls back to computing the prefix locally; or the request may be cancelled, time out or fail before it ever reaches a batch, in which case the whole offer is released. A third case -- the local cache overtaking part of the offer because another request committed the same prefix -- is handled by the same callback but cannot arise today, since a request's local match is fixed when its cache is created and only its own completed forward passes extend it.
  * **Caveat**: best-effort. For a synchronous load nothing has been transferred yet, so cancelling is exact. For `is_async=True` the transfer necessarily started inside `get_num_new_matched_tokens`, so it may already be in flight.

##### Serving a prefix on `KVCacheManagerV2`

V1 answers `get_num_new_matched_tokens` from C++ while the block manager holds its radix-tree mutex, so the local match and the query are atomic and the answer is consumed immediately. V2 has no such mutex, and its scheduling pass is speculative -- a prepared request can still be dropped at the token budget, at resize, at multimodal alignment or at cross attention, and retried in a later iteration.

The contract for connectors is unchanged, and in particular `get_num_new_matched_tokens` is still called **exactly once per request** on both managers -- a request that is asked and then deferred is not asked again when it comes back. What differs is that on V2 the runtime may resolve the answer in a later iteration than the one it asked in, and may by then be unable to honour part or all of it. That is what `cancel_load` reports.

#### 2. Worker Interface (`KvCacheConnectorWorker`)

These methods run on all workers (GPU processes) and interact with the actual GPU data.

* **`register_kv_caches(self, kv_cache_tensor: torch.Tensor)`**
  * **Description**: Called at initialization. Provides the worker with the GPU KV cache tensors.
  * **Arguments**: `kv_cache_tensor` is the underlying storage tensor for the KV cache.

* **`register_kv_cache_layout(self, layout: KvCacheLayout)`**
  * **Description**: Called at initialization **instead of** `register_kv_caches` when the KV cache manager is `KVCacheManagerV2`, whose memory cannot be expressed as one tensor: there is one slot address space per pool and one page-index space per layer group. The default implementation raises, so a connector that does not implement it can only run on V1.
  * **Arguments**: `layout` describes the byte ranges that repeat per page slot. Each `KvCacheLayerGroupLayout` carries a tuple of `KvCacheRegion`s, and the bytes for page slot `i` of a region live at `region.base + region.stride * i` for `region.size` bytes -- or equivalently at `region.as_tensor()[i]`. Page indices arriving in `RequestData.new_block_ids_by_layer_group` are scoped to a layer group and index that group's regions.
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

## Built-in Connectors

Named presets can be selected without naming a module or class:

```python
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig

kv_connector_config = KvCacheConnectorConfig(connector="mooncake-store")
```

The available presets are `lmcache`, `lmcache-mp`, `kvbm` and `mooncake-store`. The first three are external packages; `mooncake-store` ships with TensorRT-LLM and is described below.

### Mooncake distributed store (`mooncake-store`)

Publishes KV pages into a [Mooncake](https://github.com/kvcache-ai/Mooncake) store -- a shared CPU memory pool addressed by content -- so a prefix computed by one engine can be replayed by another. Regular block reuse cannot do this, because it never leaves the instance that computed the prefix.

This is a **different component** from the Mooncake transfer engine that the C++ cache transceiver uses for disaggregated prefill/decode handoff. That moves KV point to point between two known peers; this publishes pages into a pool that any peer can read. The two compose: a context server can write pages into the store and still hand off to a generation server over NIXL.

#### Requirements

* `KVCacheManagerV2` (`kv_cache_config.use_kv_cache_manager_v2: true`), since that is the manager that can describe its pools through `register_kv_cache_layout`.
* The Mooncake Python bindings: `pip install mooncake-transfer-engine`. These are installed in the release container; the source build of the C++ transfer engine does not provide them.
* A reachable Mooncake master (and metadata server, unless using `P2PHANDSHAKE`). See the [Mooncake documentation](https://kvcache-ai.github.io/Mooncake/). `trtllm-serve` can start one for a single engine; see below.
* GPU-only KV cache tiers: set `kv_cache_config.host_cache_size: 0` and `disk_cache_size: 0`. A page evicted to another tier has its GPU slot reassigned, which would invalidate the addresses registered with the store.

#### Configuration

Describe the pool in `kv_connector_config.mooncake_store` and `trtllm-serve` provisions it during bringup: it resolves the master, renders the client config, and exports `MOONCAKE_CONFIG_PATH` before the ranks that open store handles are spawned.

```yaml
kv_connector_config:
  connector: mooncake-store
  mooncake_store:
    master_server_address: 10.0.0.1:50051   # a master with its own lifetime
    protocol: rdma
    device_name: mlx5_0
    global_segment_size: 32GiB
    local_buffer_size: 1GiB
```

Replacing `master_server_address` with `launch_master: true` makes the server start a `mooncake_master` itself and use it, so a single-instance deployment needs nothing prepared outside `trtllm-serve`. **That master lives and dies with the server**, which makes it wrong for anything else: several engines that should share one pool would each get their own, and a pool meant to survive a restart cannot be owned by the thing restarting. Those deployments run a master with its own lifetime and name it in `master_server_address`.

`TRTLLM_MOONCAKE_MASTER_BINARY` overrides the binary a launched master runs, and `TRTLLM_MOONCAKE_MASTER_TIMEOUT` (default 60s) how long startup waits for any master to accept connections -- reaching a master that is not there otherwise fails inside every rank after the model has loaded. Set `TRTLLM_MOONCAKE_RUN_DIR` to keep the generated client config and the master's log, which are otherwise in a temporary directory removed at shutdown.

Topology can equally come from a JSON file named by `MOONCAKE_CONFIG_PATH`, using the same schema as the vLLM Mooncake store connector so one deployment can point both engines at the same pool:

```json
{
  "metadata_server": "http://127.0.0.1:8080/metadata",
  "master_server_address": "127.0.0.1:50051",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "global_segment_size": "32GiB",
  "local_buffer_size": "1GiB"
}
```

An inherited `MOONCAKE_CONFIG_PATH` wins over `mooncake_store` and is logged as doing so, so an orchestrator that already provisions the pool -- as the SLURM benchmark harness does -- keeps working unchanged.

Three further settings are TensorRT-LLM's rather than Mooncake's, and stay in the environment because they are per process rather than per pool:

| Variable | Default | Meaning |
|---|---|---|
| `TRTLLM_MOONCAKE_STORE_ROLE` | `both` | `producer` writes only, `consumer` reads only, `both` does both. |
| `TRTLLM_MOONCAKE_STORE_PREFIX` | `trtllm` | Leading component of every key, for isolating deployments that share a pool. |
| `TRTLLM_MOONCAKE_STORE_MODEL_KEY` | model directory basename | Identity keys are namespaced by. Two engines share cache only when they agree on it, so the default is the basename rather than the full path -- the same checkpoint is routinely mounted elsewhere on another host, which is exactly what sharing is for. |

In a disaggregated deployment, run context servers as `both` and leave generation servers unconfigured. Generated tokens are rarely a reused prefix, so writing them costs bandwidth for no hit rate.

#### Partial block reuse is forced off

`kv_cache_config.enable_partial_reuse` is set to `false` when this connector is configured, with a warning, whether or not it was requested explicitly. It defaults to `true`, so most deployments will see that warning.

The store is addressed by whole blocks. The connector is handed the device match as `num_computed_tokens` and offers only blocks beyond it, but it can only resume from a block boundary -- so when the device match ends mid-block, it declines the lookup and the store is not consulted at all. Partial reuse is precisely what puts the match off a boundary, which means it trades part of one block of device reuse for every stored block of the remaining prefix. Measured on MiniMax-M3, leaving it enabled declined 97.2% of lookups and left actual prompt cache read at 35% against a 96% ceiling; forcing it off raised that to 94% and roughly doubled throughput.

#### How it keys pages

`KVCacheManagerV2` reports `RequestData.block_hashes` empty, so the connector derives block identity itself: a blake2b chain where each block's hash covers its own tokens *and* every token before it, seeded by the request's `cache_salt`. A key is `<prefix>/<model>/w<world size>r<rank>/lg<layer group>/t<tokens per block>b<bytes per page>/<block hash>`. The namespace pins down everything that would make the stored bytes mean something different, so a mismatched shard count, layer group or page geometry reads as a cache miss rather than as garbage.

The value for one key is the concatenation of that layer group's regions for one page slot, handed to Mooncake's multi-buffer batch APIs as a list of `(address, size)` pairs.

#### Transfer behavior

* **Loads are synchronous**, performed in `start_load_kv` before the forward pass. A failed load raises: the runtime has already counted those tokens as computed, so a partial load is a wrong answer rather than a slow one.
* **Saves are asynchronous**, handed to a background thread behind a CUDA event recorded on the forward stream. The pages are only complete once the pass that wrote them retires, and blocking the executor loop on an RDMA write is the cost the store exists to avoid. The leader reports such requests as saving asynchronously, so their pages stay pinned until `get_finished` confirms the writes landed. A dropped save is logged rather than raised -- it only costs a future cache miss.
* Pages the store already holds are skipped, so several ranks or instances converging on the same prefix write it once.

#### Unsupported configurations

These are rejected at startup, before any request is admitted:

| Configuration | Reason |
|---|---|
| Context parallelism | A rank holds a slice of the sequence rather than whole blocks of it, so one key would name different bytes on different ranks. |
| Sliding-window attention / VSWA | A page's validity depends on where the window sits, which is a property of the request that read it rather than of the tokens it holds. |
| MiniMax-M3 with `sparse_disable_index_value: false` | The index-V cache is a plain tensor outside the paged pools, so a replayed prefix would pair stored index-K with stale index-V. Disaggregated serving applies the same restriction. |
| Pipeline parallelism | Untested rather than unsound. Use tensor parallelism. |
| `KVCacheManagerV1` | Identity here is a per-layer-group hash chain; V1 supplies real block hashes over a single flat block space. |

Beam search, attention data parallelism, non-GPU cache tiers and Mamba caches are rejected for all connectors by the executor.

#### Example

`examples/llm-api/configs/trtllm_mooncake_store_connector_extra.yaml` is a starting point for `trtllm-serve`.

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
