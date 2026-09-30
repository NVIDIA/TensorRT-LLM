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
* The Mooncake Python bindings: `pip install mooncake-transfer-engine`. These are installed in the release container; the source build of the C++ transfer engine does not provide them.
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
```

`trtllm-serve` then reads the manifest, adds this server's settings and this node's detected RDMA devices, renders the Mooncake client config, and exports `MOONCAKE_CONFIG_PATH` before the ranks that open store handles are spawned.

Only `pool` is required. `pool` also accepts a bare `host:port` for joining a master run without this CLI, in which case the pool-wide settings take their defaults and keeping them consistent is the deployment's problem.

Sizes are written as binary units (`160GiB`) or byte counts. `GB` and friends are **refused** in anything both engines may read: this parser scales them by 1000 and vLLM's Mooncake parser by 1024, so `80GB` would name two different segments. The rendered client config always holds resolved integers for the same reason.

`master_timeout` (default 60s) is how long a server waits for the manifest to appear and the master to accept connections. Without that wait, a master that is not there yet fails inside every rank after the model has loaded. `run_dir` keeps the generated client config and the segment records, which are otherwise in a temporary directory removed at shutdown. The master command takes `--timeout` for the same wait and `--binary` to name the executable.

There is no environment override for any of this. `MOONCAKE_CONFIG_PATH` is the one variable the connector reads, and it belongs to Mooncake rather than to TensorRT-LLM.

#### Servers whose ranks the launcher starts

Provisioning happens in the server process and reaches the ranks that open store handles by exporting `MOONCAKE_CONFIG_PATH` for them to inherit. That holds when the LLM constructor spawns them. It does not when the launcher starts one task per rank, as `trtllm-llmapi-launch` under a scheduler does, because those ranks were already running.

Setting `run_dir` covers that case: the rendered config is read back from `<run_dir>/mooncake.json` by any rank that inherited no path, so every rank of a multi-GPU server joins the pool its own leader provisioned. It has to be a directory all of that server's ranks see, which under a scheduler means a shared filesystem, and **one per server rather than one per job**, since two servers sharing it render one client config between them:

```yaml
mooncake_store:
  pool: file:///shared/run/pool.json
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

`KVCacheManagerV2` reports `RequestData.block_hashes` empty, so the connector derives block identity itself: a blake2b chain where each block's hash covers its own tokens *and* every token before it, seeded by the request's `cache_salt`. A key is `<prefix>/<model>/w<attention shard count>r<attention shard rank>/lg<layer group>/t<tokens per block>b<bytes per page>/<block hash>`. Under ADP all owners use `w1r0`, since each holds complete attention KV; the MPI owner rank remains local transfer state. The namespace pins down everything that would make the stored bytes mean something different, so a mismatched shard count, layer group or page geometry reads as a cache miss rather than as garbage.

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

Beam search and Mamba caches are rejected for all connectors by the executor, as are non-GPU cache tiers for any connector that registers pages. Attention DP is supported through a local scheduler adapter on every owner.

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
